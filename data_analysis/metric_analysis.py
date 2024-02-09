import numpy as np
import pandas as pd
import plotly.express as px
import sys
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional
from logging import Logger, getLogger, StreamHandler, FileHandler, Formatter

from data_types import Metrics, Recording, RecordingEnvironment, concat_metrics
from step_detection import DataHandler, StepDetector, ParsedRecording


class AnalysisController:
    def __init__(self, model: Recording, window_duration=0.2, logger: Optional[Logger]=None, log_file='latest.log', **kwargs) -> None:
        self.logger = self.init_logger(log_file) if logger is None else logger
        self.model = ParsedRecording.from_recording(model, logger=self.logger)
        weights = self.model.get_frequency_weights(window_duration, plot=False)
        noise = self.model.get_noise()
        # step_model = self.model.get_step_model(window_duration, plot_model=False, plot_steps=False)
        self._detector = StepDetector(
            fs=self.model.env.fs,
            window_duration=window_duration,
            noise_profile=noise,
            # step_model=step_model,
            freq_weights=weights,
            logger=self.logger,
            **kwargs
        )

    @staticmethod
    def init_logger(log_file: Optional[str] = None) -> Logger:
        logger = getLogger()
        logger.setLevel("DEBUG")
        handler = FileHandler(log_file, mode='w+') if log_file else StreamHandler(sys.stdout)
        handler.setFormatter(Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def get_metrics(self, datasets: List[Recording], abort_on_limit=False, plot_title=None, plot_dist=False, plot_vs_env=False, plot_signals=False) -> Tuple[Metrics, Metrics, pd.DataFrame]:
        """Analyzes a sequence of recordings and returns metrics"""
        if not len(datasets):
            raise ValueError("No datasets provided")
        if not all(isinstance(data, Recording) for data in datasets):
            raise TypeError("All datasets must be of type Recording")
        results = []
        for data in datasets:
            try:
                results.append(self.get_recording_metrics(data, abort_on_limit, plot=plot_signals))
            except Exception as e:
                self.logger.error(f"Failed to get metrics for {data.filepath}: {e}")
                raise e
        measured_sets, source_of_truth_sets, algorithm_errors = zip(*results)
        measured = concat_metrics(measured_sets)
        source_of_truth = concat_metrics(source_of_truth_sets)
        algorithm_error = pd.concat(algorithm_errors)
        if plot_dist:
            err = measured.error(source_of_truth)
            err.index = [d.filepath for d in datasets]
            melted_err = err.melt(value_name="error", var_name="metric", ignore_index=False)
            melted_err.dropna(inplace=True)
            melted_err.reset_index(inplace=True)
            fig = px.box(
                melted_err,
                x="metric",
                y="error",
                hover_data="index",
                points="all",
                title=plot_title if plot_title else "Metric Error Distribution",
                labels={"error": "Error (%)", "metric": "Metric"}
            )
            fig.show()
        if plot_vs_env:
            err = measured.error(source_of_truth)
            err.index = [d.filepath for d in datasets]
            self._plot_error_vs_env(err, datasets, plot_title)
        return measured, source_of_truth, algorithm_error

    def _plot_error_vs_env(self, err, datasets: List[Recording], plot_title=None, show=True):
        """Plot showing grouped box plots of average metric error, grouped by each recording environment variables"""
        varied_vars = self._get_varied_env_vars(datasets)
        env_df = self._get_env_df(datasets)
        df = pd.concat([env_df, err], axis=1)
        fig = make_subplots(rows=len(varied_vars), cols=1, subplot_titles=[f"{key} comparison" for key in varied_vars])
        fig.update_layout(title=plot_title if plot_title else "Metric Error Distribution")
        for i, (env_var, values) in enumerate(varied_vars.items(), start=1):
            for value in values:
                subset = df[df[env_var] == value]
                subset = subset.drop(env_df.columns, axis=1)
                subset = subset.melt(value_name="error", var_name="metric")
                fig.add_box(x=subset["metric"], y=subset["error"], name=f"{env_var} = {value}", boxpoints="all", row=i, col=1)
        if show:
            fig.show()
        return fig

    @staticmethod
    def _get_env_df(datasets: List[Recording]) -> pd.DataFrame:
        """Returns a dataframe of environmental variables. Each row corresponds to a recording. Each column corresponds to an environmental variable."""
        env_dicts = {d.filepath: d.env.to_dict() for d in datasets}
        env_df = pd.DataFrame(env_dicts).T
        return env_df

    @staticmethod
    def _get_varied_env_vars(datasets: List[Recording], exclude=['notes']) -> dict[str, list[str]]:
        """Returns a dictionary of environmental variables that vary across datasets"""
        env_vars = {key: [] for key in RecordingEnvironment.__annotations__}
        for data in datasets:
            for key, value in data.env.to_dict().items():
                if value is not None and value not in env_vars[key] and key not in exclude:
                    env_vars[key].append(value)
        varied_vars = {key: value for key, value in env_vars.items() if len(value) > 1}
        return varied_vars

    def get_recording_metrics(self, data: Recording, abort_on_limit=False, plot=False) -> Tuple[Metrics, Metrics, pd.DataFrame]:
        """Analyzes a recording and returns metrics"""
        if data.env.fs != self.model.env.fs:
            raise ValueError(f"Recording fs ({data.env.fs}) does not match model fs ({self.model.env.fs})")
        correct_steps = self._get_true_step_timestamps(data)
        abort_limit = None if not abort_on_limit else len(correct_steps)
        step_groups = self._detector.get_step_groups(np.array(data.ts), abort_limit, plot, truth=correct_steps)
        if len(step_groups):
            self.logger.info(f"Found {len(step_groups)} step groups in {data.filepath}")
        predicted_steps = np.concatenate(step_groups) if len(step_groups) else []
        measured = Metrics(*step_groups)
        source_of_truth = Metrics(correct_steps)
        algorithm_error = self._get_algorithm_error(predicted_steps, correct_steps)
        return measured, source_of_truth, algorithm_error

    @staticmethod
    def _get_true_step_timestamps(data: Recording) -> List[float]:
        return [event.timestamp for event in data.events if event.category == 'step']

    @staticmethod
    def _get_algorithm_error(measured_times: np.ndarray, correct_times: np.ndarray) -> pd.DataFrame:
        """
        Calculates the algorithm error of a recording. There are three types of errors:
        - Incorrect measurements: The algorithm found a step when there was none (False Positive)
        - Missed steps: The algorithm missed a step (False Negative)
        - Measurement error: The algorithm found a step correctly, but at the wrong time

        Parameters
        ----------
        measured_times : np.ndarray
            List of timestamps where the algorithm found steps
        correct_times : np.ndarray
            List of step timestamps from the source of truth
        """
        if not len(measured_times):
            return pd.DataFrame({
                "error": [np.nan],
                "stderr": [np.nan],
                "incorrect": [0],
                "missed": [len(correct_times)]
            })
        missed_steps = 0
        measurement_errors = {}
        for step_stamp in correct_times:
            possible_errors = np.abs(measured_times - step_stamp)
            best_measurement = np.argmin(possible_errors)
            # If the measurement is already the best one for another step, it means we missed this step
            if best_measurement in measurement_errors:
                missed_steps += 1
            else:
                measurement_errors[best_measurement] = possible_errors[best_measurement]
        # TODO: If there is a missed step and a false positive, it will be matched and the error will be large
        incorrect_measurements = len(measured_times) - len(measurement_errors)
        errors = list(measurement_errors.values())

        return pd.DataFrame({
            "error": [np.mean(errors)],
            "stderr": [np.std(errors)],
            "incorrect": [incorrect_measurements],
            "missed": [missed_steps]
        })




if __name__ == "__main__":
    # DataHandler().plot(walk_speed='normal', user='ron', footwear='socks', wall_radius=1.89)

    model_data = Recording.from_file('datasets/2023-11-09_18-42-33.yaml')
    params = {'window_duration': 0.09610293596218258, 'min_signal': 0.9255774291395902, 'min_step_delta': 0.3473642689296651, 'max_step_delta': 1.3466980759130174, 'confirm_coefs': [0.06441405583227566, 1.721246167409189, 0.34013823260166054, 0.007652060221684964], 'unconfirm_coefs': [0.027288371317633953, 0.38371428061888646, 1.7425691202573512, 0.781486599075381], 'reset_coefs': [0.7469581128236926, 1.291174847574172, 1.786924086990361, 1.4176149229234183]}
    controller = AnalysisController(model_data, **params)
    datasets = DataHandler().get(user='ron', location='Aarons Studio')
    print(controller.get_metrics(datasets, plot_dist=True, plot_title=str(params)))

    # bad_recordings = [
    #     'datasets/2023-11-09_18-54-28.yaml',
    #     # 'datasets/2023-11-09_18-42-33.yaml',
    #     'datasets/2023-11-09_18-44-35.yaml',
    # ]

    # datasets = [Recording.from_file(f) for f in bad_recordings]
    # controller.get_metrics(datasets, plot_signals=True)
