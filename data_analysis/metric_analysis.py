import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Tuple

from data_types import Metrics, Recording, RecordingEnvironment, concat_metrics
from step_detection import DataHandler, StepDetector, ParsedRecording


class AnalysisController:
    def __init__(self, model: Recording, window_duration=0.2) -> None:
        self.model = ParsedRecording.from_recording(model)
        weights = self.model.get_frequency_weights(window_duration, plot=False)
        noise = self.model.get_noise()
        step_model = self.model.get_step_model(window_duration, plot_model=False, plot_steps=False)
        self._detector = StepDetector(
            fs=self.model.env.fs,
            window_duration=window_duration,
            noise_profile=noise,
            # step_model=step_model,
            freq_weights=weights
        )

    def get_metrics(self, datasets: List[Recording], plot_dist=False, plot_vs_env=False, plot_signals=False) -> Tuple[Metrics, Metrics, pd.DataFrame]:
        """Analyzes a sequence of recordings and returns metrics"""
        if not len(datasets):
            raise ValueError("No datasets provided")
        if not all(isinstance(data, Recording) for data in datasets):
            raise ValueError("All datasets must be of type Recording")
        results = [self.get_recording_metrics(data, plot=plot_signals) for data in datasets]
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
            fig = px.box(melted_err, x="metric", y="error", hover_data="index", points="all", title="Metric Error Distribution", labels={"error": "Error (%)", "metric": "Metric"})
            fig.show()
        if plot_vs_env:
            err = measured.error(source_of_truth)
            err.index = [d.filepath for d in datasets]
            self._plot_error_vs_env(err, datasets)
        return measured, source_of_truth, algorithm_error

    def _plot_error_vs_env(self, err, datasets: List[Recording], show=True):
        """Plot showing grouped box plots of average metric error, grouped by each recording environment variables"""
        varied_vars = self._get_varied_env_vars(datasets)
        env_df = self._get_env_df(datasets)
        df = pd.concat([env_df, err], axis=1)
        fig = make_subplots(rows=len(varied_vars), cols=1, subplot_titles=[f"{key} comparison" for key in varied_vars])
        fig.update_layout(title="Metric Error Distribution")
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

    def get_recording_metrics(self, data: Recording, plot=False) -> Tuple[Metrics, Metrics, pd.DataFrame]:
        """Analyzes a recording and returns metrics"""
        step_groups = self._detector.get_step_groups(data.ts, plot, truth=self._get_step_timestamps(data))
        predicted_steps = np.concatenate(step_groups) if len(step_groups) else []
        measured = Metrics(*step_groups)
        correct_steps = self._get_step_timestamps(data)
        algorithm_error = self._get_algorithm_error(predicted_steps, correct_steps)
        source_of_truth = Metrics(correct_steps)
        return measured, source_of_truth, algorithm_error

    @staticmethod
    def _get_step_timestamps(data: Recording) -> List[float]:
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

    @staticmethod
    def optimize_threshold_weights(datasets: List[Recording], plot=False, **kwargs) -> Tuple[float, float]:
        raise NotImplementedError()



if __name__ == "__main__":
    # DataHandler().plot(walk_speed='normal', user='ron', footwear='socks', wall_radius=1.89)

    model_data = Recording.from_file('datasets/2023-11-09_18-42-33.yaml')
    controller = AnalysisController(model_data)
    datasets = DataHandler().get(user='ron', location='Aarons Studio')
    controller.get_metrics(datasets, plot_dist=True)

    bad_recordings = [
        'datasets/2023-11-09_18-54-28.yaml',
        # 'datasets/2023-11-09_18-42-33.yaml',
        'datasets/2023-11-09_18-44-35.yaml',
    ]

    datasets = [Recording.from_file(f) for f in bad_recordings]
    controller.get_metrics(datasets, plot_signals=True)
