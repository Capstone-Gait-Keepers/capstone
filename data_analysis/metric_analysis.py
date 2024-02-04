import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Tuple

from data_types import Metrics, Recording, RecordingEnvironment, concat_metrics
from step_detection import DataHandler, StepDetector, ParsedRecording


class MetricAnalyzer:
    def __init__(self, step_detector: StepDetector) -> None:
        self._detector = step_detector

    def analyze(self, *vibes: np.ndarray, **kwargs) -> Metrics:
        """
        Analyzes a recording and returns a dictionary of metrics
        """
        step_groups = []
        for ts in vibes:
            step_groups.extend(self._detector.get_step_groups(ts, **kwargs))
        if not len(step_groups):
            raise ValueError("No valid step sections found")
        return Metrics(*step_groups)


class AnalysisController(MetricAnalyzer):
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
        super().__init__(self._detector)

    def get_metrics(self, *datasets: Recording, plot=True, **kwargs) -> Tuple[Metrics, Metrics, pd.DataFrame]:
        """Analyzes a sequence of recordings and returns metrics"""
        if not len(datasets):
            raise ValueError("No datasets provided")
        measured_sets, source_of_truth_sets, algorithm_errors = [], [], []
        for data in datasets:
            measured, source_of_truth, algorithm_error = self._get_metrics(data, **kwargs)
            measured_sets.append(measured)
            source_of_truth_sets.append(source_of_truth)
            algorithm_errors.append(algorithm_error)
        measured = concat_metrics(measured_sets)
        source_of_truth = concat_metrics(source_of_truth_sets)
        algorithm_error = pd.concat(algorithm_errors)
        if plot:
            err = measured.error(source_of_truth)
            err.index = [d.filepath for d in datasets]
            err_series = err.melt(value_name="error", var_name="metric")
            err_series.dropna(inplace=True)
            fig = px.box(err_series, x="metric", y="error", points="all", title="Metric Error Distribution", labels={"error": "Error (%)", "metric": "Metric"})
            fig.show()
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

    def _get_metrics(self, data: Recording, **kwargs):
        """Analyzes a recording and returns metrics"""
        measured = Metrics()
        predicted_steps = []
        step_groups = self._detector.get_step_groups(data.ts, **kwargs)
        if len(step_groups):
            measured = Metrics(*step_groups)
            predicted_steps = np.concatenate(step_groups)
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
    model_data = Recording.from_file('datasets/2023-11-09_18-42-33.yaml')
    controller = AnalysisController(model_data)
    # data = Recording.from_file('datasets/2023-11-09_18-46-43.yaml')
    # controller.get_metric_error(data, plot=True)

    # DataHandler().plot(walk_speed='normal', user='ron', footwear='socks', wall_radius=1.89)

    # walk_type='normal', user='ron', wall_radius=1.89 -> 5.6% cadence, fucked STGA
    datasets = DataHandler().get(user='ron', location='Aarons Studio')
    measured, truth, alg_err = controller.get_metrics(*datasets)
