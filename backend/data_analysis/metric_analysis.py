import numpy as np
import pandas as pd
import plotly.express as px
import sys
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from typing import List, Tuple, Optional, Iterable
from logging import Logger, getLogger, StreamHandler, FileHandler, Formatter

from data_types import Metrics, Recording, RecordingEnvironment, SensorType, get_optimal_analysis_params
from step_detection import DataHandler, StepDetector, TimeSeriesProcessor


class RecordingProcessor:
    """Collection of methods for processing recordings"""
    def __init__(self) -> None: ...

    @staticmethod
    def get_steps_from_truth(rec: Recording, step_duration=0.4, shift_percent=0.2, align_peak=True, plot=False) -> List[np.ndarray]:
        """
        Uses the source of truth to parse the accelerometer data pertaining to steps
        """
        proc = TimeSeriesProcessor(rec.env.fs)
        offset = int(shift_percent * rec.env.fs / 2)
        window_size = proc.timestamp_to_index(step_duration)
        step_measurements = []
        for event in rec.events:
            if event.category == 'step':
                start = proc.timestamp_to_index(event.timestamp) - offset
                step_data = rec.ts[start : start+window_size]
                if align_peak:
                    energy = proc.get_energy(step_data, step_duration / 5)
                    start += np.argmax(energy) - offset
                    step_data = rec.ts[start : start+window_size]
                if start + window_size > len(rec.ts):
                    continue
                if len(step_data) != window_size:
                    raise ValueError(f"Step data is the wrong size: {len(step_data)} != {window_size}")
                step_measurements.append(step_data)
        if plot:
            fig = go.Figure()
            fig.update_layout(title="Step Data")
            timestamps = np.linspace(0, step_duration, len(step_measurements[0]))
            for step in step_measurements:
                fig.add_scatter(x=timestamps, y=proc.get_energy(step, step_duration/2))
            fig.show()
        return step_measurements

    @staticmethod
    def get_noise(rec: Recording, plot=False) -> np.ndarray:
        """
        Find the noise floor of the accelerometer data
        """
        proc = TimeSeriesProcessor(rec.env.fs)
        first_event = rec.events[0].timestamp
        noise = rec.ts[:proc.timestamp_to_index(first_event)]
        if plot:
            timestamps = np.linspace(0, len(noise) / rec.fs, len(noise))
            fig = go.Figure()
            fig.update_layout(title="Noise", showlegend=False)
            fig.add_scatter(x=timestamps, y=noise)
            fig.show()
        return noise

    @staticmethod
    def get_frequency_weights(rec: Recording, window_duration=0.2, plot=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the dominant frequencies pertaining to steps
        """
        proc = TimeSeriesProcessor(rec.env.fs)
        step_data = RecordingProcessor.get_steps_from_truth(rec, window_duration*2, align_peak=True, plot=plot)
        freqs = proc.get_window_fft_freqs(window_duration)
        amp_per_step_freq_time = []

        for step in step_data:
            if not len(step):
                raise ValueError(f"Step data is empty: {step_data}")
            amps = proc.rolling_window_fft(step, window_duration, stride=1, ignore_dc=True)
            amp_per_step_freq_time.append(amps)
        amp_per_step_freq_time = np.asarray(amp_per_step_freq_time)
        amp_per_step_freq = np.mean(amp_per_step_freq_time, axis=-1)
        amp_per_freq = np.mean(amp_per_step_freq, axis=0)

        noise = RecordingProcessor.get_noise(rec)
        noise_amp_per_freq_time = proc.rolling_window_fft(noise, window_duration, stride=1, ignore_dc=True)
        noise_amp_per_freq = np.mean(noise_amp_per_freq_time, axis=-1)
        freq_weights = amp_per_freq / noise_amp_per_freq # Noise of zero?
        freq_weights /= np.max(freq_weights)

        if plot:
            timestamps = np.linspace(0, window_duration, len(amp_per_step_freq_time[0]))
            num_rows = int(np.ceil(np.sqrt(len(step_data))))
            fig = make_subplots(rows=num_rows, cols=num_rows, column_titles=['Step Heat Maps (Freq vs Time)'])
            for i, amps in enumerate(amp_per_step_freq_time, start=1):
                fig.add_heatmap(x=timestamps, y=freqs, z=amps, row=(i // num_rows) + 1, col=(i % num_rows) + 1)
            fig.show()
            fig = go.Figure()
            fig.update_layout(title="Amplitude vs Frequency For Each Step", showlegend=False)
            for step_amp_per_freq in amp_per_step_freq:
                fig.add_scatter(x=freqs, y=step_amp_per_freq)
            fig.show()
            fig = go.Figure()
            fig.update_layout(title="Noise Amplitude vs Frequency", showlegend=False)
            fig.add_scatter(x=freqs, y=noise_amp_per_freq)
            fig.show()
            fig = go.Figure()
            fig.update_layout(title="Amplitude vs Frequency (Average of all steps)", showlegend=False)
            fig.add_scatter(x=freqs, y=freq_weights)
            fig.show()
        return freq_weights

    @staticmethod
    def get_step_model(rec: Recording, window_duration=0.2, plot_model=False, plot_steps=False) -> np.ndarray:
        """Creates a model of step energy vs time"""
        proc = TimeSeriesProcessor(rec.env.fs)
        step_data = RecordingProcessor.get_steps_from_truth(rec)
        energy_per_step = []
        for step in step_data:
            # TODO: Use weights from get_frequency_weights
            energy = proc.get_energy(step, window_duration)
            energy_per_step.append(energy / np.max(energy))
        step_model = np.mean(energy_per_step, axis=0)
        timestamps = np.linspace(0, window_duration, len(energy_per_step[0]))
        if plot_steps:
            fig = go.Figure()
            fig.update_layout(title="Energy vs Time")
            for energy in energy_per_step:
                fig.add_scatter(x=timestamps, y=energy)
            fig.update_xaxes(title_text="Time (s)")
            fig.update_yaxes(title_text="Energy")
            fig.show()
        if plot_model:
            fig = go.Figure()
            fig.update_layout(title="Model Step", showlegend=False)
            fig.add_scatter(x=timestamps, y=step_model)
            fig.update_xaxes(title_text="Time (s)")
            fig.update_yaxes(title_text="Energy")
            fig.show()
        return step_model



class AnalysisController:
    def __init__(self, model: Recording=None, fs=None, noise_amp=None, window_duration=0.2, logger: Optional[Logger]=None, log_file='latest.log', **kwargs) -> None:
        self.logger = self.init_logger(log_file) if logger is None else logger
        if model:
            noise = RecordingProcessor.get_noise(model)
            weights = RecordingProcessor.get_frequency_weights(model, window_duration, plot=False)
            # step_model = RecordingProcessor.get_step_model(model, window_duration, plot_model=False, plot_steps=False)
            self._detector = StepDetector(
                fs=model.env.fs,
                window_duration=window_duration,
                noise_profile=noise,
                # step_model=step_model,
                # freq_weights=weights,
                logger=self.logger,
                **kwargs
            )
            self.fs = model.env.fs
        elif fs is None or noise_amp is None:
            raise ValueError("Must provide either a model or a sampling frequency (fs) and a noise amplitude (noise_amp)")
        else:
            self.fs = fs
            self._detector = StepDetector(fs, window_duration, noise_profile=np.random.rand(10) * noise_amp, logger=self.logger, **kwargs)

    @staticmethod
    def init_logger(log_file: Optional[str] = None) -> Logger:
        logger = getLogger()
        logger.setLevel("DEBUG")
        handler = FileHandler(log_file, mode='w+') if log_file else StreamHandler(sys.stdout)
        handler.setFormatter(Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def get_metric_error(self, datasets: Iterable[Recording], append_env=False, plot_signals=False, plot_title=None, plot_dist=False, plot_vs_env=False) -> pd.DataFrame:
        """Calculates the metric error based on a set of datasets and returns a dataframe"""
        results = []
        recordings = []
        for data in datasets:
            recordings.append(data)
            result = self.get_recording_metrics(data, plot_signals)
            results.append(result)
        measured, truth, _ = self._parse_metric_results(results)
        err = measured.error(truth)
        err.index = [rec.filepath for rec in recordings]
        if plot_dist:
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
            self._plot_error_vs_env(err, recordings, plot_title)
        if append_env:
            env_df = self.get_env_df(recordings)
            err = pd.concat([env_df, err], axis=1)
        return err

    def get_metrics(self, datasets: Iterable[Recording], plot_signals=False) -> Tuple[Metrics, Metrics, pd.DataFrame]:
        """Analyzes a sequence of recordings and returns metrics"""
        results = []
        for data in datasets:
            result = self.get_recording_metrics(data, plot_signals)
            results.append(result)
        measured, source_of_truth, algorithm_error = self._parse_metric_results(results)
        self.logger.debug(f"Finished analyzing all {measured.recordings} datasets")
        return measured, source_of_truth, algorithm_error

    def _parse_metric_results(self, results: Iterable[Tuple[Metrics, Metrics, pd.DataFrame]]) -> Tuple[Metrics, Metrics, pd.DataFrame]:
        """Parses the results of a sequence of recording metrics"""
        r = list(results)
        if not len(r):
            raise ValueError("No datasets provided")
        measured_sets, source_of_truth_sets, algorithm_errors = zip(*r)
        measured = sum(measured_sets)
        source_of_truth = sum(source_of_truth_sets)
        algorithm_error = pd.concat(algorithm_errors, ignore_index=True)
        return measured, source_of_truth, algorithm_error

    def _plot_error_vs_env(self, err, datasets: List[Recording], plot_title=None, show=True):
        """Plot showing grouped box plots of average metric error, grouped by each recording environment variables"""
        varied_vars = self._get_varied_env_vars(datasets)
        env_df = self.get_env_df(datasets)
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
    def get_env_df(datasets: List[Recording]) -> pd.DataFrame:
        """Returns a dataframe of environmental variables. Each row corresponds to a recording. Each column corresponds to an environmental variable."""
        env_dicts = {d.filepath: d.env.to_dict() for d in datasets}
        env_df = pd.DataFrame(env_dicts).T
        return env_df

    @staticmethod
    def _get_varied_env_vars(datasets: List[Recording], exclude=['notes']) -> dict[str, list[str]]:
        """Returns a dictionary of environmental variables that vary across datasets"""
        env_vars = {key: [] for key in RecordingEnvironment.keys()}
        for data in datasets:
            for key, value in data.env.to_dict().items():
                if value is not None and value not in env_vars[key] and key not in exclude:
                    env_vars[key].append(value)
        varied_vars = {key: value for key, value in env_vars.items() if len(value) > 1}
        return varied_vars
    
    def get_recording_metrics(self, data: Recording, plot=False):
        """Analyzes a recording and returns metrics"""
        if not isinstance(data, Recording):
            raise TypeError(f"Data must be of type Recording, not {type(data)}")
        try:
            return self._get_recording_metrics(data, plot)
        except Exception as e:
            self.logger.error(f"Failed to get metrics for {data.filepath}: {e}")
            raise e

    def _get_recording_metrics(self, data: Recording, plot=False) -> Tuple[Metrics, Metrics, pd.DataFrame]:
        """Analyzes a recording and returns metrics"""
        if data.env.fs != self.fs:
            raise ValueError(f"Recording fs ({data.env.fs}) does not match model fs ({self.fs})")
        true_step_groups = self._get_true_step_timestamps(data)
        true_steps = self.merge_step_groups(true_step_groups)
        step_groups = self._detector.get_step_groups(np.array(data.ts), plot, truth=true_steps, plot_title=data.filepath)
        if len(step_groups):
            self.logger.info(f"Found {len(step_groups)} step groups in {data.filepath}")
            self.logger.debug(f"Step groups: {step_groups}")
        predicted_steps = self.merge_step_groups(step_groups)
        measured = Metrics(step_groups)
        source_of_truth = Metrics(true_step_groups)
        algorithm_error = self._get_algorithm_error(predicted_steps, true_steps)
        return measured, source_of_truth, algorithm_error

    def get_snrs(self, recs: Iterable[Recording]):
        """Calculates the signal to noise ratio of a list of recordings"""
        return np.array([self.get_snr(rec) for rec in recs])

    def get_snr(self, rec: Recording):
        """Calculates the energy signal to noise ratio of a single recording"""
        if np.all(rec.ts == 0):
            self.logger.debug("Bad recording, skipping")
            return np.nan
        steps = RecordingProcessor.get_steps_from_truth(rec)
        noise_energy = self._detector.get_energy(RecordingProcessor.get_noise(rec))
        step_energy = np.concatenate([self._detector.get_energy(ts) for ts in steps])
        snr = np.var(step_energy) / np.var(noise_energy)
        return snr

    def _get_true_step_timestamps(self, data: Recording, ignore_quality=False, max_step_delta=2) -> List[np.ndarray]:
        """Returns the true step timestamps of a recording, enforcing a maximum step delta if the recording quality is not normal"""
        timestamps = [np.array([event.timestamp for event in data.events if event.category == 'step'])]
        if not ignore_quality and data.env.quality != 'normal':
            return self._detector._enforce_max_step_delta(timestamps, max_step_delta)
        return timestamps
    
    def get_false_step_rates(self, datasets: Iterable[Recording], plot_dist=False, plot_signals=False) -> dict:
        df = self.get_algorithm_error(datasets, plot_dist=plot_dist, plot_signals=plot_signals)
        false_negative = df['missed'].sum() / df['step_count'].sum()
        false_positive = df['incorrect'].sum() / df['step_count'].sum()
        return {"false_negative": false_negative, "false_positive": false_positive}

    def get_algorithm_error(self, datasets: Iterable[Recording], plot_dist=False, plot_signals=False) -> pd.DataFrame:
        """Calculates the algorithm error of a list of recordings"""
        errs = []
        recordings = []
        for data in datasets:
            try:
                recordings.append(data)
                errs.append(self.get_recording_algorithm_error(data, plot=plot_signals))
            except Exception as e:
                self.logger.error(f"Failed to get algorithm error for {data.filepath}: {e}")
                raise e
        df = pd.concat(errs)
        df.index = [d.filepath for d in recordings]
        if plot_dist:
            env_df = self.get_env_df(recordings)
            plot_df = pd.concat([env_df['quality'], df], axis=1)
            fig = make_subplots(rows=3, cols=1, subplot_titles=["Error", "Incorrect", "Missed"])
            fig.add_box(x=plot_df['quality'], y=plot_df['error'], boxpoints='all', row=1, col=1)
            fig.add_box(x=plot_df['quality'], y=plot_df['incorrect'], boxpoints='all', row=2, col=1)
            fig.add_box(x=plot_df['quality'], y=plot_df['missed'] / plot_df['step_count'], boxpoints='all', row=3, col=1)
            fig.show()
        return df

    def get_recording_algorithm_error(self, data: Recording, plot=False) -> pd.DataFrame:
        """Calculates the algorithm error of a single recording"""
        if data.env.fs != self._detector.fs:
            raise ValueError(f"Recording fs ({data.env.fs}) does not match model fs ({self.model.env.fs})")
        correct_steps = self.merge_step_groups(self._get_true_step_timestamps(data))
        step_groups = self._detector.get_step_groups(np.array(data.ts), plot=plot, truth=correct_steps)
        predicted_steps = self.merge_step_groups(step_groups)
        return self._get_algorithm_error(predicted_steps, correct_steps)

    @staticmethod
    def merge_step_groups(step_groups: List[np.ndarray]) -> np.ndarray:
        """Merges a list of step groups into a single array"""
        return np.concatenate(step_groups) if len(step_groups) else np.array([])

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
                "missed": [len(correct_times)],
                "step_count": [len(correct_times)],
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
        incorrect_measurements = len(measured_times) - len(measurement_errors)
        errors = list(measurement_errors.values())

        return pd.DataFrame({
            "error": [np.mean(errors) if len(errors) else np.nan],
            "stderr": [np.std(errors) if len(errors) else np.nan],
            "incorrect": [incorrect_measurements],
            "missed": [missed_steps],
            "step_count": [len(correct_times)],
        })




if __name__ == "__main__":
    sensor_type = SensorType.ACCEL
    params = get_optimal_analysis_params(sensor_type)
    controller = AnalysisController(**params)
    datasets = DataHandler.from_sensor_type(sensor_type).get_lazy(user='ron', location='Aarons Studio')
    # print(controller.get_metric_error(datasets, plot_dist=True, plot_title=str(params)))
    print(controller.get_false_step_rates(datasets, plot_dist=True))
