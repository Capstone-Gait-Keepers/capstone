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
        if len(rec.events) == 0:
            return rec.ts
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

        noise_amp_per_freq = RecordingProcessor.get_noise_frequency_weights(rec, window_duration, plot=plot)
        freq_weights = amp_per_freq / noise_amp_per_freq
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
            fig.update_layout(title="Amplitude vs Frequency (Average of all steps)", showlegend=False)
            fig.add_scatter(x=freqs, y=freq_weights)
            fig.show()
        return freq_weights

    @staticmethod
    def get_noise_frequency_weights(rec: Recording, window_duration=0.2, plot=False) -> np.ndarray:
        """Creates a model of noise frequency weights, ranging from 0 to 1, where 1 is the strongest frequency component of the noise floor"""
        proc = TimeSeriesProcessor(rec.env.fs)
        noise = RecordingProcessor.get_noise(rec)
        noise_amp_per_freq_time = proc.rolling_window_fft(noise, window_duration, stride=1, ignore_dc=True)
        noise_amp_per_freq = np.mean(noise_amp_per_freq_time, axis=-1)
        noise_amp_per_freq /= np.max(noise_amp_per_freq)
        if plot:
            freqs = proc.get_window_fft_freqs(window_duration)
            fig = px.scatter(x=freqs, y=noise_amp_per_freq, title="Noise Amplitude vs Frequency")
            fig.show()
        return noise_amp_per_freq

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

    @staticmethod
    def get_snr(rec: Recording, use_weights=False):
        """Calculates the energy signal to noise ratio of a single recording"""
        if np.all(rec.ts == 0):
            return np.nan
        proc = TimeSeriesProcessor(rec.env.fs)
        noise_ts = RecordingProcessor.get_noise(rec)
        step_ts = np.concatenate(RecordingProcessor.get_steps_from_truth(rec))
        weights = RecordingProcessor.get_frequency_weights(rec) if use_weights else None
        return proc.get_snr(step_ts, noise_ts, weights=weights)

    @staticmethod
    def get_snrs(recs: Iterable[Recording], **kwargs):
        """Calculates the signal to noise ratio of a list of recordings"""
        return np.array([RecordingProcessor.get_snr(rec, **kwargs) for rec in recs])



class AnalysisController:
    def __init__(self, model: Recording=None, fs=None, noise_amp=None, window_duration=0.2, logger: Optional[Logger]=None, log_file='latest.log', **kwargs) -> None:
        self.logger = self.init_logger(log_file) if logger is None else logger
        if model:
            noise = RecordingProcessor.get_noise(model)
            # weights = RecordingProcessor.get_frequency_weights(model, window_duration, plot=False)
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
            self._detector = StepDetector(fs, window_duration, noise_profile=np.random.rand(int(10*window_duration*fs)) * noise_amp, logger=self.logger, **kwargs)

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
        measured, truth, alg_err = self._parse_metric_results(results)
        err = measured.error(truth)
        err.index = [rec.tag for rec in recordings]
        # Log false negative and false positive rates
        env_df = self.get_env_df(recordings)
        alg_err = pd.concat([env_df['quality'], alg_err], axis=1)
        false_neg, false_pos = self._get_false_rates(alg_err, min_steps=3)
        self.logger.info(f"FNR = {false_neg:.4f}, FPR = {false_pos:.4f}")
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
        tags = []
        for data in datasets:
            if len(data.ts) < self._detector._window_duration * self.fs:
                self.logger.warning(f"Recording {data.tag} is too short to analyze")
                continue
            result = self.get_recording_metrics(data, plot_signals)
            results.append(result)
            if data.tag:
                tags.append(data.tag)
        measured, source_of_truth, algorithm_error = self._parse_metric_results(results)
        if len(tags) == len(results):
            measured.set_index(tags)
            source_of_truth.set_index(tags)
            algorithm_error['tags'] = tags
            algorithm_error.set_index('tags')
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
        env_dicts = {d.tag: d.env.to_dict() for d in datasets}
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
        # TODO: Daniel fix this - "TypeError: Data must be of type Recording, not <class 'data_analysis.data_types.Recording'>"
        if not isinstance(data, Recording):
            self.logger.warning(f"Data might not be of type recording: {type(data)}")
        try:
            return self._get_recording_metrics(data, plot)
        except Exception as e:
            self.logger.error(f"Failed to get metrics for {data.tag}: {e}")
            raise e

    def _get_recording_metrics(self, data: Recording, plot=False) -> Tuple[Metrics, Metrics, pd.DataFrame]:
        """Analyzes a recording and returns metrics"""
        if data.env.fs != self.fs:
            raise ValueError(f"Recording fs ({data.env.fs}) does not match model fs ({self.fs})")
        true_step_groups = self._get_true_step_timestamps(data)
        true_steps = self.merge_step_groups(true_step_groups)
        step_groups = self._detector.get_step_groups(np.array(data.ts), plot, truth=true_steps, plot_title=data.tag)
        if len(step_groups):
            self.logger.info(f"Found {len(step_groups)} step groups in {data.tag} ({len(true_steps)} true steps)")
            self.logger.debug(f"Step groups: {step_groups}")
        predicted_steps = self.merge_step_groups(step_groups)
        measured = Metrics(step_groups)
        source_of_truth = Metrics(true_step_groups)
        algorithm_error = self._get_algorithm_error(predicted_steps, true_steps)
        return measured, source_of_truth, algorithm_error

    def _get_true_step_timestamps(self, data: Recording, ignore_quality=False, max_step_delta=2) -> List[np.ndarray]:
        """Returns the true step timestamps of a recording, enforcing a maximum step delta if the recording quality is not normal"""
        if not isinstance(data.env, RecordingEnvironment):
            return []
        timestamps = [np.array([event.timestamp for event in data.events if event.category == 'step'])]
        if not ignore_quality and data.env.quality != 'normal':
            return self._detector._enforce_max_step_delta(timestamps, max_step_delta)
        return timestamps

    def get_false_rates(self, datasets: Iterable[Recording], min_steps=3, plot_dist=False, plot_signals=False):
        """Calculates the false negative and false positive rates of a list of recordings"""
        recordings = [*datasets]
        df = self.get_algorithm_error(recordings, plot_dist=plot_dist, plot_signals=plot_signals)
        env_df = self.get_env_df(recordings)
        df = pd.concat([env_df['quality'], df], axis=1)
        return self._get_false_rates(df, min_steps)

    def _get_false_rates(self, alg_err: pd.DataFrame, min_steps=2):
        """
        Calculates the false negative and false positive rates of a list of recordings.

        Parameters
        ----------
        alg_err : pd.DataFrame
            A dataframe of algorithm errors for each recording.
            Must contain 'quality', 'missed', 'incorrect', and 'step_count' columns.
        min_steps : int
            The minimum number of steps required for a recording to be counted as
            successfully detected steps. Default is 2, which is the minimum number
            of steps for any metric.
        """
        # TODO: Include pause and turn? step_count > 0?
        positives = alg_err[alg_err['quality'] == 'normal']
        negatives = alg_err[alg_err['quality'] == 'chaotic']
        if len(positives) == 0:
            false_neg = np.nan
        else:
            steps_detected = positives['step_count'] - positives['missed']
            false_neg = len(positives[steps_detected < min_steps]) / len(positives)
        if len(negatives) == 0:
            false_pos = np.nan
        else:
            false_pos = len(negatives[negatives['incorrect'] > 0]) / len(negatives)
        return false_neg, false_pos

    def get_algorithm_error(self, datasets: Iterable[Recording], plot_dist=False, plot_signals=False) -> pd.DataFrame:
        """Calculates the algorithm error of a list of recordings"""
        errs = []
        recordings = []
        for data in datasets:
            try:
                recordings.append(data)
                errs.append(self.get_recording_algorithm_error(data, plot=plot_signals))
            except Exception as e:
                self.logger.error(f"Failed to get algorithm error for {data.tag}: {e}")
                raise e
        df = pd.concat(errs)
        df.index = [d.tag for d in recordings]
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


def compare_sensor_snrs(use_weights=True):
    """
    Compares the signal to noise ratios of piezo and accelerometer sensors. If use_weights is True, 
    the frequency weights will be used to calculate the SNR.

    # Current Results
    use_weights = False:
     - Piezo SNR = 7.48
     - Accel SNR = 12.70
    use_weights = True:
     - Piezo SNR = 21.23
     - Accel SNR = 15.72
    """
    datasets = DataHandler.from_sensor_type(SensorType.PIEZO).get_lazy(user='ron', quality='normal', session="2024", location='Aarons Studio')
    piezo_snrs = RecordingProcessor.get_snrs(datasets, use_weights=use_weights)
    print(piezo_snrs)
    datasets = DataHandler.from_sensor_type(SensorType.ACCEL).get_lazy(user='ron', quality='normal', session="2024", location='Aarons Studio')
    accel_snrs = RecordingProcessor.get_snrs(datasets, use_weights=use_weights)
    print(accel_snrs)

    print(f"Avg piezo snr: {np.nanmean(piezo_snrs):.2f}, Avg accel snr: {np.nanmean(accel_snrs):.2f}")

    # Plot histograms
    improvement = piezo_snrs - accel_snrs
    fig = go.Figure()
    # Label with index
    fig.add_histogram(x=improvement, text=[*zip(piezo_snrs, accel_snrs)], name="Piezo SNR Improvement")
    fig.show()



if __name__ == "__main__":
    sensor_type = SensorType.PIEZO
    params = get_optimal_analysis_params(sensor_type)
    controller = AnalysisController(**params)
    datasets = DataHandler.from_sensor_type(sensor_type).get_lazy(user='ron', limit=4, location='Aarons Studio')
    # print(controller.get_metric_error(datasets, plot_dist=True, plot_signals=False, plot_title=str(params)))
    # print(controller.get_false_rates(datasets, plot_dist=False))
    print(controller.get_metrics(datasets, plot_signals=False)[0].by_recordings())
