import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple

from data_types import Recording


class DataHandler:
    def __init__(self, folder = "datasets") -> None:
        self.folder = folder

    def get(self, **filters) -> Dict[str, Recording]:
        """Walk through datasets folder and return all records that match the filters"""
        filepaths = [file for root, dirs, files in os.walk(self.folder) for file in files if file.endswith(".yaml")]
        filepaths.remove('example.yaml')
        datasets = {filename: Recording.from_file(os.path.join(self.folder, filename)) for filename in filepaths}
        for key, value in filters.items():
            datasets = {filename: dataset for filename, dataset in datasets.items() if getattr(dataset.env, key) == value}
        return datasets

    def plot(self, **filters):
        """Walk through datasets folder and plot all recording that match the filters"""
        datasets = self.get(**filters)
        fig = make_subplots(rows=len(datasets), cols=1, subplot_titles=[*datasets.keys()], shared_xaxes=True)
        fig.update_layout(title=str(filters), showlegend=False)
        for i, data in enumerate(datasets.values(), start=1):
            timestamps = np.linspace(0, len(data.ts) / data.env.fs, len(data.ts))
            fig.add_trace(go.Scatter(x=timestamps, y=data.ts, name='vibes'), row=i, col=1)
            # TODO: Better text to identify between datasets
            # env_vars = data.env.to_dict()
            # env_vars = {key: value for key, value in env_vars.items() if value is not None and key not in filters}
            # for y, (key, value) in enumerate(env_vars.items(), start=-len(env_vars) // 2):
            #     fig.add_annotation(x=timestamps[len(timestamps)//2], y=y/80, text=f"{key} = {value}", xshift=0, showarrow=False, row=i, col=1)
        fig.show()


class TimeSeriesProcessor:
    def __init__(self, fs: float) -> None:
        self.fs = fs

    def timestamp_to_index(self, timestamp: float) -> int:
        """
        Converts a timestamp to an index in a time series

        Parameters
        ----------
        timestamp : float
            Timestamp in seconds
        fs : float
            Sampling frequency in Hz

        Returns
        -------
        int
            Index of the timestamp in the time series
        """
        return np.floor(timestamp * self.fs).astype(int)

    @staticmethod
    def rolling_window(a: np.ndarray, window_size: int, stride: int = None) -> np.ndarray:
        """
        Creates a rolling window of a time series

        Parameters
        ----------
        a : np.ndarray
            Time series to be windowed
        window_size : int
            Size of the window in samples
        stride : int, optional
            Number of samples to move the window by. If None, stride is equal to window size

        Returns
        -------
        np.ndarray
            Windowed time series
        """
        if stride is None:
            stride = window_size
        return np.asarray([a[i:i + window_size] for i in range(0, len(a) - window_size, stride)])

    def rolling_window_fft(self, a: np.ndarray, window_duration: int, stride: int, ignore_dc=False) -> np.ndarray:
        """
        Creates a rolling window of a time series and computes the FFT of each window
        """
        window_size = self.timestamp_to_index(window_duration)
        padded_a = np.pad(a, (window_size // 2, window_size // 2), mode='edge')
        intervals = TimeSeriesProcessor.rolling_window(padded_a, window_size, stride)
        if ignore_dc:
            intervals -= np.mean(intervals, axis=1, keepdims=True)
        # TODO: Better condition for tapering?
        if stride < window_size:
            tapering_window = np.vstack([np.blackman(window_size)] * len(intervals))
            intervals *= tapering_window
        rolling_fft = np.fft.fft(intervals, axis=-1).T[:window_size//2]
        return np.abs(rolling_fft)

    def get_window_fft_freqs(self, window_duration) -> np.ndarray:
        window_size = self.timestamp_to_index(window_duration)
        return np.fft.fftfreq(window_size, 1/self.fs)[:window_size//2] # Ignore negative frequencies

    def get_energy(self, vibes: np.ndarray, window_duration=0.2, stride=1, weights=None) -> np.ndarray:
        """
        Find the energy of the accelerometer data
        """
        p_vibes = vibes - np.mean(vibes, axis=0) # Subtract DC offset
        amps = self.rolling_window_fft(p_vibes, window_duration, stride)
        energy = np.average(amps, axis=0, weights=weights)
        if energy.shape != vibes.shape:
            raise ValueError(f"Energy is the wrong shape: {energy.shape} != {vibes.shape}")
        return energy

    @staticmethod
    def get_peak_indices(signal: np.ndarray, threshold: float):
        """Find the indices of the peaks in a time series above a certain threshold"""
        de_dt = np.gradient(signal)
        optima = np.diff(np.sign(de_dt), prepend=[0]) != 0
        indices = np.where((signal > threshold) & optima)[0]
        return indices


class StepDetector(TimeSeriesProcessor):
    def __init__(self, fs: float, window_duration: float, noise_profile: np.ndarray, step_model=None, freq_weights=None) -> None:
        """
        Parameters
        ----------
        window_duration : float
            Duration of the rolling window in seconds
        step_model : np.ndarray, optional
            Model of step energy vs time. Model is cross correlated with the energy of the time series
        freq_weights : np.ndarray, optional
            Weights to apply to each frequency when averaging the frequency spectrum
        """
        super().__init__(fs)
        self._noise = self.get_energy(noise_profile)
        self._window_duration = window_duration
        self._step_model = step_model
        self._freq_weights = freq_weights
        if freq_weights is not None and len(freq_weights) != len(self.get_window_fft_freqs(window_duration)):
            raise ValueError(f"Length of freq_weights ({len(freq_weights)}) must match the window duration ({window_duration})")

    def get_step_groups(self, ts: np.ndarray, **kwargs) -> List[np.ndarray]:
        """
        Analyzes a recording and returns a dictionary of metrics
        """
        steps, uncertain_steps = self._find_steps(ts, **kwargs)
        step_groups = self._resolve_step_sections(steps, uncertain_steps)
        return step_groups

    def _find_steps(self, vibes: np.ndarray, plot=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Counts the number of steps in a time series of accelerometer data. This function should not use
        anything from `data.events` except for plotting purposes. This is because it is meant to mimic
        a blind step detection algorithm.

        Parameters
        ----------
        data : Recording
            Time series of accelerometer data, plus the environmental data
        """
        timestamps = np.linspace(0, len(vibes) / self.fs, len(vibes))
        ### Time series processing
        amps = self.rolling_window_fft(vibes, self._window_duration, stride=1, ignore_dc=True)
        ### Frequency domain processing
        # Weight average based on shape of frequency spectrum
        energy = np.average(amps, axis=0, weights=self._freq_weights)
        # Cross correlate step model with energy
        if self._step_model is not None:
            energy = np.correlate(energy, self._step_model, mode='same')
            model_autocorr = np.correlate(self._step_model, self._step_model, mode='valid')
            energy /= np.max(model_autocorr)
        confirmed_threshold, uncertain_threshold = self._get_energy_thresholds(np.max(energy))
        confirmed_indices = self.get_peak_indices(energy, confirmed_threshold)
        # TODO: This finds confirmed peaks as well. Fix this
        uncertain_indices = self.get_peak_indices(energy, uncertain_threshold) if uncertain_threshold is not None else []
        uncertain_indices = np.setdiff1d(uncertain_indices, confirmed_indices)    
        confirmed_stamps = timestamps[confirmed_indices]
        uncertain_stamps = timestamps[uncertain_indices]
        # TODO: Find start of step within confirmed window?

        if plot:
            titles = ("Raw Timeseries", "Scrolling FFT", "Average Energy")
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=titles)
            fig.add_scatter(x=timestamps, y=vibes, name='vibes', showlegend=False, row=1, col=1)
            freqs = self.get_window_fft_freqs(self._window_duration)
            fig.add_heatmap(x=timestamps, y=freqs, z=amps, row=2, col=1)
            fig.add_scatter(x=timestamps, y=energy, name='energy', showlegend=False, row=3, col=1)
            for threshold in (confirmed_threshold, uncertain_threshold):
                fig.add_hline(y=threshold, row=3, col=1)
            # TODO: Add source of truth back to plot
            # for event in data.events:
            #     fig.add_vline(x=event.timestamp + 0.05, line_color='green', row=1, col=1)
            #     fig.add_annotation(x=event.timestamp + 0.05, y=0, xshift=-17, text="Step", showarrow=False, row=1, col=1)
            for confirmed in confirmed_stamps:
                fig.add_vline(x=confirmed, line_dash="dash", row=1, col=1)
                fig.add_annotation(x=confirmed, y=-0.3, xshift=-10, text="C", showarrow=False, row=1, col=1)
            for uncertain in uncertain_stamps:
                fig.add_vline(x=uncertain, line_dash="dot", row=1, col=1)
                fig.add_annotation(x=uncertain, y=-0.3, xshift=-10, text="U", showarrow=False, row=1, col=1)
            # fig.write_html("normal_detection.html")
            fig.show()
        return confirmed_stamps, uncertain_stamps

    # TODO: Parameterize weights of max_sig, max_noise and np.std(noise) to find optimal thresholds
    def _get_energy_thresholds(self, max_sig, c=0.7, u1=.5, u2=0, u3=0):
        noise_max = np.max(self._noise)
        noise_std = np.std(self._noise)
        confirmed_threshold = c*max_sig + (1-c)*noise_max
        uncertain_threshold = u1*max_sig + (1-u1)*noise_max + u2*noise_std + u3
        assert uncertain_threshold <= confirmed_threshold, f"Uncertain threshold ({uncertain_threshold}) must be less than confirmed threshold ({confirmed_threshold})"
        return uncertain_threshold, confirmed_threshold

    @staticmethod
    def _resolve_step_sections(confirmed_stamps: np.ndarray, uncertain_stamps: np.ndarray = [], min_step_delta=0.05) -> List[np.ndarray]:
        """Groups confirmed steps into sections, ignoring unconfirmed steps. Sections must have at least 3 steps."""
        all_steps = np.concatenate([confirmed_stamps, uncertain_stamps])
        if len(set(set(all_steps))) != len(all_steps):
            raise ValueError("All step stamps must be unique")
        if not np.all(np.diff(confirmed_stamps) > min_step_delta):
            raise ValueError(f"Confirmed step stamps must be at least {min_step_delta}s apart")

        confirmed = pd.Series([True] * len(confirmed_stamps), index=confirmed_stamps)
        if len(uncertain_stamps):
            unconfirmed_steps = pd.Series([False] * len(uncertain_stamps), index=uncertain_stamps)
            confirmed = pd.concat([confirmed, unconfirmed_steps])
        confirmed = confirmed.sort_index()

        # Upgrading unconfirmed steps to confirmed steps if there's only one unconfirmed step between two confirmed steps
        for prev_step, current_step, next_step in zip(confirmed.index[:-2], confirmed.index[1:-1], confirmed.index[2:]):
            if not confirmed[current_step] and confirmed[prev_step] and confirmed[next_step]:
                confirmed[current_step] = True

        # Grouping confirmed steps into sections
        current_section = 0
        section_indices = [current_section] * len(confirmed)
        for i, (prev_step, current_step) in enumerate(zip(confirmed.iloc[:-1], confirmed.iloc[1:]), 1):
            if current_step and not prev_step:
                current_section += 1
            section_indices[i] = current_section
        steps = pd.DataFrame({'confirmed': confirmed, 'section': section_indices}, index=confirmed.index)
        steps = steps[steps.confirmed] # Ignore unconfirmed steps that were not upgraded
        steps = steps.groupby('section').filter(lambda x: len(x) >= 3) # Ignore sections with less than 3 steps
        sections = [group.index.values for _, group in steps.groupby('section')]

        # TODO: Set hard bounds on min/max step delta, cadence and asymmetry and ignore sections that don't meet them
        return sections


class MetricAnalyzer:
    def __init__(self, step_detector: StepDetector) -> None:
        self._detector = step_detector

    def get_recording_temporal_asymmetry(self, ts: np.ndarray):
        """
        Calculates the temporal asymmetry of a time series. 
        """
        step_timestamps = self._detector.find_steps(ts)
        return self.get_temporal_asymmetry(step_timestamps)

    @staticmethod
    def get_temporal_asymmetry(step_timestamps: np.ndarray):
        """
        Calculates the temporal asymmetry of a list of step timestamps. 
        """
        if len(step_timestamps) < 3:
            return np.nan
        step_durations = np.diff(step_timestamps)
        return np.abs(np.mean(step_durations[1:] / step_durations[:-1]) - 1) / np.mean(step_durations)
    
    def get_recording_cadence(self, ts: np.ndarray):
        """
        Calculates the cadence of a time series. 
        """
        step_timestamps = self._detector.find_steps(ts)
        return self.get_cadence(step_timestamps)

    @staticmethod
    def get_cadence(step_timestamps: np.ndarray):
        """
        Calculates the cadence of a list of step timestamps. 
        """
        if len(step_timestamps) < 2:
            return np.nan
        step_durations = np.diff(step_timestamps)
        return 1 / np.mean(step_durations)

    # TODO
    @staticmethod
    def get_recording_gait_type(ts: np.ndarray):
        raise NotImplementedError()


@dataclass
class ParsedRecording(Recording):
    @classmethod
    def from_recording(cls, recording: Recording):
        return cls(recording.env, recording.events, recording.ts, recording.filepath)

    # TODO: Accept a TimeSeriesProcessor with the signal conversion-type specified (e.g. energy vs correlation)
    def __post_init__(self):
        self.processor = TimeSeriesProcessor(self.env.fs)

    def get_steps_from_truth(self, step_duration=0.4, shift_percent=0.2, align_peak=True, plot=False) -> List[np.ndarray]:
        """
        Uses the source of truth to parse the accelerometer data pertaining to steps
        """
        offset = int(shift_percent * self.env.fs / 2)
        window_size = self.processor.timestamp_to_index(step_duration)
        step_measurements = []
        for event in self.events:
            if event.category == 'step':
                start = self.processor.timestamp_to_index(event.timestamp) - offset
                step_data = self.ts[start : start+window_size]
                if align_peak:
                    energy = self.processor.get_energy(step_data, step_duration / 5)
                    start += np.argmax(energy) - offset
                    step_data = self.ts[start : start+window_size]
                if start + window_size > len(self.ts):
                    print("Step is too close to the end of the recording, skipping. Consider decreasing the step duration")
                    continue
                if len(step_data) != window_size:
                    raise ValueError(f"Step data is the wrong size: {len(step_data)} != {window_size}")
                step_measurements.append(step_data)
        if plot:
            fig = go.Figure()
            fig.update_layout(title="Step Data", showlegend=False)
            for step in step_measurements:
                fig.add_scatter(x=np.linspace(0, step_duration, len(step)), y=step)
            fig.show()
        return step_measurements

    def plot(self, show=True):
        """Plot a recording"""
        timestamps = np.linspace(0, len(self.ts) / self.env.fs, len(self.ts))
        fig = go.Figure()
        if self.filepath:
            fig.update_layout(title=self.filepath, showlegend=False)
        fig.add_scatter(x=timestamps, y=self.ts, name='vibes')
        for event in self.events:
            fig.add_vline(x=event.timestamp, line_color='green')
            fig.add_annotation(x=event.timestamp, y=0, xshift=-17, text=event.category, showarrow=False)
        if show:
            fig.show()
        return fig

    def get_noise(self, plot=False) -> np.ndarray:
        """
        Find the noise floor of the accelerometer data
        """
        first_event = self.events[0].timestamp
        noise = self.ts[:self.processor.timestamp_to_index(first_event)]
        if plot:
            timestamps = np.linspace(0, len(noise) / self.env.fs, len(noise))
            fig = go.Figure()
            fig.update_layout(title="Noise", showlegend=False)
            fig.add_scatter(x=timestamps, y=noise)
            fig.show()
        return noise

    @property
    def snr(self) -> float:
        """
        Find the signal-to-noise ratio of the accelerometer data
        """
        noise_var = np.var(self.get_noise())
        signal_var = np.var(np.concatenate(self.get_steps_from_truth()))
        snr = signal_var / noise_var
        return snr

    # TODO
    def get_snr_vs_time(self, plot=False) -> np.ndarray:
        raise NotImplementedError()

    def get_frequency_weights(self, window_duration=0.2, plot=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the dominant frequencies pertaining to steps
        """
        DC = np.mean(self.ts, axis=0)
        step_data = self.get_steps_from_truth(window_duration*2)
        freqs = self.processor.get_window_fft_freqs(window_duration)
        amp_per_step_freq_time = []

        for step in step_data:
            amps = self.processor.rolling_window_fft(step - DC, window_duration, stride=1)
            amp_per_step_freq_time.append(amps)
        amp_per_step_freq_time = np.asarray(amp_per_step_freq_time)
        amp_per_step_freq = np.mean(amp_per_step_freq_time, axis=-1)
        amp_per_freq = np.mean(amp_per_step_freq, axis=0)
        amp_per_freq /= np.max(amp_per_freq)

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
            fig.add_scatter(x=freqs, y=amp_per_freq)
            fig.show()
        return amp_per_freq

    def get_step_model(self, window_duration=0.2, plot_model=False, plot_steps=False) -> np.ndarray:
        """Creates a model of step energy vs time"""
        step_data = self.get_steps_from_truth()
        energy_per_step = []
        for step in step_data:
            # TODO: Use weights from get_frequency_weights
            energy = self.processor.get_energy(step, window_duration)
            energy_per_step.append(energy / np.max(energy))
        step_model = np.mean(energy_per_step, axis=0)
        timestamps = np.linspace(0, window_duration, len(energy_per_step[0]))
        if plot_steps:
            fig = go.Figure()
            fig.update_layout(title="Energy vs Time")
            for energy in energy_per_step:
                fig.add_scatter(x=timestamps, y=energy)
            fig.show()
        if plot_model:
            fig = go.Figure()
            fig.update_layout(title="Model Step", showlegend=False)
            fig.add_scatter(x=timestamps, y=step_model)
            fig.show()
        return step_model


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
        self._analyzer = MetricAnalyzer(self._detector)

    # TODO: Return Metrics object
    def analyze(self, ts: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Analyzes a recording and returns a dictionary of metrics
        """
        step_groups = self._detector.get_step_groups(ts, **kwargs)
        if not len(step_groups):
            raise ValueError("No valid step sections found")
        metrics = {
            "asymmetry": MetricAnalyzer.get_temporal_asymmetry,
            "cadence": MetricAnalyzer.get_cadence,
        }
        return {metric: AnalysisController._get_func_metric(func, step_groups) for metric, func in metrics.items()}

    # TODO: Implement better
    def analyze_error(self, ts: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Analyzes a recording and returns a dictionary of metrics
        """
        step_groups = self._detector.get_step_groups(ts, **kwargs)
        if not len(step_groups):
            raise ValueError("No valid step sections found")
        metrics = {
            "asymmetry": MetricAnalyzer.get_temporal_asymmetry,
            "cadence": MetricAnalyzer.get_cadence,
        }
        return {metric: AnalysisController._get_func_metric_error(func, ts, self.model.events) for metric, func in metrics.items()}
    
    def _get_func_metric(func: callable, step_groups: List[np.ndarray]):
        """Calculates the % metric of a list of step timestamps, given a function to compute the metric."""
        return np.mean([func(steps) for steps in step_groups])

    @staticmethod
    def get_algorithm_error(measured_times: np.ndarray, correct_stamps: np.ndarray):
        """
        Calculates the algorithm error of a recording. There are three types of errors:
        - Incorrect measurements: The algorithm found a step when there was none (False Positive)
        - Missed steps: The algorithm missed a step (False Negative)
        - Measurement error: The algorithm found a step correctly, but at the wrong time

        Parameters
        ----------
        measured_times : np.ndarray
            List of timestamps where the algorithm found steps
        correct_stamps : np.ndarray
            List of step timestamps from the source of truth
        """
        if not len(measured_times):
            raise ValueError("Algorithm did not find any steps")
        missed_steps = 0
        measurement_errors = {}
        for step_stamp in correct_stamps:
            possible_errors = np.abs(measured_times - step_stamp)
            best_measurement = np.argmin(possible_errors)
            # If the measurement is already the best one for another step, it means we missed this step
            if best_measurement in measurement_errors:
                missed_steps += 1
            else:
                measurement_errors[best_measurement] = possible_errors[best_measurement]
        incorrect_measurements = len(measured_times) - len(measurement_errors)
        errors = list(measurement_errors.values())
        return {
            "error": np.mean(errors),
            "stderr": np.std(errors),
            "incorrect": incorrect_measurements,
            "missed": missed_steps
        }

    @staticmethod
    def get_metric_error(measured_stamps: np.ndarray, correct_stamps: np.ndarray):
        """Calculates the % metric error of a measured timestamp list, given the source of truth."""
        metrics = {
            "asymmetry": MetricAnalyzer.get_temporal_asymmetry,
            "cadence": MetricAnalyzer.get_cadence,
        }
        return {metric: AnalysisController._get_func_metric_error(func, measured_stamps, correct_stamps) for metric, func in metrics.items()}

    @staticmethod
    def _get_func_metric_error(func: callable, measured_stamps: np.ndarray, correct_stamps: np.ndarray):
        """Calculates the % error between the two timestamp lists, given a function to compute the metric."""
        correct_metric = func(correct_stamps)
        measured_metric = func(measured_stamps)
        return np.abs(measured_metric - correct_metric) / correct_metric

    @staticmethod
    def optimize_treshold_weights(datasets: List[Recording], plot=False, **kwargs) -> Tuple[float, float]:
        raise NotImplementedError()



if __name__ == "__main__":
    model_data = Recording.from_file('datasets/2023-11-09_18-42-33.yaml')
    controller = AnalysisController(model_data)
    data = Recording.from_file('datasets/2023-11-09_18-46-43.yaml')
    print(controller.analyze(model_data.ts, plot=False))
    

    # DataHandler().plot(walk_type='normal', user='ron', walk_speed='normal', footwear='socks', wall_radius=1.89)

    # datasets = get_datasets(walk_type='normal', user='ron', walk_speed='normal', footwear='socks')
    # for data in datasets:
    #     steps = find_steps(data, amp_threshold=5 * get_noise_variance(data), freq_weights=weights, plot=True)
