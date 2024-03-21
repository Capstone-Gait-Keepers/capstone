import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional
from logging import Logger, getLogger
from itertools import product

from data_types import Recording, SensorType


class DataHandler:
    folder_map = {
        SensorType.ACCEL: "datasets/bno055",
        SensorType.PIEZO: "datasets/piezo",
    }

    def __init__(self, folder = "datasets") -> None:
        self.folder = folder
    
    @classmethod
    def from_sensor_type(cls, sensor_type: SensorType) -> 'DataHandler':
        return cls(cls.folder_map[sensor_type])

    def get(self, **filters) -> List[Recording]:
        """
        Walk through datasets folder and return all records that match the filters

        See `get_lazy` for more information on parameters
        """
        return list(self.get_lazy(**filters))

    def get_lazy(self, limit=None, session=None, **filters):
        """
        Walk through datasets folder and yield all records that match the filters

        Parameters
        ----------
        limit : int, optional
            Maximum number of records to return
        session : str, optional
            Session name to filter by. Typically a numeric date
        filters : keyword arguments
            Key-value pairs to filter the environmental variables by
        """
        filepaths = [file for file in os.listdir(self.folder) if file.endswith(".yaml") or file.endswith(".yml")]
        if 'example.yaml' in filepaths:
            filepaths.remove('example.yaml')
        if session is not None:
            filepaths = [file for file in filepaths if session in file]
        if limit is not None and len(filters) == 0: # If there are no filters, all files will be valid
            filepaths = filepaths[:limit]
        recs_yielded = 0
        for filename in filepaths:
            data = Recording.from_file(os.path.join(self.folder, filename))
            for key, value in filters.items():
                if getattr(data.env, key) != value:
                    break
            else:
                yield data
                recs_yielded += 1
                if limit is not None and recs_yielded >= limit:
                    break

    def plot(self, clip=False, truth=True, **filters):
        """Walk through datasets folder and plot all recording that match the filters"""
        datasets = self.get(**filters)
        if len(datasets) == 0:
            raise ValueError("No datasets found")
        fig = make_subplots(
            rows=len(datasets),
            cols=2,
            subplot_titles=[rec.tag for rec in datasets],
            shared_xaxes=True,
            specs=[[{'type': 'xy'}, {'type': 'table'}]] * len(datasets)
        )

        fig.update_layout(title=str(filters), showlegend=False)
        len_shortest = min([len(data.ts) for data in datasets])
        for i, data in enumerate(datasets, start=1):
            if clip:
                data.ts = data.ts[:len_shortest]
            timestamps = np.linspace(0, len(data.ts) / data.env.fs, len(data.ts))
            fig.add_trace(go.Scatter(x=timestamps, y=data.ts, name='vibes'), row=i, col=1)
            if truth:
                for event in data.events:
                    fig.add_vline(x=event.timestamp, line_color='green', row=i, col=1)

        for i, data in enumerate(datasets, start=1):
            env_vars = data.env.to_dict()
            env_vars = {key: value for key, value in env_vars.items() if value is not None and key not in filters}
            fig.add_trace(go.Table(
                header=dict(values=list(env_vars.keys())),
                cells=dict(values=list(env_vars.values())),
            ), row=i, col=2)
        fig.show()


class TimeSeriesProcessor:
    def __init__(self, fs: float) -> None:
        if not isinstance(fs, (int, float)):
            raise TypeError(f"fs must be a number, not {type(fs)}")
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
            Windowed time series, of shape (n_windows, window_size)
        """
        if stride is None:
            stride = window_size
        return np.asarray([a[i:i + window_size] for i in range(0, len(a) - window_size, stride)])

    def rolling_window_fft(self, a: np.ndarray, window_duration: int, stride: int, ignore_dc=False) -> np.ndarray:
        """
        Creates a rolling window of a time series and computes the FFT of each window
        """
        window_size = self.timestamp_to_index(window_duration)
        if len(a) < window_size:
            raise ValueError(f"Time series length ({len(a)}) is less than window size ({window_size})")
        intervals = TimeSeriesProcessor.rolling_window(a, window_size, stride)
        if ignore_dc:
            intervals -= np.mean(intervals, axis=1, keepdims=True)
        if stride < window_size:
            tapering_window = np.vstack([np.blackman(window_size)] * len(intervals))
            intervals *= tapering_window
        rolling_fft = np.fft.fft(intervals, axis=-1).T[:window_size//2]
        rolling_fft = np.pad(rolling_fft, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
        # Edge case when the window size is odd
        if len(a) == rolling_fft.shape[-1] + 1:
            rolling_fft = np.pad(rolling_fft, ((0, 0), (0, 1)), mode='constant')
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
    
    def get_max_power(self, vibes: np.ndarray, window_duration=0.2, stride=1, weights=None) -> float:
        """
        Find the max power of the accelerometer data, given the time series data
        """
        energy = self.get_energy(vibes, window_duration, stride, weights)
        return np.max(energy**2, axis=None)

    def get_snr(self, signal: np.ndarray, noise: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate the signal-to-noise ratio of a signal, given the time series data
        """
        signal_power = np.max(self.get_energy(signal, **kwargs)**2)
        noise_power = np.max(self.get_energy(noise, **kwargs)**2)
        return signal_power / noise_power

    @staticmethod
    def get_peak_indices(signal: np.ndarray, threshold: float, reset_threshold: float = None) -> np.ndarray:
        """
        Find the indices of the peaks in a time series above a certain threshold. 
        If a reset threshold is provided, the signal must drop below it before finding another peak
        """
        de_dt = np.gradient(signal)
        optima = np.diff(np.sign(de_dt), prepend=[0]) != 0
        peak_indices = np.where((signal > threshold) & optima)[0]
        bad_peaks = []
        if reset_threshold is not None:
            for peak1, peak2 in zip(peak_indices[:-1], peak_indices[1:]):
                if not np.any(signal[peak1:peak2] < reset_threshold):
                    bad_peaks.append(peak2)
            peak_indices = np.setdiff1d(peak_indices, bad_peaks)
        return peak_indices

    @staticmethod
    def drop_uncertain_steps(signal: np.ndarray, uncertain_indices: np.ndarray, confirmed_indices: np.ndarray, reset_threshold: float) -> np.ndarray:
        """
        Remove uncertain steps that are too close to confirmed steps. Signal must drop below reset_threshold
        before finding another peak, confirmed or unconfirmed.
        """
        uncertain_indices_to_drop = set()
        for uncertain, certain in product(uncertain_indices, confirmed_indices):
            if (certain > uncertain and signal[uncertain:certain].min() > reset_threshold) or (certain < uncertain and signal[certain:uncertain].min() > reset_threshold):
                uncertain_indices_to_drop.add(uncertain)
        return np.setdiff1d(uncertain_indices, [*uncertain_indices_to_drop, *confirmed_indices])



class StepDetector(TimeSeriesProcessor):
    def __init__(self,
                 fs: float,
                 window_duration: float,
                 noise_profile: np.ndarray,
                 min_signal: float = None,
                 min_step_delta: float = 0,
                 max_step_delta: float = 2,
                 confirm_coefs: Tuple[float, float, float, float] = (0.5, 0.3, 0, 0),
                 unconfirm_coefs: Tuple[float, float, float, float] = (0.25, 0.65, 0, 0),
                 reset_coefs: Tuple[float, float, float, float] = (0, 1, 0, 0),
                 step_model=None,
                 freq_weights=None,
                 logger: Optional[Logger]=None,
    ) -> None:
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
        self.logger = logger if logger is not None else getLogger(__name__)
        self._window_duration = window_duration
        self._min_signal = min_signal
        if min_step_delta > max_step_delta:
            raise ValueError(f"min_step_delta ({min_step_delta}) must be less than max_step_delta ({max_step_delta})")
        self._min_step_delta = min_step_delta
        self._max_step_delta = max_step_delta
        self._confirm_coefs = confirm_coefs
        self._unconfirm_coefs = unconfirm_coefs
        self._reset_coefs = reset_coefs
        # Modelling
        self._step_model = step_model
        self._freq_weights = freq_weights
        if freq_weights is not None and len(freq_weights) != len(self.get_window_fft_freqs()):
            raise ValueError(f"Length of freq_weights ({len(freq_weights)}) must match the window duration ({window_duration})")
        self._noise = self.get_energy(noise_profile)

    def get_step_groups(self, ts: np.ndarray, plot=False, truth=None, plot_table=None, plot_title=None) -> List[np.ndarray]:
        """
        Analyzes time series and returns a list of step groups
        """
        if ts[-1] is None:
            ts = ts[:-1]
            self.logger.warning("Removed last value from time series because it was None")
        if None in ts:
            raise ValueError("Time series must not contain None values")
        if np.all(ts == 0):
            self.logger.debug("Time series is all zeros, ignoring")
            return []
        steps, uncertain_steps = self._find_steps(ts, plot, truth, plot_table, plot_title)
        self.logger.debug(f"Found {len(steps)} confirmed steps and {len(uncertain_steps)} uncertain steps")
        if len(steps) > 1: # We need at least two confirmed steps to do anything
            step_groups = self._resolve_step_sections(steps, uncertain_steps)
            self.logger.info(f"Resolved {len(steps)} steps into {len(step_groups)} step groups of lengths {[len(group) for group in step_groups]}")
            if len(step_groups):
                step_groups = self._enforce_min_step_delta(step_groups)
                step_groups = self._enforce_max_step_delta(step_groups)
            return step_groups
        return []

    def get_energy(self, vibes: np.ndarray) -> np.ndarray:
        return super().get_energy(vibes, self._window_duration, weights=self._freq_weights)

    def get_window_fft_freqs(self) -> np.ndarray:
        return super().get_window_fft_freqs(self._window_duration)

    def _find_steps(self, vibes: np.ndarray, plot=False, truth=None, plot_table=None, plot_title=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Counts the number of steps in a time series of accelerometer data. This function should not use
        anything from `data.events` except for plotting purposes. This is because it is meant to mimic
        a blind step detection algorithm.

        Parameters
        ----------
        data : Recording
            Time series of accelerometer data, plus the environmental data
        plot : bool
            Whether or not to plot the results
        truth: Optional[List[float]]
            Source of truth for the steps. Only used for plotting
        plot_table: Optional[pd.DataFrame]
            Table to plot as an optional fourth row
        plot_title: Optional[str]
            Title to use for the plot
        """
        # Time series processing
        timestamps = np.linspace(0, len(vibes) / self.fs, len(vibes))
        amps = self.rolling_window_fft(vibes, self._window_duration, stride=1, ignore_dc=True)
        # Weight average based on shape of frequency spectrum
        if self._freq_weights is not None:
            amps = amps * self._freq_weights[:, np.newaxis]
        energy = np.average(amps, axis=0)
        # Cross correlate step model with energy
        if self._step_model is not None:
            energy = np.correlate(energy, self._step_model, mode='same')
            model_autocorr = np.correlate(self._step_model, self._step_model, mode='valid')
            energy /= np.max(model_autocorr)
        max_sig = np.max(energy)
        if self._min_signal is not None and max_sig < self._min_signal:
            self.logger.debug(f"Signal ({max_sig:.3f}) is less than threshold ({self._min_signal:.3f}), ignoring")
            if not plot: # Cleans up the logs and speeds things up
                return [], []
        # Step detection
        confirmed_threshold, uncertain_threshold, reset_threshold = self._get_energy_thresholds(max_sig)
        confirmed_indices = self.get_peak_indices(energy, confirmed_threshold, reset_threshold)
        uncertain_indices = self.get_peak_indices(energy, uncertain_threshold, reset_threshold)
        # Drop uncertain steps that are too close to confirmed steps, defined by reset_threshold
        uncertain_indices = self.drop_uncertain_steps(energy, uncertain_indices, confirmed_indices, reset_threshold)
        confirmed_stamps = timestamps[confirmed_indices]
        uncertain_stamps = timestamps[uncertain_indices]
        if plot:
            titles = ("Raw Timeseries", "Scrolling FFT", "Average Energy")
            fig = make_subplots(
                rows=3 if plot_table is None else 4,
                cols=1,
                shared_xaxes=True,
                subplot_titles=titles,
                specs=[[{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'table'}]] if plot_table is not None else None
            )
            if plot_title is not None:
                fig.update_layout(title=plot_title, showlegend=False)
            fig.add_scatter(x=timestamps, y=vibes, name='vibes', row=1, col=1)
            freqs = self.get_window_fft_freqs()
            fig.add_heatmap(x=timestamps, y=freqs, z=amps, row=2, col=1)
            fig.add_scatter(x=timestamps, y=energy, name='energy', row=3, col=1)
            thresholds = {'C': confirmed_threshold, 'U': uncertain_threshold, 'R': reset_threshold}
            if self._min_signal is not None:
                thresholds['Min'] = self._min_signal
            for i, (symbol, threshold) in enumerate(thresholds.items(), 2):
                fig.add_hline(y=threshold, row=3, col=1)
                fig.add_annotation(x=i/20, y=threshold + max_sig/20, text=symbol, showarrow=False, row=3, col=1)
            DC = np.mean(vibes)
            offset = 0.05 * np.max(vibes)
            if truth is not None:
                for timestamp in truth:
                    fig.add_vline(x=timestamp + 0.05, line_color='green', row=1, col=1)
                    fig.add_annotation(x=timestamp, y=DC + offset, xshift=-17, text="Step", showarrow=False, row=1, col=1)
            for confirmed in confirmed_stamps:
                fig.add_vline(x=confirmed, line_dash="dash", row=1, col=1)
                fig.add_annotation(x=confirmed, y=DC - offset, xshift=-10, text="C", showarrow=False, row=1, col=1)
            for uncertain in uncertain_stamps:
                fig.add_vline(x=uncertain, line_dash="dot", row=1, col=1)
                fig.add_annotation(x=uncertain, y=DC - offset, xshift=-10, text="U", showarrow=False, row=1, col=1)
            if plot_table is not None:
                fig.add_table(header=dict(values=plot_table.columns), cells=dict(values=plot_table.values.T), row=4, col=1)
            fig.show()
        if self._min_signal is not None and max_sig < self._min_signal:
            return [], []
        return confirmed_stamps, uncertain_stamps

    def _get_energy_thresholds(self, max_sig: float):
        """
        Calculates the thresholds for confirmed, uncertain and reset steps
        based on the energy of the time series and the noise floor
        """
        noise_max = np.max(self._noise)
        noise_std = np.std(self._noise)
        parameters = (max_sig, noise_max, noise_std, 1)
        confirmed_threshold = np.dot(self._confirm_coefs, parameters)
        uncertain_threshold = np.dot(self._unconfirm_coefs, parameters)
        reset_threshold = np.dot(self._reset_coefs, parameters)
        if uncertain_threshold > confirmed_threshold:
            self.logger.debug(f"Uncertain threshold ({uncertain_threshold:.3f}) is greater than confirmed threshold ({confirmed_threshold:.3f})")
            uncertain_threshold = confirmed_threshold
        if reset_threshold > uncertain_threshold:
            self.logger.debug(f"Reset threshold ({reset_threshold:.3f}) is greater than uncertain threshold ({uncertain_threshold:.3f})")
            reset_threshold = uncertain_threshold
        self.logger.debug(f"Thresholds: confirmed={confirmed_threshold:.3f}, uncertain={uncertain_threshold:.3f}, reset={reset_threshold:.3f}")
        return confirmed_threshold, uncertain_threshold, reset_threshold

    def _resolve_step_sections(self, confirmed_stamps: np.ndarray, uncertain_stamps: np.ndarray = []) -> List[np.ndarray]:
        """Groups confirmed steps into sections, ignoring unconfirmed steps. Sections must have at least 3 steps."""
        all_steps = np.concatenate([confirmed_stamps, uncertain_stamps])
        if len(set(all_steps)) != len(all_steps):
            raise ValueError("All step stamps must be unique")
        # Creating a series of confirmed steps
        confirmed = pd.Series([True] * len(confirmed_stamps), index=confirmed_stamps)
        if len(uncertain_stamps):
            unconfirmed_steps = pd.Series([False] * len(uncertain_stamps), index=uncertain_stamps)
            confirmed = pd.concat([confirmed, unconfirmed_steps])
        confirmed = confirmed.sort_index()

        # Upgrading unconfirmed steps to confirmed steps if there's only one unconfirmed step between two confirmed steps
        for prev_step, current_step, next_step in zip(confirmed.index[:-2], confirmed.index[1:-1], confirmed.index[2:]):
            if not confirmed[current_step] and confirmed[prev_step] and confirmed[next_step]:
                self.logger.debug(f"Upgrading unconfirmed step at {current_step}")
                confirmed[current_step] = True

        # Grouping confirmed steps into sections
        current_section = 0
        section_indices = [current_section] * len(confirmed)
        for i, (prev_step, current_step) in enumerate(zip(confirmed.iloc[:-1], confirmed.iloc[1:]), 1):
            if current_step and not prev_step:
                current_section += 1
            section_indices[i] = current_section
        steps = pd.DataFrame({'confirmed': confirmed, 'section': section_indices}, index=confirmed.index)
        step_dist = steps.confirmed.value_counts()
        if False in step_dist:
            self.logger.debug(f"Ignoring {step_dist[False]} unconfirmed steps")
        steps = steps[steps.confirmed] # Ignore unconfirmed steps that were not upgraded
        if not len(steps):
            return []
        steps = steps.groupby('section').filter(lambda x: len(x) >= 3) # Ignore sections with less than 3 steps
        sections = [group.index.values for _, group in steps.groupby('section')]
        return sections

    def _enforce_min_step_delta(self, step_groups: List[np.ndarray]) -> List[np.ndarray]:
        """Drop steps that are too close together"""
        return [self._enforce_min_step_delta_single_group(steps) for steps in step_groups]

    def _enforce_min_step_delta_single_group(self, steps: np.ndarray):
        steps_too_close = np.diff(steps) < self._min_step_delta
        if np.any(steps_too_close):
            bad_step_indices = np.insert(steps_too_close, 0, False)
            self.logger.debug(f"Enforcing min step delta, removed {bad_step_indices.sum()} step(s) at t={steps[bad_step_indices]}")
            return steps[~bad_step_indices]
        return steps

    def _enforce_max_step_delta(self, step_groups: List[np.ndarray], max_step_delta=None) -> List[np.ndarray]:
        """Splits up steps that are too far apart"""
        max_step_delta = max_step_delta if max_step_delta is not None else self._max_step_delta
        new_step_groups = []
        for i, steps in enumerate(step_groups):
            step_diffs = np.diff(steps)
            split_indices = np.where(step_diffs > max_step_delta)[0] + 1
            if len(split_indices):
                self.logger.debug(f"Enforcing max step delta: splitting up group {i} before steps {split_indices}")
                split_indices = np.insert(split_indices, 0, 0) # Add outer bounds so the data can be sliced properly
                split_indices = np.append(split_indices, len(steps))
                for start, end in zip(split_indices[:-1], split_indices[1:]):
                    new_step_groups.append(steps[start:end])
            else:
                new_step_groups.append(steps)
        return new_step_groups
