import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from data_types import Recording, Event
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional


def timestamp_to_index(timestamp: float, fs: float) -> int:
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
    return np.floor(timestamp * fs).astype(int)


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


def rolling_window_fft(a: np.ndarray, window_duration: int, fs: float, stride: int, ignore_dc=False) -> np.ndarray:
    """
    Creates a rolling window of a time series and computes the FFT of each window
    """
    window_size = timestamp_to_index(window_duration, fs)
    padded_a = np.pad(a, (window_size // 2, window_size // 2), mode='edge')
    intervals = rolling_window(padded_a, window_size, stride)
    if ignore_dc:
        intervals -= np.mean(intervals, axis=1, keepdims=True)
    # TODO: Better condition for tapering?
    if stride < window_size:
        tapering_window = np.vstack([np.blackman(window_size)] * len(intervals))
        intervals *= tapering_window
    rolling_fft = np.fft.fft(intervals, axis=-1).T[:window_size//2]
    return np.abs(rolling_fft)


def get_window_fft_freqs(window_duration, fs) -> np.ndarray:
    window_size = timestamp_to_index(window_duration, fs)
    return np.fft.fftfreq(window_size, 1/fs)[:window_size//2] # Ignore negative frequencies


def get_datasets(dataset_dir: str = "datasets", **filters) -> List[Recording]:
    """Walk through datasets folder and return all records that match the filters"""
    filepaths = [file for root, dirs, files in os.walk(dataset_dir) for file in files if file.endswith(".yaml")]
    filepaths.remove('example.yaml')
    datasets = [Recording.from_file(os.path.join(dataset_dir, filename)) for filename in filepaths]
    for key, value in filters.items():
        datasets = [dataset for dataset in datasets if getattr(dataset.env, key) == value]
    return datasets


def view_datasets(dataset_dir: str = "datasets", **filters):
    """Walk through datasets folder and plot all recording that match the filters"""
    datasets = get_datasets(dataset_dir, **filters)
    fig = make_subplots(rows=len(datasets), cols=1, shared_xaxes=True)
    fig.update_layout(title=str(filters), showlegend=False)
    for i, data in enumerate(datasets, start=1):
        timestamps = np.linspace(0, len(data.ts) / data.env.fs, len(data.ts))
        fig.add_trace(go.Scatter(x=timestamps, y=data.ts, name='vibes'), row=i, col=1)
        # TODO: Better text to identify between datasets
        env_vars = data.env.to_dict()
        env_vars = {key: value for key, value in env_vars.items() if value is not None and key not in filters}
        for y, (key, value) in enumerate(env_vars.items(), start=-len(env_vars) // 2):
            fig.add_annotation(x=timestamps[len(timestamps)//2], y=y/80, text=f"{key} = {value}", xshift=0, showarrow=False, row=i, col=1)
    fig.show()


def get_steps_from_truth(data: Recording, step_duration=0.4, shift_percent=0.2, align_peak=True, plot=False) -> List[np.ndarray]:
    """
    Uses the source of truth to parse the accelerometer data pertaining to steps
    """
    offset = int(shift_percent * data.env.fs / 2)
    window_size = timestamp_to_index(step_duration, data.env.fs)
    step_measurements = []
    for event in data.events:
        if event.category == 'step':
            start = timestamp_to_index(event.timestamp, data.env.fs) - offset
            step_data = data.ts[start : start+window_size]
            if align_peak:
                energy = get_energy(step_data, data.env.fs, step_duration / 5)
                start += np.argmax(energy) - offset
                step_data = data.ts[start : start+window_size]
            if start + window_size > len(data.ts):
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


def get_noise(data: Recording, plot=False) -> np.ndarray:
    """
    Find the noise floor of the accelerometer data
    """
    first_event = data.events[0].timestamp
    noise = data.ts[:timestamp_to_index(first_event, data.env.fs)]
    if plot:
        timestamps = np.linspace(0, len(noise) / data.env.fs, len(noise))
        fig = go.Figure()
        fig.update_layout(title="Noise", showlegend=False)
        fig.add_scatter(x=timestamps, y=noise)
        fig.show()
    return noise


def get_energy_thresholds(data: Recording, plot=False, **kwargs) -> Tuple[float, float]:
    noise = get_noise(data)
    sig_energy = get_energy(data.ts, data.env.fs, **kwargs)
    noise_energy = get_energy(noise, data.env.fs, **kwargs)
    max_sig = np.max(sig_energy)
    max_noise = np.max(noise_energy)
    # TODO: Parameterize weights of max_sig, max_noise and np.std(noise) to find optimal thresholds
    confirmed_threshold = np.mean([max_sig, max_noise])
    uncertain_threshold = np.mean([confirmed_threshold, max_noise]) # Weighted average of max signal (1/4) and noise (3/4)
    if plot:
        fig = go.Figure()
        fig.update_layout(title="Energy", showlegend=False)
        fig.add_scatter(x=np.linspace(0, len(data.ts) / data.env.fs, len(data.ts)), y=sig_energy)
        fig.add_hline(y=confirmed_threshold, line_dash="dash")
        fig.add_hline(y=uncertain_threshold, line_dash="dash")
        fig.show()
    return uncertain_threshold, confirmed_threshold


def get_snr(data: Recording) -> float:
    """
    Find the signal-to-noise ratio of the accelerometer data
    """
    noise_var = np.var(get_noise(data))
    signal_var = np.var(np.concatenate(get_steps_from_truth(data)))
    snr = signal_var / noise_var
    return snr


def get_energy(vibes: np.ndarray, fs, window_duration=0.2, stride=1, weights=None) -> np.ndarray:
    """
    Find the energy of the accelerometer data
    """
    p_vibes = vibes - np.mean(vibes, axis=0) # Subtract DC offset
    amps = rolling_window_fft(p_vibes, window_duration, fs, stride)
    energy = np.average(amps, axis=0, weights=weights)
    if energy.shape != vibes.shape:
        raise ValueError(f"Energy is the wrong shape: {energy.shape} != {vibes.shape}")
    return energy


def plot_recording(filepath: str):
    """Plot a recording from a YAML file"""
    data = Recording.from_file(filepath)
    timestamps = np.linspace(0, len(data.ts) / data.env.fs, len(data.ts))
    fig = go.Figure()
    fig.update_layout(title=filepath, showlegend=False)
    fig.add_scatter(x=timestamps, y=data.ts, name='vibes')
    for event in data.events:
        fig.add_vline(x=event.timestamp, line_color='green')
        fig.add_annotation(x=event.timestamp, y=0, xshift=-17, text=event.category, showarrow=False)
    fig.show()


def get_frequency_weights(data: Recording, window_duration=0.2, plot=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the dominant frequencies pertaining to steps
    """
    DC = np.mean(data.ts, axis=0)
    step_data = get_steps_from_truth(data, window_duration*2)
    freqs = get_window_fft_freqs(window_duration, data.env.fs)
    amp_per_step_freq_time = []

    for step in step_data:
        amps = rolling_window_fft(step - DC, window_duration, data.env.fs, stride=1)
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
    return freqs, amp_per_freq


def get_step_model(data: Recording, plot_model=False, plot_steps=False) -> np.ndarray:
    """Creates a model of step energy vs time"""
    step_data = get_steps_from_truth(data)
    energy_per_step = []
    window_duration = 0.2
    for step in step_data:
        # TODO: Use weights from get_frequency_weights
        energy = get_energy(step, data.env.fs, window_duration)
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


def find_steps(
        data: Recording,
        confirmed_threshold: float,
        uncertain_threshold: Optional[float] = None,
        window_duration=0.2,
        stride=1,
        step_model=None,
        freq_weights=None,
        plot=False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Counts the number of steps in a time series of accelerometer data. This function should not use
    anything from `data.events` except for plotting purposes. This is because it is meant to mimic
    a blind step detection algorithm.

    Parameters
    ----------
    data : Recording
        Time series of accelerometer data, plus the environmental data
    uncertain_threshold : float
        Threshold for the energy of a step to be considered uncertain
    confirmed_threshold : float
        Threshold for the energy of a step to be considered confirmed
    window_duration : int, optional
        Duration of the rolling window in seconds
    stride : int, optional
        Number of samples to move the window by. If None, stride is equal to window duration
    step_model : np.ndarray, optional
        Model of step energy vs time. Model is cross correlated with the energy of the time series
    freq_weights : np.ndarray, optional
        Weights to apply to each frequency when averaging the frequency spectrum
    """
    vibes = data.ts
    fs = data.env.fs
    timestamps = np.linspace(0, len(vibes) / fs, len(vibes))
    ### Time series processing
    amps = rolling_window_fft(vibes, window_duration, fs, stride, ignore_dc=True)
    ### Frequency domain processing
    # Weight average based on shape of frequency spectrum
    # TODO: Ensure weights correspond to correct frequencies
    # TODO: Minimum step duration? Ignore steps that are too close together?
    energy = np.average(amps, axis=0, weights=freq_weights)
    # Cross correlate step model with energy
    if step_model is not None:
        energy = np.correlate(energy, step_model, mode='same')
        model_autocorr = np.correlate(step_model, step_model, mode='valid')
        energy /= np.max(model_autocorr)
    confirmed_indices = get_peak_indices(energy, confirmed_threshold)
    uncertain_indices = get_peak_indices(energy, uncertain_threshold) if uncertain_threshold is not None else []
    uncertain_indices = np.setdiff1d(uncertain_indices, confirmed_indices)    
    confirmed_stamps = timestamps[confirmed_indices] * stride # Rescale to original time series
    uncertain_stamps = timestamps[uncertain_indices] * stride
    # TODO: Find start of step within confirmed window?

    if plot:
        titles = ("Raw Timeseries", "Scrolling FFT", "Average Energy")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=titles)
        fig.add_scatter(x=timestamps, y=vibes, name='vibes', showlegend=False, row=1, col=1)
        freqs = get_window_fft_freqs(window_duration, fs)
        fig.add_heatmap(x=timestamps, y=freqs, z=amps, row=2, col=1)
        fig.add_scatter(x=timestamps, y=energy, name='energy', showlegend=False, row=3, col=1)
        for threshold in (confirmed_threshold, uncertain_threshold):
            fig.add_hline(y=threshold, row=3, col=1)
        for event in data.events:
            fig.add_vline(x=event.timestamp + 0.05, line_color='green', row=1, col=1)
            fig.add_annotation(x=event.timestamp + 0.05, y=0, xshift=-17, text="Step", showarrow=False, row=1, col=1)
        for confirmed in confirmed_stamps:
            fig.add_vline(x=confirmed, line_dash="dash", row=1, col=1)
            fig.add_annotation(x=confirmed, y=-0.3, xshift=-10, text="C", showarrow=False, row=1, col=1)
        for uncertain in uncertain_stamps:
            fig.add_vline(x=uncertain, line_dash="dot", row=1, col=1)
            fig.add_annotation(x=uncertain, y=-0.3, xshift=-10, text="U", showarrow=False, row=1, col=1)
        # fig.write_html("normal_detection.html")
        fig.show()

    # TODO: Hysteresis: Count small steps if they are between two large steps
    # TODO: Only count multiple confirmed steps?
    return confirmed_stamps, uncertain_stamps


def get_peak_indices(signal: np.ndarray, threshold: float):
    """Find the indices of the peaks in a time series above a certain threshold"""
    de_dt = np.gradient(signal)
    optima = np.diff(np.sign(de_dt), prepend=[0]) != 0
    indices = np.where((signal > threshold) & optima)[0]
    return indices


def resolve_step_sections(confirmed_stamps: np.ndarray, uncertain_stamps: np.ndarray = [], min_step_delta=0.05) -> List[np.ndarray]:
    """Groups confirmed steps into sections, ignoring unconfirmed steps. Sections must have at least 3 steps."""
    all_steps = np.concatenate([confirmed_stamps, uncertain_stamps])
    if len(set(set(all_steps))) != len(all_steps):
        raise ValueError("All step stamps must be unique")
    if not np.all(np.diff(confirmed_stamps) > min_step_delta):
        raise ValueError(f"Confirmed step stamps must be at least {min_step_delta}s apart")

    confirmed = pd.Series([True] * len(confirmed_stamps), index=confirmed_stamps)
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
    return sections


# TODO: Include possible steps
def get_temporal_asymmetry(step_timestamps: np.ndarray):
    """
    Calculates the temporal asymmetry of a list of step timestamps. 
    """
    step_durations = np.diff(step_timestamps)
    return np.abs(np.mean(step_durations[1:] / step_durations[:-1]) - 1)


# TODO: Include possible steps
def get_cadence(step_timestamps: np.ndarray):
    """
    Calculates the cadence of a list of step timestamps. 
    """
    step_durations = np.diff(step_timestamps)
    return 1 / np.mean(step_durations)


def get_algorithm_error(measured_step_times: np.ndarray, events: List[Event]):
    """
    Calculates the algorithm error of a recording. There are three types of errors:
    - Incorrect measurements: The algorithm found a step when there was none (False Positive)
    - Missed steps: The algorithm missed a step (False Negative)
    - Measurement error: The algorithm found a step correctly, but at the wrong time

    Parameters
    ----------
    measured_step_times : np.ndarray
        List of timestamps where the algorithm found steps
    events : List[Event]
        List of events from the source of truth
    """
    if not len(measured_step_times):
        raise ValueError("Algorithm did not find any steps")
    true_step_times = [event.timestamp for event in events if event.category == 'step']
    missed_steps = 0
    measurement_errors = {}
    for step_stamp in true_step_times:
        possible_errors = np.abs(measured_step_times - step_stamp)
        best_measurement = np.argmin(possible_errors)
        # If the measurement is already the best one for another step, it means we missed this step
        if best_measurement in measurement_errors:
            missed_steps += 1
        else:
            measurement_errors[best_measurement] = possible_errors[best_measurement]
    incorrect_measurements = len(measured_step_times) - len(measurement_errors)
    errors = [*measurement_errors.values()]
    return {
        "error": np.mean(errors),
        "stderr": np.std(errors),
        "incorrect": incorrect_measurements,
        "missed": missed_steps
    }


def get_metric_error(measured_times: np.ndarray, events: List[Event]):
    """Calculates the % metric error of a measured timestamp list, given the source of truth."""
    metrics = {
        "asymmetry": get_temporal_asymmetry,
        "cadence": get_cadence,
    }
    correct_times = [event.timestamp for event in events if event.category == 'step']
    return {metric: get_func_metric_error(func, correct_times, measured_times) for metric, func in metrics.items()}


def get_func_metric_error(func: callable, correct_times: np.ndarray, measured_times: np.ndarray):
    """Calculates the % error between the two timestamp lists, given a function to compute the metric."""
    correct_metric = func(correct_times)
    measured_metric = func(measured_times)
    return np.abs(measured_metric - correct_metric) / correct_metric


# TODO
def get_gait_type(data: Recording):
    ...


if __name__ == "__main__":
    # model_data = Recording.from_file('datasets/2023-10-29_18-16-34.yaml')
    model_data = Recording.from_file('datasets/2023-10-29_18-20-13.yaml')
    freqs, weights = get_frequency_weights(model_data, plot=False)
    step_model = get_step_model(model_data, plot_model=False, plot_steps=False)
    uncertain_threshold, confirmed_threshold = get_energy_thresholds(model_data, plot=False)

    # data = Recording.from_file('datasets/2023-10-29_18-16-34.yaml')
    data = Recording.from_file('datasets/2023-10-29_18-20-13.yaml')
    steps, uncertain_steps = find_steps(data,
        confirmed_threshold,
        uncertain_threshold,
        freq_weights=weights,
        # step_model=step_model,
        plot=True
    )
    print("Step Stamps:", steps)
    print(f"Asymmetry: {get_temporal_asymmetry(steps) * 100:.2f} %")
    print(f"Steps/s: {get_cadence(steps):.2f}")
    print(f"Algorithmic error: {get_algorithm_error(steps, data.events)}")
    print(f"Metric error: {get_metric_error(steps, data.events)}")
    step_groups = resolve_step_sections(steps, uncertain_steps)
    if not len(step_groups):
        print("No valid step sections found")
    for steps in step_groups:
        print("Step Stamps:", steps)
        print(f"Asymmetry: {get_temporal_asymmetry(steps) * 100:.2f} %")
        print(f"Steps/s: {get_cadence(steps):.2f}")
        print(f"Algorithmic error: {get_algorithm_error(steps, data.events)}")
        print(f"Metric error: {get_metric_error(steps, data.events)}")

    # view_datasets(walk_type='normal', user='ron', walk_speed='normal', footwear='socks')

    # datasets = get_datasets(walk_type='normal', user='ron', walk_speed='normal', footwear='socks')
    # for data in datasets:
    #     steps = find_steps(data, amp_threshold=5 * get_noise_variance(data), freq_weights=weights, plot=True)
