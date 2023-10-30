import os
import numpy as np
import plotly.graph_objects as go
from data_types import Recording, Event
from plotly.subplots import make_subplots
from typing import List


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


def rolling_window(a: np.ndarray, window_size: int, stride: int = None):
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


def rolling_window_fft(a: np.ndarray, window_duration: int, stride: int, fs: float):
    """
    Creates a rolling window of a time series and computes the FFT of each window
    """
    window_size = timestamp_to_index(window_duration, fs)
    intervals = rolling_window(a, window_size, stride)
    # TODO: Better condition for tapering?
    if stride < window_size:
        tapering_window = np.vstack([np.blackman(window_size)] * len(intervals))
        intervals *= tapering_window
    rolling_fft = np.fft.fft(intervals, axis=-1).T[:window_size//2]
    timestamps = np.linspace(0, len(a) / fs, len(a))
    fft_timestamps = timestamps[::stride] + window_duration/2 # Add half window duration to center the window
    amps = np.abs(rolling_fft)
    freqs = np.fft.fftfreq(window_size, 1/fs)[:window_size//2] # Ignore negative frequencies
    return fft_timestamps, freqs, amps



def view_datasets(dataset_dir: str = "datasets", **filters):
    """Walk through datasets folder and plot all csv files"""
    filepaths = [file for root, dirs, files in os.walk(dataset_dir) for file in files if file.endswith(".yaml")]
    filepaths.remove('example.yaml')
    datasets = [Recording.from_file(os.path.join(dataset_dir, filename)) for filename in filepaths]

    # Filter datasets based on environment filters
    for key, value in filters.items():
        datasets = [dataset for dataset in datasets if getattr(dataset.env, key) == value]

    fig = make_subplots(rows=len(datasets), cols=1, shared_xaxes=True)
    fig.update_layout(title=str(filters), showlegend=False)
    for i, data in enumerate(datasets, start=1):
        timestamps = np.linspace(0, len(data.ts) / data.env.fs, len(data.ts))
        fig.add_trace(go.Scatter(x=timestamps, y=data.ts, name='vibes'), row=i, col=1)
        # TODO: Better text to identify between datasets. Drop entries that are common to all datasets
        env_vars = data.env.to_dict()
        env_vars = {key: value for key, value in env_vars.items() if value is not None}
        for y, (key, value) in enumerate(env_vars.items(), start=-len(env_vars) // 2):
            fig.add_annotation(x=timestamps[len(timestamps)//2], y=y/80, text=f"{key} = {value}", xshift=0, showarrow=False, row=i, col=1)
    fig.show()


def get_steps_from_truth(data: Recording, step_duration=0.35, shift_percent=0.2):
    """
    Uses the source of truth to parse the accelerometer data pertaining to steps
    """
    step_measurements = []
    for event in data.events:
        if event.category == 'step':
            start = timestamp_to_index(event.timestamp - step_duration * shift_percent, data.env.fs)
            end = start + timestamp_to_index(step_duration, data.env.fs)
            step_measurements.append(data.ts[start:end])
    return step_measurements


def get_noise_floor(data: Recording):
    """
    Find the noise floor of the accelerometer data
    """
    first_step = data.events[0].timestamp # TODO: Assumes the first event is a step
    noisy_starting_data = data.ts[:timestamp_to_index(first_step, data.env.fs)]
    noisy_starting_data = noisy_starting_data - np.mean(noisy_starting_data)
    # fig = go.Figure()
    # fig.add_scatter(x=np.linspace(0, len(noisy_starting_data) / data.env.fs, len(noisy_starting_data)), y=noisy_starting_data)
    # fig.show()
    rms = np.sqrt(np.mean(noisy_starting_data**2))
    return rms


def get_frequency_weights(data: Recording, window_duration=0.2, plot=False):
    """
    Find the dominant frequencies pertaining to steps
    """
    DC = np.mean(data.ts, axis=0)
    step_data = get_steps_from_truth(data)
    amp_per_step_freq_time = []

    for step in step_data:
        fft_timestamps, freqs, amps = rolling_window_fft(step - DC, window_duration, 1, data.env.fs)
        amp_per_step_freq_time.append(amps)
    amp_per_step_freq_time = np.asarray(amp_per_step_freq_time)
    amp_per_step_freq = np.mean(amp_per_step_freq_time, axis=-1)
    amp_per_freq = np.mean(amp_per_step_freq, axis=0)
    amp_per_freq /= np.max(amp_per_freq)

    if plot:
        num_rows = int(np.ceil(np.sqrt(len(step_data))))
        fig = make_subplots(rows=num_rows, cols=num_rows, column_titles=['Step Heat Maps (Freq vs Time)'])
        for i, amps in enumerate(amp_per_step_freq_time, start=1):
            fig.add_heatmap(x=fft_timestamps, y=freqs, z=amps, row=(i // num_rows) + 1, col=(i % num_rows) + 1)
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


def find_steps(data: Recording, window_duration=0.2, stride=1, amp_threshold=0.3, freq_weights=None, plot=False) -> int:
    """
    Counts the number of steps in a time series of accelerometer data. This function should not use
    anything from `data.events` except for plotting purposes. This is because it is meant to mimic
    a blind step detection algorithm.

    Parameters
    ----------
    data : Recording
        Time series of accelerometer data, plus the environmental data
    window_duration : int, optional
        Duration of the rolling window in seconds
    stride : int, optional
        Number of samples to move the window by. If None, stride is equal to window duration
    amp_threshold : float, optional
        Minimum amplitude of a peak to be considered a step
    freq_weights : np.ndarray, optional
        Weights to apply to each frequency when averaging the frequency spectrum
    """
    vibes = data.ts
    fs = data.env.fs
    timestamps = np.linspace(0, len(vibes) / fs, len(vibes))
    ### Time series processing
    p_vibes = vibes - np.mean(vibes, axis=0) # Subtract DC offset
    fft_timestamps, freqs, amps = rolling_window_fft(p_vibes, window_duration, stride, fs)
    ### Frequency domain processing
    # Weight average based on shape of frequency spectrum
    # TODO: Ensure weights correspond to correct frequencies
    if freq_weights is None:
        freq_weights = np.ones_like(freqs)
    # TODO: Minimum step duration? Ignore steps that are too close together?
    energy = np.average(amps, axis=0, weights=freq_weights)
    de_dt = np.gradient(energy)
    optima = np.diff(np.sign(de_dt), prepend=[0]) != 0
    peak_indices = np.where((energy > amp_threshold) & optima)[0] * stride # Rescale to original time series
    peak_stamps = fft_timestamps[peak_indices]
    # TODO: Find start of step within confirmed window?

    if plot:
        titles = ("Raw Timeseries", "Scrolling FFT", "Average Energy")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=titles)
        fig.add_scatter(x=timestamps, y=vibes, name='vibes', showlegend=False, row=1, col=1)
        fig.add_heatmap(x=fft_timestamps, y=freqs, z=amps, row=2, col=1)
        fig.add_scatter(x=fft_timestamps, y=energy, name='energy', showlegend=False, row=3, col=1)
        fig.add_hline(y=amp_threshold, row=3, col=1)
        for event in data.events:
            fig.add_vline(x=event.timestamp, line_color='green', row=1, col=1)
            fig.add_annotation(x=event.timestamp, y=0, xshift=-17, text="Step", showarrow=False, row=1, col=1)
        for peak in peak_stamps:
            fig.add_vline(x=peak, line_dash="dash", row=1, col=1)
        # fig.write_html("normal_detection.html")
        fig.show()

    # TODO: Hysteresis: Count small steps if they are between two large steps
    # TODO: Only count multiple confirmed steps?
    return peak_stamps


def get_temporal_asymmetry(step_timestamps: List[float]):
    """
    Calculates the temporal asymmetry of a list of step timestamps. 
    """
    step_durations = np.diff(step_timestamps)
    return np.abs(np.mean(step_durations[1:] / step_durations[:-1]) - 1)


def get_cadence(step_timestamps: List[float]):
    """
    Calculates the cadence of a list of step timestamps. 
    """
    step_durations = np.diff(step_timestamps)
    return 1 / np.mean(step_durations)


def get_algorithm_error(measured_step_times: List[float], events: List[Event]):
    """
    Calculates the algorithm error of a recording. There are three types of errors:
    - Incorrect measurements: The algorithm found a step when there was none (False Positive)
    - Missed steps: The algorithm missed a step (False Negative)
    - Measurement error: The algorithm found a step correctly, but at the wrong time

    Parameters
    ----------
    measured_step_times : List[float]
        List of timestamps where the algorithm found steps
    events : List[Event]
        List of events from the source of truth
    """
    if not len(measured_step_times):
        raise ValueError("Algorithm did not find any steps")
    true_step_times = [event.timestamp for event in events if event.category == 'step']
    errors = []
    missed_steps = 0
    selected_measurements = []
    for step_stamp in true_step_times:
        measurement_error = np.abs(measured_step_times - step_stamp)
        best_measurement = np.argmin(measurement_error)
        # If the measurement is already the best one for another step, it means we missed this step
        if best_measurement in selected_measurements:
            missed_steps += 1
        else:
            selected_measurements.append(best_measurement)
            errors.append(measurement_error[best_measurement])
    incorrect_measurements = len(measured_step_times) - len(selected_measurements)
    return {
        "error": np.mean(errors),
        "incorrect": incorrect_measurements,
        "missed": missed_steps
    }

if __name__ == "__main__":
    freqs, weights = get_frequency_weights(Recording.from_file('datasets/2023-10-29_18-16-34.yaml'), plot=False)
    data = Recording.from_file('datasets/2023-10-29_18-16-34.yaml')
    # data = Recording.from_file('datasets/2023-10-29_18-20-13.yaml')
    steps = find_steps(data, amp_threshold=10 * get_noise_floor(data), freq_weights=weights, plot=True)
    print(f"Asymmetry: {get_temporal_asymmetry(steps) * 100:.2f} %")
    print(f"Steps/s: {get_cadence(steps):.2f}")
    print(f"Algorithm error: {get_algorithm_error(steps, data.events)}")
    # view_datasets(walk_type='normal')
