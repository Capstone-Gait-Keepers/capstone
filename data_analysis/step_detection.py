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


def rolling_window_fft(a: np.ndarray, window_duration: int, fs: float, stride: int):
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
    return np.abs(rolling_fft)


def get_window_fft_freqs(window_duration, fs):
    window_size = timestamp_to_index(window_duration, fs)
    return np.fft.fftfreq(window_size, 1/fs)[:window_size//2] # Ignore negative frequencies


def get_window_timestamps(window_duration, fs, stride=1):
    window_size = timestamp_to_index(window_duration, fs)
    timestamps = np.linspace(0, window_size / fs, window_size)
    return timestamps[::stride] + window_duration/2 # Add half window duration to center the window


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


def get_steps_from_truth(data: Recording, step_duration=0.5, shift_percent=0.3, align_peak=True):
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
    return step_measurements


def get_noise_floor(data: Recording):
    """
    Find the noise floor of the accelerometer data
    """
    first_event = data.events[0].timestamp
    noisy_starting_data = data.ts[:timestamp_to_index(first_event, data.env.fs)]
    noisy_starting_data -= np.mean(noisy_starting_data)
    # fig = go.Figure()
    # fig.add_scatter(x=np.linspace(0, len(noisy_starting_data) / data.env.fs, len(noisy_starting_data)), y=noisy_starting_data)
    # fig.show()
    rms = np.sqrt(np.mean(noisy_starting_data**2))
    return rms


def get_energy(vibes: np.ndarray, fs, window_duration=0.2, stride=1, weights=None):
    """
    Find the energy of the accelerometer data
    """
    p_vibes = vibes - np.mean(vibes, axis=0) # Subtract DC offset
    amps = rolling_window_fft(p_vibes, window_duration, fs, stride)
    energy = np.average(amps, axis=0, weights=weights)
    # Prepend half of the window duration with zeroes to center the window
    padding = int(window_duration * fs / 2)
    energy = np.pad(energy, (padding, padding), mode='constant')
    if energy.shape != vibes.shape:
        raise ValueError(f"Energy is the wrong shape: {energy.shape} != {vibes.shape}")
    return energy



def get_frequency_weights(data: Recording, window_duration=0.2, plot=False):
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
        fft_timestamps = get_window_timestamps(window_duration, data.env.fs)
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


def get_step_model(data: Recording, plot=False):
    """"""
    step_data = get_steps_from_truth(data)
    energy_per_step = []
    window_duration = 0.2
    for step in step_data:
        energy = get_energy(step, data.env.fs, window_duration)
        energy_per_step.append(energy / np.max(energy))
    if plot:
        fig = go.Figure()
        fig.update_layout(title="Average Energy vs Time", showlegend=False)
        fft_timestamps = get_window_timestamps(window_duration, data.env.fs)
        for energy in energy_per_step:
            fig.add_scatter(x=fft_timestamps, y=energy)
        fig.show()


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
    amps = rolling_window_fft(p_vibes, window_duration, fs, stride)
    fft_timestamps = get_window_timestamps(window_duration, fs, stride)
    ### Frequency domain processing
    # Weight average based on shape of frequency spectrum
    # TODO: Ensure weights correspond to correct frequencies
    # TODO: Minimum step duration? Ignore steps that are too close together?
    energy = np.average(amps, axis=0, weights=freq_weights)
    de_dt = np.gradient(energy)
    optima = np.diff(np.sign(de_dt), prepend=[0]) != 0
    peak_indices = np.where((energy > amp_threshold) & optima)[0] * stride # Rescale to original time series
    print(len(fft_timestamps), max(peak_indices))
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
            fig.add_vline(x=event.timestamp + 0.05, line_color='green', row=1, col=1)
            fig.add_annotation(x=event.timestamp + 0.05, y=0, xshift=-17, text="Step", showarrow=False, row=1, col=1)
        for peak in peak_stamps:
            fig.add_vline(x=peak, line_dash="dash", row=1, col=1)
        fig.write_html("normal_detection.html")
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


# TODO
def get_gait_type(data: Recording):
    ...


if __name__ == "__main__":
    # freqs, weights = get_frequency_weights(Recording.from_file('datasets/2023-10-29_18-16-34.yaml'), plot=False)
    data = Recording.from_file('datasets/2023-10-29_18-16-34.yaml')
    # data = Recording.from_file('datasets/2023-10-29_18-20-13.yaml')
    # steps = find_steps(data, amp_threshold=10 * get_noise_floor(data), freq_weights=weights, plot=True)
    # print(f"Asymmetry: {get_temporal_asymmetry(steps) * 100:.2f} %")
    # print(f"Steps/s: {get_cadence(steps):.2f}")
    # print(f"Algorithm error: {get_algorithm_error(steps, data.events)}")

    # view_datasets(walk_type='normal', user='ron', walk_speed='normal', footwear='socks')

    # data = get_datasets(walk_type='normal', user='ron', walk_speed='normal', footwear='socks')
    # steps = find_steps(data[0], amp_threshold=5 * get_noise_floor(data[0]), freq_weights=weights, plot=True)

    get_step_model(data, plot=True)
