import os
import numpy as np
import plotly.graph_objects as go
from data_types import Recording
from plotly.subplots import make_subplots


def rolling_window(a: np.ndarray, window_size: int, overlap: int = 0):
    """
    Creates a rolling window of a time series

    Parameters
    ----------
    a : np.ndarray
        Time series to be windowed
    window_size : int
        Size of the window in samples
    overlap : int
        Number of samples to overlap between windows

    Returns
    -------
    np.ndarray
        Windowed time series
    """
    return np.asarray([a[i:i + window_size] for i in range(0, len(a) - window_size, window_size - overlap)])


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


def get_steps(data: Recording, step_duration=0.2):
    """
    Uses the source of truth to parse the accelerometer data pertaining to steps
    """
    step_measurements = []
    for event in data.events:
        if event.category == 'step':
            start = event.timestamp
            end = start + step_duration
            step_measurements.append(data.ts[start:end])
    return step_measurements


def find_steps(data: Recording, window_duration=0.2, stride=1, amp_threshold=0.3, plot=False) -> int:
    """
    Counts the number of steps in a time series of accelerometer data

    Parameters
    ----------
    data : Recording
        Time series of accelerometer data, plus the environmental data
    window_duration : int, optional
        Duration of the rolling window in seconds
    stride : int, optional
        Number of samples to move the window by. If None, stride is equal to window duration
    """
    vibes = data.ts
    fs = data.env.fs
    window = np.floor(window_duration * fs).astype(int)
    timestamps = np.linspace(0, len(vibes) / fs, len(vibes))
    ### Time series processing
    # TODO: Measure noise floor to determine threshold
    p_vibes = vibes - np.mean(vibes, axis=0) # Subtract DC offset
    if stride is None:
        stride = window # Tiny stride to get more data points
    overlap = window - stride
    intervals = rolling_window(p_vibes, window, overlap)
    if overlap: # TODO: Better condition for tapering?
        tapering_window = np.vstack([np.blackman(window)] * len(intervals))
        intervals *= tapering_window
    ### Frequency domain processing
    rolling_fft = np.fft.fft(intervals, axis=-1).T[:window//2] # Ignore negative frequencies
    fft_timestamps = timestamps[::stride] + window_duration/2 # Add half window duration to center the window
    amps = np.abs(rolling_fft)
    # TODO: Weight average based on shape of frequency spectrum
    energy = np.mean(amps, axis=0)
    de_dt = np.gradient(energy)
    optima = np.diff(np.sign(de_dt), prepend=[0]) != 0
    peak_indices = np.where((energy > amp_threshold) & optima)[0]
    peak_indices *= stride # Rescale to original time series
    peak_stamps = fft_timestamps[peak_indices]
    # TODO: Find start of step within confirmed window

    if plot:
        titles = ("Raw Timeseries", "Scrolling FFT", "Average Energy")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=titles)
        fig.add_scatter(x=timestamps, y=vibes, name='vibes', showlegend=False, row=1, col=1)
        freqs = np.fft.fftfreq(window, 1/fs)[:window//2] # Ignore negative frequencies
        fig.add_trace(go.Heatmap(x=fft_timestamps, y=freqs, z=amps), row=2, col=1)
        fig.add_scatter(x=fft_timestamps, y=energy, name='energy', showlegend=False, row=3, col=1)
        fig.add_hline(y=amp_threshold, row=3, col=1)
        for event in data.events:
            fig.add_vline(x=event.timestamp, line_color='green', row=1, col=1)
            fig.add_annotation(x=event.timestamp, y=0, xshift=-17, text="Step", showarrow=False, row=1, col=1)
        for peak in peak_stamps:
            fig.add_vline(x=peak, line_dash="dash", row=1, col=1)
        # fig.write_html("normal_detection.html")
        fig.add_scatter(x=fft_timestamps, y=de_dt, name='de_dt', showlegend=False, row=4, col=1)
        fig.show()

    # TODO: Hysterisis: Count small steps if they are between two large steps
    # TODO: Only count multiple confirmed steps?
    return peak_stamps



if __name__ == "__main__":
    data = Recording.from_file('datasets/2023-10-29_18-16-34.yaml')
    # data = Recording.from_file('datasets/2023-10-29_18-20-13.yaml')
    print("Steps: ", find_steps(data, amp_threshold=0.15, plot=True))

    # view_datasets(walk_type='normal')
