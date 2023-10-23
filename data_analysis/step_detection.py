import os
import numpy as np
import plotly.graph_objects as go
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


def view_datasets(*keywords: str, dataset_dir: str = "datasets", fs = 1):
    """Walk through datasets folder and plot all csv files"""
    datasets = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".csv") and all(keyword.lower() in file.lower() for keyword in keywords):
                datasets.append(file)

    fig = make_subplots(rows=len(datasets), cols=1, shared_xaxes=True)
    for i, filename in enumerate(datasets, start=1):
        vibes = np.genfromtxt(os.path.join(root, filename), delimiter=',')
        timestamps = np.linspace(0, len(vibes) / fs, len(vibes))
        fig.add_trace(go.Scatter(x=timestamps, y=vibes, name='vibes'), row=i, col=1)
        fig.update_xaxes(title_text=filename, row=i, col=1)
    fig.show()


def count_steps(vibes: np.ndarray, fs: int, window_duration=0.1, amp_threshold=0.3, plot=False) -> int:
    """
    Counts the number of steps in a time series of accelerometer data

    Parameters
    ----------
    vibes : np.ndarray
        Time series of accelerometer data
    fs : int
        Sampling frequency of the accelerometer data, in Hz
    window_duration : int, optional
        Duration of the rolling window in seconds, by default 1
    """
    window = np.floor(window_duration * fs).astype(int)
    timestamps = np.linspace(0, len(vibes) / fs, len(vibes))
    ### Time series processing
    p_vibes = vibes - np.mean(vibes, axis=0) # Subtract DC offset
    intervals = rolling_window(p_vibes, window)
    ### Frequency domain processing
    rolling_fft = np.fft.fft(intervals, axis=-1).T[:window//2] # Ignore negative frequencies
    amps = np.abs(rolling_fft)
    peaks = np.mean(amps, axis=0)
    if plot:
        freqs = np.fft.fftfreq(window, 1/fs)[:window//2] # Ignore negative frequencies
        titles = ("Raw Timeseries", "Scrolling FFT", "Average Energy")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=titles)
        fig.add_trace(go.Scatter(x=timestamps, y=vibes, name='vibes'), row=1, col=1)
        # fig.add_trace(go.Scatter(x=timestamps, y=p_vibes, name='pvibes'), row=2, col=1)
        fig.add_trace(go.Heatmap(x=timestamps[::window], y=freqs, z=amps), row=2, col=1)
        fig.add_trace(go.Scatter(x=timestamps[::window], y=peaks, name='peaks'), row=3, col=1)
        fig.add_hline(y=amp_threshold, row=3, col=1)
        # fig.write_html("normal_detection.html")
        fig.show()

    # TODO: Count wide peaks as 1 step
    # TODO: Return timestamps of steps
    # Later
    # TODO: Hysterisis: Count small steps if they are between two large steps
    # TODO: Only count multiple confirmed steps?
    return np.count_nonzero(peaks > amp_threshold)



if __name__ == "__main__":
    vibes = np.genfromtxt('datasets/Ron Regular Room Walk Hardwood.csv', delimiter=',', skip_header=True)
    sample_rate_Hz = 100
    print("Steps: ", count_steps(vibes, sample_rate_Hz, plot=True, amp_threshold=0.3))

    # view_datasets("living", "shuffle")
