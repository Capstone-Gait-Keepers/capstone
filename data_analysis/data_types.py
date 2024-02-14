
import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from plotly import graph_objects as go
from ruamel.yaml import YAML
from scipy.signal import hilbert
from scipy.stats import entropy
from typing import Optional



@dataclass
class Event:
    category: str
    timestamp: float # Seconds since the start of the recording
    distance: Optional[float] = None
    duration: Optional[float] = None
    
    def to_dict(self):
        data = deepcopy(self.__dict__)
        if self.distance is None:
            del data['distance']
        if self.duration is None:
            del data['duration']
        data['timestamp'] = round(float(data['timestamp']), 5)
        return data


# Path the user walked. Assumed to be a straight line. Measurements define a triangle :)
@dataclass
class WalkPath:
    start: float # Start distance from sensor (m)
    stop: float # End distance from sensor (m)
    length: float # Total length of the walk (m)


@dataclass
class RecordingEnvironment:
    location: str
    fs: float
    user: str
    floor: str
    footwear: str
    walk_type: str
    obstacle_radius: float
    wall_radius: float
    path: Optional[WalkPath] = None
    quality: str = 'normal'
    walk_speed: str = 'normal'
    temperature: Optional[float] = None # in Celsius
    notes: str = ''

    def __post_init__(self):
        valid_walks = ['normal', 'stomp', 'shuffle', 'limp', 'griddy']
        valid_footwear = ['barefoot', 'shoes', 'socks', 'slippers']
        if self.walk_type not in valid_walks:
            raise ValueError(f'Invalid walk type "{self.walk_type}". Valid walks are: {valid_walks}')
        if self.footwear not in valid_footwear:
            raise ValueError(f'Invalid footwear "{self.footwear}". Valid footwear are: {valid_footwear}')

    def to_dict(self):
        return asdict(self)


@dataclass
class Recording:
    env: RecordingEnvironment
    events: list[Event] = field(default_factory=list)
    ts: np.ndarray = field(default_factory=np.zeros(0))
    filepath: Optional[str] = None

    @classmethod
    def from_file(cls, filename: str):
        try:
            yaml = YAML()
            with open(filename) as file:
                data = yaml.load(file)
            rec = cls.from_dict(data)
            rec.filepath = filename
            return rec
        except Exception as e:
            raise ValueError(f'Failed to load recording from "{filename}".') from e

    @classmethod
    def from_dict(cls, data: dict):
        env = RecordingEnvironment(**data['env'])
        if data['env']['path'] is not None:
            env.path = WalkPath(**data['env']['path'])
        events = [Event(**event) for event in data['events']]
        return cls(env, events, np.asarray(data['ts']))

    def to_yaml(self, filename: str):
        yaml = YAML()
        yaml.dump(self.to_dict(), Path(filename))

    def to_dict(self):
        return {
            'env': self.env.to_dict(),
            'events': [event.to_dict() for event in self.events],
            'ts': self.ts.tolist()
        }
    
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


class Metrics:
    """
    A collection of the following gait metrics:

    - Step Count: Total number of steps recorded.
    - Stride Time Gait Asymmetry (STGA): A measure of the difference in stride time between the left and right legs.
    - Stride Time: The average time between steps of the same foot.
    - Cadence: The average number of steps per second.
    - Stride Time Coefficient of Variation: A measure of the regularity of the stride time. Low values indicate high regularity, which is good.
    - Stride Time Phase Synchronization Index: A measure of the phase synchronization between the left and right legs. High values indicate high phase synchronization, which is good.
    - Stride Time Conditional Entropy: A measure of the regularity of the stride time. Low values indicate high regularity, which is good
    """

    summed_vars = set(['step_count'])

    def __init__(self, *timestamp_groups: np.ndarray, recording_id=0):
        func_map = {
            'step_count': len,
            'STGA': self._get_STGA,
            'stride_time': self._get_stride_time,
            'cadence': self._get_cadence,
            'var_coef': self._get_stride_time_CV,
            'phase_sync': self._get_phase_sync,
            'conditional_entropy': self._get_conditional_entropy,
        }
        self.keys = list(func_map.keys())
        if len(timestamp_groups) == 0:
            self._df = pd.DataFrame({key: [np.nan] for key in self.keys})
            return
        data = {key: [func_map[key](timestamps) for timestamps in timestamp_groups] for key in self.keys}
        self._df = pd.DataFrame.from_dict(data)
        self._df['recording_id'] = [recording_id] * len(self._df)

    def __getitem__(self, key):
        if len(self._df) == 0:
            return np.nan
        if key in self.summed_vars:
            return np.sum(self._df[key].values)
        return np.average(self._df[key].values, weights=self._df['step_count'].values)

    def __len__(self):
        return len(self._df)

    @staticmethod
    def _get_STGA(timestamps: np.ndarray):
        if len(timestamps) < 3:
            return np.nan
        # TODO: Update stride time definition
        stride_times = Metrics._get_stride_times(timestamps)
        # TODO: Does this match literature?
        # https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=aeee9316f2a72d0f89e59f3c5144bf69a695730b
        return np.abs(np.mean(stride_times[1:] / stride_times[:-1]) - 1) / np.mean(stride_times)

    @staticmethod
    def _get_cadence(timestamps: np.ndarray):
        if len(timestamps) < 2:
            return np.nan
        return 1 / np.mean(np.diff(timestamps))

    def _get_stride_time(self, timestamps):
        stride_times = self._get_stride_times(timestamps)
        if len(stride_times):
            return np.mean(stride_times)
        return np.nan

    def _get_stride_time_CV(self, timestamps: np.ndarray):
        return self._get_var_coef(self._get_stride_times(timestamps))

    @staticmethod
    def _get_stride_times(timestamps: np.ndarray) -> np.ndarray:
        if len(timestamps) < 2:
            return np.empty((0))
        # TODO: Update stride time definition
        return np.diff(timestamps)

    @staticmethod
    def _get_var_coef(dist):
        if len(dist) < 3:
            return np.nan
        """General formula for coefficient of variation"""
        return np.std(dist) / np.mean(dist)

    @staticmethod
    def _get_phase_sync(timestamps: np.ndarray, num_bins=40):
        if len(timestamps) < 4:
            return np.nan
        if len(timestamps) % 2 != 0:
            timestamps = np.copy(timestamps)[:-1]
        timestamps_right_foot = timestamps[1::2]
        timestamps_left_foot = timestamps[::2]
        analytic_signal1 = hilbert(timestamps_left_foot)
        analytic_signal2 = hilbert(timestamps_right_foot)
        phase1 = np.unwrap(np.angle(analytic_signal1))
        phase2 = np.unwrap(np.angle(analytic_signal2))
        phase_difference = phase1 - phase2
        H = Metrics._calculate_shannon_entropy(phase_difference, num_bins)
        H_max = np.log2(num_bins)
        return (H_max - H) / H_max

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7247739
    @staticmethod
    def _get_conditional_entropy(timestamps: np.ndarray):
        if len(timestamps) < 4:
            return np.nan
        timestamps_left_foot = timestamps[::2]
        timestamps_right_foot = timestamps[1::2]
        stride_times_left_foot = np.diff(timestamps_left_foot)
        stride_times_right_foot = np.diff(timestamps_right_foot)
        shannon_entropy_left_foot = Metrics._calculate_shannon_entropy(stride_times_left_foot)
        shannon_entropy_right_foot = Metrics._calculate_shannon_entropy(stride_times_right_foot)
        return (shannon_entropy_left_foot + shannon_entropy_right_foot) / 2

    @staticmethod
    def _calculate_shannon_entropy(stride_times: np.ndarray, num_bins=40):
        counts, _ = np.histogram(stride_times, bins=num_bins)
        probabilities = counts / sum(counts)
        return entropy(probabilities, base=2)

    def __add__(self, other: 'Metrics'):
        """Combines two Metrics objects by averaging their values."""
        if not isinstance(other, Metrics):
            raise ValueError('Can only add Metrics to Metrics.')
        if not len(self):
            return other
        if not len(other):
            return self
        self._df = pd.concat([self._df, other._df])
        return self

    def error(self, truth: 'Metrics') -> pd.DataFrame:
        """Returns the % error between two Metrics objects. Groups by recording_id."""
        if not isinstance(truth, Metrics):
            raise ValueError('Can only compare Metrics to Metrics.')
        if len(truth) != self._df['recording_id'].nunique():
            raise ValueError(f'Cannot compare Metrics of different lengths. {len(truth)} != {self._df["recording_id"].nunique()} ({len(self)})')
        metric_groups = self._df.groupby('recording_id').mean()
        metric_groups = metric_groups.filter(items=self.keys, axis=1)
        truth_groups = truth._df.groupby('recording_id').mean()
        truth_groups = truth_groups.filter(items=self.keys, axis=1)
        error = np.abs(metric_groups - truth_groups) / truth_groups
        return error

    def __str__(self) -> str:
        return str(self._df)

# TODO: Why can't I just do sum :(
def concat_metrics(metrics_list: list[Metrics]) -> Metrics:
    """Concatenates a list of Metrics objects into one."""
    if not len(metrics_list):
        raise ValueError('Cannot concatenate an empty list of Metrics.')
    for i, m in enumerate(metrics_list):
        if not isinstance(m, Metrics):
            raise ValueError('Can only concatenate Metrics objects.')
        m._df['recording_id'] = [i] * len(m._df)
    m = metrics_list[0]
    m._df = pd.concat([new_m._df for new_m in metrics_list])
    return m
