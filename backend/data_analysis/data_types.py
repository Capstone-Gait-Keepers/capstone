
import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from plotly import graph_objects as go
from ruamel.yaml import YAML
from scipy.signal import hilbert
from scipy.stats import entropy
from typing import List, Optional



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

    @property
    def tangent_distance(self) -> float:
        """Compute the closest distance to the sensor along the path using Heron's Formula."""
        # h=0.5*sqrt(a+b+c)*sqrt(-a+b+c)*sqrt(a-b+c)*sqrt(a+b-c)/b
        a, b, c = self.start, self.length, self.stop
        return 0.5 * np.sqrt(a + b + c) * np.sqrt(-a + b + c) * np.sqrt(a - b + c) * np.sqrt(a + b - c) / b



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

    @staticmethod
    def keys():
        return RecordingEnvironment.__dataclass_fields__.keys()


@dataclass
class Recording:
    env: RecordingEnvironment
    events: list[Event] = field(default_factory=list)
    ts: np.ndarray = field(default_factory=np.zeros(0))
    filepath: Optional[str] = None
    sensor_type: Optional[str] = None

    @classmethod
    def from_file(cls, filename: str):
        try:
            yaml = YAML()
            with open(filename) as file:
                data = yaml.load(file)
            rec = cls.from_dict(data)
            rec.filepath = filename
            rec.sensor_type = SensorType.PIEZO if 'piezo' in filename else SensorType.ACCEL
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

    def to_file(self, filename: str):
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

    def __init__(self, timestamp_groups: List[np.ndarray], recording_id=0):
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
            self._df['recording_id'] = [recording_id]
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

    @property
    def recordings(self):
        return self._df['recording_id'].nunique()

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

    @staticmethod
    def _get_stride_time(timestamps):
        stride_times = Metrics._get_stride_times(timestamps)
        if len(stride_times):
            return np.mean(stride_times)
        return np.nan

    @staticmethod
    def _get_stride_time_CV(timestamps: np.ndarray):
        return Metrics._get_var_coef(Metrics._get_stride_times(timestamps))

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
            raise ValueError(f'Can only add Metrics to Metrics, not {type(other)}. ({other})')
        if not len(self):
            return other
        if not len(other):
            return self
        last_recording_id = self._df['recording_id'].max()
        other._df['recording_id'] += last_recording_id + 1
        self._df = pd.concat([self._df, other._df])
        return self

    def __radd__(self, other: 'Metrics | int'):
        """Combines two Metrics objects by averaging their values."""
        return self if other == 0 else self.__add__(other)

    def error(self, truth: 'Metrics') -> pd.DataFrame:
        """Returns the % error between two Metrics objects. Groups by recording_id."""
        if not isinstance(truth, Metrics):
            raise ValueError('Can only compare Metrics to Metrics.')
        if truth.recordings != self.recordings:
            raise ValueError(f'Cannot compare Metrics of different lengths. truth.recordings ({len(truth.recordings)}) != self.recordings ({self.recordings}) (len(self) = {len(self)})')
        self_rec_metrics = self.by_recordings()
        truth_rec_metrics = truth.by_recordings()
        error = abs(self_rec_metrics - truth_rec_metrics) / truth_rec_metrics
        # Where they are both NaN, the error is 0
        error = error.where(truth_rec_metrics.notna() | self_rec_metrics.notna(), 0)
        return error

    def by_recordings(self) -> pd.DataFrame:
        # TODO: Not all attributes should be averaged. Some should be summed.
        rec_groups = self._df.groupby('recording_id').mean()
        rec_metrics = rec_groups.filter(items=self.keys, axis=1)
        return rec_metrics

    def __str__(self) -> str:
        return str(self._df)



class SensorType(Enum):
    PIEZO = 'piezo'
    ACCEL = 'bno055'


def get_model_recording(sensor_type: SensorType) -> Recording:
    """Returns a model recording for the given sensor type, hand picked for its quality."""
    if not isinstance(sensor_type, SensorType):
        raise ValueError(f'Invalid sensor type "{sensor_type}".')
    file_map = {
        SensorType.PIEZO: 'datasets/piezo/2024-02-11_18-28-53.yaml',
        SensorType.ACCEL: 'datasets/bno055/2023-11-09_18-42-33.yaml',
    }
    return Recording.from_file(file_map[sensor_type])


def get_optimal_analysis_params(sensor_type: SensorType, include_model=True) -> dict:
    """Returns the optimal analysis parameters for the given sensor type, based on previous optimize.py results."""
    param_map = {
        SensorType.PIEZO: {},
        SensorType.ACCEL: {
            'window_duration': 0.2927981091746967,
            'min_signal': 0.06902195485649608,
            'min_step_delta': 0.7005074596681514,
            'max_step_delta': 1.7103077671127291,
            'confirm_coefs': [0.13795802814939168, 0.056480535457810385, 1.2703933010798438, 0.0384835095362413],
            'unconfirm_coefs': [1.0670316188983877, 1.0511076985832117, 1.160496215083792, 1.6484084554908836],
            'reset_coefs': [0.7869793593332175, 1.6112694921747566, 0.12464680752843472, 1.1399207966364366]
        },
    }
    params = param_map[sensor_type]
    if include_model:
        params['model'] = get_model_recording(sensor_type)
    return params
