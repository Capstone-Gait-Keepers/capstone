
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
class SensorEnvironment:
    fs: float

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def keys():
        return RecordingEnvironment.__dataclass_fields__.keys()


@dataclass
class RecordingEnvironment(SensorEnvironment):
    location: str
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


@dataclass
class Recording:
    env: SensorEnvironment 
    events: list[Event] = field(default_factory=list)
    ts: np.ndarray = field(default_factory=np.zeros(0))
    tag: Optional[str] = None
    sensor_type: Optional[str] = None

    def __post_init__(self):
        self.ts = np.array(self.ts).squeeze()
        if len(self.ts) and self.ts.ndim != 1:
            raise ValueError(f'ts must be a 1D array, not {self.ts.ndim}D.')

    @classmethod
    def from_real_data(cls, fs: float, data: np.ndarray, tag=None):
        return cls(SensorEnvironment(fs), events=[], ts=data, tag=tag)

    @classmethod
    def from_file(cls, filename: str):
        try:
            yaml = YAML()
            with open(filename) as file:
                data = yaml.load(file)
            rec = cls.from_dict(data)
            rec.tag = filename
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
        timestamps = np.linspace(0, len(self.ts) / self.env.fs, len(self.ts), endpoint=False)
        fig = go.Figure()
        if self.tag:
            fig.update_layout(title=self.tag, showlegend=False)
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
        func_map = self.get_func_map()
        self.keys = self.get_keys()
        if len(timestamp_groups) == 0:
            self._df = pd.DataFrame({key: [np.nan] for key in self.keys})
            self._df['step_count'] = [0]
        else:
            data = {key: [func_map[key](timestamps) for timestamps in timestamp_groups] for key in self.keys}
            self._df = pd.DataFrame.from_dict(data)
        self._df['recording_id'] = [recording_id] * len(self._df)

    @classmethod
    def get_func_map(cls):
        return {
            'step_count': len,
            'STGA': cls._get_STGA,
            'stride_time': cls._get_stride_time,
            'cadence': cls._get_cadence,
            'var_coef': cls._get_stride_time_CV,
            'phase_sync': cls._get_phase_sync,
            'conditional_entropy': cls._get_conditional_entropy,
        }

    @staticmethod
    def get_control() -> pd.DataFrame:
        return pd.DataFrame.from_dict({
            'step_count': [np.inf],
            'STGA': [0],
            'stride_time': [1.036],
            'cadence': [1.692],
            'var_coef': [0.017],
            'phase_sync': [0.812],
            'conditional_entropy': [0.007],
        })

    @classmethod
    def get_keys(cls):
        return list(cls.get_func_map().keys())

    def __getitem__(self, key):
        return self.aggregate_metric(self._df[key], weights=self._df['step_count'])

    @staticmethod
    def aggregate_metric(data: pd.Series, weights=None):
        if len(data) == 0 or data.isna().all() or (weights is not None and weights.isna().all()):
            return np.nan
        if data.name in Metrics.summed_vars:
            return data.sum()
        return Metrics.nanaverage(data, weights=weights)

    @staticmethod
    def nanaverage(a: np.ndarray, weights: np.ndarray):
        return np.nansum(a*weights)/((~np.isnan(a))*weights).sum()

    def __len__(self):
        return len(self._df)

    @property
    def recordings(self):
        return self._df['recording_id'].nunique()

    @staticmethod
    def _get_STGA(timestamps: np.ndarray):
        if len(timestamps) < 4:
            return np.nan
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
        a, b = Metrics._get_side_stride_times(timestamps)
        stride_times = np.empty((a.size + b.size,), dtype=timestamps.dtype)
        stride_times[0::2] = a
        stride_times[1::2] = b
        return stride_times    

    @staticmethod
    def _get_side_stride_times(timestamps: np.ndarray) -> np.ndarray:
        if len(timestamps) < 2:
            return np.empty((0))
        a = np.diff(timestamps[::2])
        b = np.diff(timestamps[1::2])
        return a, b  

    @staticmethod
    def _get_var_coef(dist):
        """General formula for coefficient of variation"""
        if len(dist) < 3:
            return np.nan
        return np.std(dist) / np.mean(dist)

    @staticmethod
    def _get_phase_sync(timestamps: np.ndarray, num_bins=8):
        if len(timestamps) < 8:
            return np.nan
        if len(timestamps) % 2 != 0:
            timestamps = np.copy(timestamps)[:-1]
        left_stride_times, right_stride_times = Metrics._get_side_stride_times(timestamps)
        analytic_signal1 = hilbert(left_stride_times)
        analytic_signal2 = hilbert(right_stride_times)
        phase1 = np.unwrap(np.angle(analytic_signal1))
        phase2 = np.unwrap(np.angle(analytic_signal2))
        phase_difference = phase1 - phase2
        H = Metrics._calculate_shannon_entropy(phase_difference, num_bins)
        H_max = np.log2(num_bins)
        return (H_max - H) / H_max

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7247739
    @staticmethod
    def _get_conditional_entropy(timestamps: np.ndarray, num_bins=8):
        if len(timestamps) < 6:
            return np.nan
        if len(timestamps) % 2 != 0:
            timestamps = np.copy(timestamps)[:-1]
        left_stride_times, right_stride_times = Metrics._get_side_stride_times(timestamps)
        left_cond_entropy = Metrics._calculate_cond_entropy(left_stride_times, right_stride_times, num_bins)
        right_cond_entropy = Metrics._calculate_cond_entropy(right_stride_times, left_stride_times, num_bins)
        return (left_cond_entropy + right_cond_entropy) / 2

    @staticmethod
    def _calculate_shannon_entropy(values: np.ndarray, num_bins: int, bin_range=(-np.pi, np.pi)):
        if len(values) <= 1:
            raise ValueError('Stride times must have at least 2 elements.')
        bins = Metrics._get_hist_bins(values, num_bins, bin_range)
        counts, bins = np.histogram(values, bins=bins)
        probabilities = counts / sum(counts)
        return entropy(probabilities, base=2)

    @staticmethod
    def _calculate_cond_entropy(S1: np.ndarray, S2: np.ndarray, num_bins: int, bin_range=(0, 3)):
        """
        Calculate the conditional entropy H(S1|S2) given two 1D numpy arrays S1 and S2.
        """
        # Joint distribution of S1 and S2
        bins = Metrics._get_hist_bins(np.concatenate([S1, S2]), num_bins, bin_range)
        p_joint, _, _ = np.histogram2d(S1, S2, bins=bins, density=True)
        p_S2 = np.sum(p_joint, axis=0) # Marginal distribution of S2
        H_S1_S2 = entropy(p_joint.flatten(), base=2) # H(S1, S2)
        H_S2 = entropy(p_S2, base=2) # H(S2)
        H_S1_given_S2 = H_S1_S2 - H_S2 # H(S1 | S2)
        return H_S1_given_S2

    @staticmethod
    def _get_hist_bins(values, num_bins: int, bin_range: tuple):
        if num_bins > len(values):
            num_bins = len(values) - 1
        bin_range = (min(values.min(), bin_range[0]), max(values.max(), bin_range[1]))
        return np.linspace(*bin_range, num_bins)

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
        self._df = pd.concat([self._df, other._df], ignore_index=True, sort=False)
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
        self_rec_metrics = self.by_tag()
        truth_rec_metrics = truth.by_tag()
        error = abs(self_rec_metrics - truth_rec_metrics)
        # Normalize for non-summed metrics
        for key in Metrics.get_keys():
            if key not in Metrics.summed_vars:
                error[key] = error[key] / truth_rec_metrics[key]
        # Where they are both NaN, the error is 0
        error = error.where(error != np.inf, np.nan)
        error = error.where(truth_rec_metrics.notna() | self_rec_metrics.notna(), 0)
        # Add classification bool
        correct_class = (self_rec_metrics['step_count'] > 0) ^ (truth_rec_metrics['step_count'] == 0)
        correct_class.name = "correct_class"
        error = pd.concat([correct_class, error], axis=1)
        return error

    def by_tag(self, smooth_window=0) -> pd.DataFrame:
        df = self._df.groupby('recording_id').apply(self.aggregate)
        if smooth_window:
            df = df.rolling(smooth_window, min_periods=1).median()
        return df

    def set_index(self, new_ids: list):
        old_ids = self._df['recording_id'].unique()
        if len(new_ids) != len(old_ids):
            raise ValueError(f'New IDs ({len(new_ids)}) must be the same length as old IDs ({len(old_ids)}).')
        self._df['recording_id'].replace(dict(zip(old_ids, new_ids)), inplace=True)

    @staticmethod
    def aggregate(data: pd.DataFrame):
        """
        Aggregates a group of metrics into a single row. All metrics are assumed to be
        from the same recording. The `summed_vars` are summed, while all other metrics
        are averaged, weighted by the step count.
        """
        if len(data) == 0:
            return pd.Series({key: np.nan for key in Metrics.get_keys()})
        results = {}
        for metric in Metrics.get_keys():
            results[metric] = Metrics.aggregate_metric(data[metric], weights=data['step_count'])
        return pd.Series(results)

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


PARAM_MAP = {
    SensorType.PIEZO: [
        # false-negative: 0%, false-positive: 61.54%
        {
            'window_duration': 0.1139349947424978,
            'min_signal': 0.037631118631130805,
            'min_step_delta': 0.18095205271502524,
            'max_step_delta': 1.950418846273037,
            'confirm_coefs': [0.5512649631395441, 0.24808347703417744, 0.2171562044598988, 0.0],
            'unconfirm_coefs': [0.29895567028655945, 0.7053939275078811, 0.05568592936569661, 0.0],
            'reset_coefs': [0.25883540025668905, 0.9598419926532735, 0.03623646301128214, 1.365225936022818]
        },
        # false-negative: 46.67%, false-positive: 30.77%
        {
            'window_duration': 0.5,
            'min_signal': 0.0,
            'min_step_delta': 0.0,
            'max_step_delta': 1.94702008044308,
            'confirm_coefs': [0.2877633076936741, 0.4879086903112724, 0.9106027824886926, 0.0],
            'unconfirm_coefs': [0.48855200391108067, 0.6020246859906139, 0.2541307658769687, 0.0],
            'reset_coefs': [0.034887749913138366, 1.121288775057767, 0.0, 0.0]
        },
        # false-negative: 46.67%, false-positive: 23.08%
        {
            'window_duration': 0.2, # window_duration
            'min_signal': 0.01,  # min_signal
            'min_step_delta': 0.1,  # min_step_delta
            'max_step_delta': 2,  # max_step_delta
            'confirm_coefs': [0.4, 0.3, 0, 0], # confirmed
            'unconfirm_coefs': [0.2, 0.65, 0, 0], # unconfirmed
            'reset_coefs': [0, 0.7, 0, 0], # reset
        },
    ],
    SensorType.ACCEL: [
        # false-negative: 63.08%, false-positive: 61.54%
        {
            'window_duration': 0.2, # window_duration
            'min_signal': 0.05,  # min_signal
            'min_step_delta': 0.1,  # min_step_delta
            'max_step_delta': 2,  # max_step_delta
            'confirm_coefs': [0.5, 0.3, 0, 0], # confirmed
            'unconfirm_coefs': [0.25, 0.65, 0, 0], # unconfirmed
            'reset_coefs': [0, 1, 0, 0], # reset
        },
        # false-negative: 26.15%, false-positive: 76.92%
        {
            'window_duration': 0.39383795370402874,
            'min_signal': 0.11874029008975184,
            'min_step_delta': 0.006114890298333564,
            'max_step_delta': 0.9996980505314539,
            'confirm_coefs': [0.4634656640188022, 0.15375031628153313, 0.05074975259937076, 0.0],
            'unconfirm_coefs': [0.19269100597190514, 0.6527249453428609, 0.7010081078892039, 0.4687034237688712],
            'reset_coefs': [0.38434914601277703, 0.9407486648938483, 0.6447618736934766, 0.6469422341301987]
        },
        # false-negative: 72.31%, false-positive: 0%
        {
            'window_duration': 0.06997036182119981,
            'min_signal': 0.6282611648972323,
            'min_step_delta': 0.37596442685765374,
            'max_step_delta': 0.49160770343425664,
            'confirm_coefs': [0.11193261417632372, 0.4320858954031941, 1.893098538808411, 0.07908746553703738],
            'unconfirm_coefs': [1.3976705681092856, 1.5234063263359925, 0.7101148865664979, 1.5667306931088163],
            'reset_coefs': [0.9161274358669074, 1.5974889872123756, 1.5278097857114015, 0.6556597273881369]
        },
    ],
}


def get_optimal_analysis_params(sensor_type: SensorType, include_model=True, version=-1) -> dict:
    """
    Returns the optimal analysis parameters for the given sensor type, based on previous optimize.py results.
    False rates are based on all available data for the given sensor type.

    Parameters
    ----------
    sensor_type : SensorType
        The sensor type to get the optimal parameters for.
    include_model : bool
        Whether to include the model recording in the returned parameters.
    version : int
        The version of the parameters to return. -1 returns the latest version.
    """
    params = PARAM_MAP[sensor_type][version]
    if include_model:
        params['model'] = get_model_recording(sensor_type)
    return params
