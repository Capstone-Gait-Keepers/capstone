
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from plotly import graph_objects as go
from ruamel.yaml import YAML
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




class SensorType(Enum):
    PIEZO = 'piezo'
    ACCEL = 'bno055'


def get_model_recording(sensor_type: SensorType, fs: int) -> Recording:
    """Returns a model recording for the given sensor type, hand picked for its quality."""
    if not isinstance(sensor_type, SensorType):
        raise ValueError(f'Invalid sensor type "{sensor_type}".')
    file_map = {
        SensorType.PIEZO: {
            200: 'datasets/piezo/2024-02-11_18-28-53.yaml',
            500: 'datasets/piezo_custom/1.1-alt.yml',
        },
        SensorType.ACCEL: {
            100: 'datasets/bno055/2023-11-09_18-42-33.yaml',
        },
    }
    return Recording.from_file(file_map[sensor_type][fs])


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
            'window_duration': 0.15, # window_duration
            'min_step_delta': 0.3,  # min_step_delta
            'max_step_delta': 1.2,  # max_step_delta
            'confirm_coefs': [0.07, 0, 0, 0.013], # confirmed
            'unconfirm_coefs': [0.04, 0, 0, 0.013], # unconfirmed
            'reset_coefs': [0, 0, 0, 0.013], # reset
        },
        # Loss=0.135
        # {
        #     'window_duration': 0.15917045728367368,
        #     'min_step_delta': 0.2997857786406394,
        #     'max_step_delta': 1.6592433964815125,
        #     'confirm_coefs': [0.0019137034298423172, 0.45932149582073245, 0, 0.008695670392463106],
        #     'unconfirm_coefs': [0.001490315322084881, 0.357701121453802, 0, 0.006771838634764541],
        #     'reset_coefs': [0.0010589, 0, 0, 0.008]
        # }
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
        # {'window_duration': 0.1528245877697354, 'min_signal': 0.009775206784765858, 'min_step_delta': 0.6002018071406573, 'max_step_delta': 1.3097133385347495, 'confirm_coefs': [0.3307906011750318, 0.3307653564083102, 1.113442691369579, 0.0061192363028736985], 'unconfirm_coefs': [0.04492663010156339, 1.827242358777819, 0.2522668781282773, 0.7158149543809781], 'reset_coefs': [0.5989798589222364, 1.9959103922131698, 0.6011319760688445, 1.9103389380818625]}
    ],
}


def get_optimal_analysis_params(sensor_type: SensorType, fs: int, include_model=True, version=-1) -> dict:
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
        params['model'] = get_model_recording(sensor_type, fs)
    return params
