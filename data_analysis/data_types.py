
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
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
    path: WalkPath
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
        events = [Event(**event) for event in data['events']]
        return cls(env, events, np.asarray(data['ts']))

    def to_yaml(self, filename: str):
        yaml = YAML()
        yaml.dump(self.to_dict(), Path(filename))

    def to_dict(self):
        return {
            'env': self.env.to_dict(),
            'events': [event.to_dict() for event in self.events],
            'ts': self.ts
        }


class Metrics:
    def __init__(self, timestamps: np.ndarray):
        self.sections = 1
        self.step_count = len(timestamps)
        self.temporal_asymmetry = self._get_temporal_asymmetry(timestamps)
        self.cadence = self._get_cadence(timestamps)
        # self.gait_type = self._get_gait_type(timestamps)

    def _get_temporal_asymmetry(self, timestamps: np.ndarray):
        if len(timestamps) < 3:
            return np.nan
        step_durations = np.diff(timestamps)
        return np.abs(np.mean(step_durations[1:] / step_durations[:-1]) - 1) / np.mean(step_durations)

    def _get_cadence(self, timestamps: np.ndarray):
        if len(timestamps) < 2:
            return np.nan
        return 1 / np.mean(np.diff(timestamps))

    def _get_gait_type(self, timestamps: np.ndarray):
        raise NotImplementedError()

    def __add__(self, other: 'Metrics'):
        """Combines two Metrics objects by averaging their values."""
        if not isinstance(other, Metrics):
            raise ValueError('Can only add Metrics to Metrics.')
        for key in self.__dict__.keys():
            if key == 'step_count':
                self.__dict__[key] += other.__dict__[key]
            else:
                self.__dict__[key] = np.nanmean([self.__dict__[key], other.__dict__[key]])
        self.sections += 1
        return self

    def error(self, truth: 'Metrics') -> 'Metrics':
        """Returns the % error between two Metrics objects."""
        if not isinstance(truth, Metrics):
            raise ValueError('Can only compare Metrics to Metrics.')
        error = Metrics(np.zeros(0))
        for key in self.__dict__.keys():
            if key == 'step_count':
                error.__dict__[key] = np.abs(self.__dict__[key] - truth.__dict__[key])
            else:
                error.__dict__[key] = np.abs(self.__dict__[key] - truth.__dict__[key]) / truth.__dict__[key]
        return error

    def __str__(self) -> str:
        metrics = [f'{key}: {value:.3f}' for key, value in self.__dict__.items()]
        return ', '.join(metrics)
