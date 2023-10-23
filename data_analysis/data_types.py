
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
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


@dataclass
class RecordingEnvironment:
    location: str
    fs: float
    user: str
    floor: str
    obstacle_radius: float
    wall_radius: float

    def to_dict(self):
        return deepcopy(self.__dict__)


@dataclass
class Recording:
    env: RecordingEnvironment
    events: list[Event] = field(default_factory=list)
    ts: list[float] = field(default_factory=list)

    @classmethod
    def from_file(cls, filename: str):
        yaml = YAML()
        with open(filename) as file:
            data = yaml.load(file)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict):
        env = RecordingEnvironment(**data['env'])
        events = [Event(**event) for event in data['events']]
        return cls(env, events, data['ts'])

    def to_yaml(self, filename: str):
        yaml = YAML()
        yaml.dump(self.to_dict(), Path(filename))

    def to_dict(self):
        return {
            'env': self.env.to_dict(),
            'events': [event.to_dict() for event in self.events],
            'ts': self.ts
        }
