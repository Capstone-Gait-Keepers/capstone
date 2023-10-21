
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from ruamel.yaml import YAML
from typing import Optional


class Source(Enum):
    USER = 'user'
    PET = 'pet'
    OBJECT = 'object'
    CANE = 'cane'


@dataclass
class Event:
    category: str
    timestamp: float # Seconds since the start of the recording
    source: Source
    distance: Optional[float] = None
    
    def to_dict(self, include_source=True):
        data = deepcopy(self.__dict__)
        if include_source:
            data['source'] = self.source.value
        else:
            del data['source']
        if self.distance is None:
            del data['distance']
        return data


@dataclass
class RecordingEnvironment:
    location: str
    fs: float
    floor: str
    obstacle_radius: float
    wall_radius: float
    version: int = 1 # ! Update the version if the data format changes
    sources: list[Source] = field(default_factory=lambda: [Source.USER])

    def to_dict(self):
        data = deepcopy(self.__dict__)
        data['sources'] = [source.value for source in data['sources']]
        return data


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
        if len(env.sources) == 0:
            raise ValueError("No sources specified in the recording environment")
        events = [Event(**event) for event in data['events']]
        if len(env.sources) == 1:
            for event in events:
                event.source = env.sources[0]
        return cls(env, events, data['ts'])
    
    def to_yaml(self, filename: str):
        yaml = YAML()
        yaml.dump(self.to_dict(), Path(filename))

    def to_dict(self):
        include_source = len(self.env.sources) > 1
        return {
            'env': self.env.to_dict(),
            'events': [event.to_dict(include_source) for event in self.events],
            'ts': self.ts
        }
