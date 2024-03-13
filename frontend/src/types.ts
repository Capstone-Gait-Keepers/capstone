export interface SensorConfig {
  id: number;
  userid: number;
  model: string;
  floor: string;
  last_timestamp: string;
  num_recordings: number;
}

export interface User {
  name: string;
  email: string;
  password: string;
  sensorid: number;
}

export type MetricSequence = {
  date: string;
  value: number;
}[]

export interface Metrics {
  [key: string]: MetricSequence
}
