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
  sensor_id: number;
}

export interface Metrics {
  metrics: Record<string, number[]>;
  dates: string[];
}
