export interface SensorConfig {
  id: number;
  userid: number;
  model: string;
  floor: string;
  last_timestamp: string;
  num_recordings: number;
}

export interface User {
  email: string;
  password: string;
  sensor_id: number;
}
