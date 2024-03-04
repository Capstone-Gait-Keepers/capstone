export interface SensorConfig {
  id: number;
  userid: number;
  model: string;
  floor: string;
  last_timestamp: string;
  num_recordings: number;
}

export interface User {
  userid: string;
  username: string;
  email: string;
  password: string;
}
