import type { Metrics, User } from "./types";
import store from './store.js'


export async function queryBackend<T>(endpoint: String, params = {}): Promise<T | null> {
  const url = import.meta.env.VITE_BACKEND_URL;
  if (!url) {
    console.error("VITE_BACKEND_URL not defined. Please define it in .env file.");
    return null;
  }
  if (!endpoint || endpoint === "" || endpoint[0] !== "/") {
    console.error("Invalid endpoint: " + endpoint + ". Endpoint must start with /.");
    return null;
  }
  console.debug("Retrieving data from backend: " + url + endpoint);
  try {
    let full_url = url + endpoint;
    if (Object.keys(params).length > 0) {
      full_url += '?' + new URLSearchParams(params);
    }
    const response = await fetch(full_url);
    if (response.status < 200 || response.status >= 300) {
      console.error("Failed to fetch data from backend. Status: " + response.status + " " + response.statusText);
      return null;
    }
    return response.json();
  } catch (error) {
    console.error(error);
    return null;
  }
}

export async function signup(user: User) {
  store.user = await queryBackend<User>("/signup", {email: user.email, password: user.password, sensor_id: user.sensor_id});
  return true;
}

// Returns full User object
export async function login(email: string, password: string) {
  store.user = await queryBackend<User>("/login");
  return true
}

export async function getMetrics(startDate: string | null = null, endDate: string | null = null): Promise<Metrics | null> {
  const email = store.user?.email;
  if (!email) {
    console.error("User not logged in.");
    return null;
  }
  // return queryBackend<Metrics>("/api/metrics", {email, startDate, endDate});
  return {
    cadence: [{date: '2021-01-01', value: 0.5}, {date: '2021-01-02', value: 0.48}, {date: '2021-01-03', value: 0.47}, {date: '2021-01-04', value: 0.46}, {date: '2021-01-05', value: 0.45}],
    stride_time: [{date: '2021-01-01', value: 0.9}, {date: '2021-01-02', value: 0.98}, {date: '2021-01-03', value: 0.97}, {date: '2021-01-04', value: 0.96}, {date: '2021-01-05', value: 0.45}],
    var_coef: [{date: '2021-01-01', value: 0.5}, {date: '2021-01-02', value: 0.48}, {date: '2021-01-03', value: 0.47}, {date: '2021-01-04', value: 0.46}, {date: '2021-01-05', value: 0.45}],
    stga: [{date: '2021-01-01', value: 0.5}, {date: '2021-01-02', value: 2.48}, {date: '2021-01-03', value: 2.47}, {date: '2021-01-04', value: 2.46}, {date: '2021-01-05', value: 2.45}],
    cond_entropy: [{date: '2021-01-01', value: 2.5}, {date: '2021-01-02', value: 0.48}, {date: '2021-01-03', value: 0.47}, {date: '2021-01-04', value: 0.46}, {date: '2021-01-05', value: 0.45}],
    phase_sync: [{date: '2021-01-01', value: 0.5}, {date: '2021-01-02', value: 0.48}, {date: '2021-01-03', value: 4.47}, {date: '2021-01-04', value: 0.46}, {date: '2021-01-05', value: 0.45}],
  };
}

export async function getUserStatus(email: string) {} // For insight page, crunches metrics on backend
