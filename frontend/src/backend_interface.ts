import type { Metrics, User } from "./types";
import { store } from './store.js'


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
  try {
    let full_url = url + endpoint;
    if (Object.keys(params).length > 0) {
      // Remove null or undefined values from params
      const new_params = Object.fromEntries(Object.entries(params).filter(([_, v]) => v != null)) as Record<string, string>;
      full_url += '?' + new URLSearchParams(new_params);
    }
    console.debug("Retrieving data from backend: " + full_url);
    const response = await fetch(full_url);
    if (response.status < 200 || response.status >= 300) {
      console.error("Failed to fetch data from backend. Status: " + response.status + " " + response.statusText);
      return null;
    }
    // Replace 'null' with null
    return response.json();
    // const replacer = (key: string, value: any) => value === 'null' ? null : value;
    // return JSON.parse(JSON.stringify(response), replacer);
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
  const resp = await queryBackend<Record<string, any>>("/api/get_metrics/" + email);
  if (!resp) {
    return null;
  }
  const dates = resp['recording_id'];
  const metrics = {...resp};
  delete metrics['recording_id'];
  return { dates, metrics } as Metrics;
  return {
    dates: ["2020-04-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"],
    metrics: {
      "var_coef": [0.1, 0.1, 0.12, 0.1, 0.1],
      "STGA": [0.1, 0.2, 0.3, 0.4, 0.1],
      "phase_sync": [0.1, 0.2, 0.3, 0.4, 0.5],
      "conditional_entropy": [0.1, 0.2, 0.3, 0.4, 0.5],
      "stride_time": [0.1, 0.2, 0.3, 0.4, 0.5],
      "cadence": [0.5, 0.1, 0.2, 0.2, 0.1],
    }
  };
}

export async function getUserStatus(email: string) {} // For insight page, crunches metrics on backend
