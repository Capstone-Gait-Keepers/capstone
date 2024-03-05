import type { User } from "./types";


export async function queryBackend<T>(endpoint: String): Promise<T | null> {
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
    const response = await fetch(url + endpoint);
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
  return true;
}

// Returns full User object
export async function login(email: string, password: string) {
  queryBackend<User>("/login");
}

export async function getMetrics(email: string, startDate: string | null, endDate: string | null = null) {}

export async function getUserStatus(email: string) {} // For insight page, crunches metrics on backend
