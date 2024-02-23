export async function queryBackend(endpoint: String): Promise<any> {
    const url = import.meta.env.VITE_BACKEND_URL;
    console.log("url: ", url);
    if (!url) {
        throw new Error("VITE_BACKEND_URL not defined. Please define it in .env file.");
    }
  const response = await fetch(url + endpoint);
    return response.json();
}
