const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000/api").replace(/\/$/, "");

export function buildApiUrl(path) {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL_V2}${normalized}`;
}

export async function apiFetch(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), options.timeoutMs || 20000);

  try {
    const response = await fetch(buildApiUrl(path), {
      ...options,
      signal: controller.signal,
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      const message = payload?.message || payload?.detail || `HTTP ${response.status}`;
      throw new Error(message);
    }

    return payload;
  } finally {
    clearTimeout(timeout);
  }
}

const API_BASE_URL_V2 = API_BASE_URL;

export { API_BASE_URL_V2 };

