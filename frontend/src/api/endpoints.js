import { apiFetch } from "./http";

const jsonHeaders = { "Content-Type": "application/json" };

export const api = {
  getStatus: () => apiFetch("/detection/status"),
  getStats: () => apiFetch("/detection/stats"),
  getDetailedStats: () => apiFetch("/detection/stats/detailed"),
  getSettings: () => apiFetch("/settings"),
  getClasses: () => apiFetch("/detection/classes"),
  setClasses: (enabled) =>
    apiFetch("/detection/classes", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ enabled }),
    }),

  startCount: () =>
    apiFetch("/detection/count/start", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({}),
    }),
  pauseCount: () => apiFetch("/detection/count/pause", { method: "POST" }),
  stopCount: () => apiFetch("/detection/count/stop", { method: "POST" }),
  resetCount: () => apiFetch("/detection/count/reset", { method: "POST" }),

  loadModel: (formData) =>
    apiFetch("/video/model/load", { method: "POST", body: formData }),
  listModels: () => apiFetch("/video/model/list"),
  loadLocalModel: (path) =>
    apiFetch("/video/model/load_local", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ path }),
    }),
  modelInfo: () => apiFetch("/video/model/info"),

  detectCameras: () => apiFetch("/video/cameras"),
  setSource: (source) =>
    apiFetch("/video/source/set", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ source }),
    }),
  uploadVideo: (formData) =>
    apiFetch("/video/source/load", { method: "POST", body: formData }),

  setLine: (point1, point2, resetCounts = true) =>
    apiFetch("/settings/line/set", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ point1, point2, reset_counts: resetCounts }),
    }),
  clearLine: () => apiFetch("/settings/line/clear", { method: "POST" }),
  getLine: () => apiFetch("/settings/line"),

  updateSettings: (data) =>
    apiFetch("/settings", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(data),
    }),
  loadPreset: (name) =>
    apiFetch("/settings/preset", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ name }),
    }),
  saveSettings: () => apiFetch("/settings/save", { method: "POST" }),

  saveLog: () => apiFetch("/logs/save", { method: "POST" }),
  listLogs: () => apiFetch("/logs/list"),
  startAnalysis: (videoPath) =>
    apiFetch("/logs/analysis/start", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ video_path: videoPath }),
    }),
  analysisProgress: () => apiFetch("/logs/analysis/progress"),
};
