// Axios client for the ParkWatch AI FastAPI backend.
//
// - Base URL points at the FastAPI server (http://localhost:8000/api).
// - A request interceptor injects the bearer token from localStorage.
// - A response interceptor surfaces a clean error message and auto-logs-out on 401.
// - useFetch / useLazyRequest hooks standardise loading + error handling in tabs.

import axios from "axios";
import { useCallback, useEffect, useRef, useState } from "react";

export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";
const TOKEN_KEY = "parkwatch_token";

export const tokenStore = {
  get: () => localStorage.getItem(TOKEN_KEY),
  set: (t) => localStorage.setItem(TOKEN_KEY, t),
  clear: () => localStorage.removeItem(TOKEN_KEY),
};

const api = axios.create({ baseURL: API_BASE, timeout: 20000 });

// Inject Authorization header when a token is present.
api.interceptors.request.use((config) => {
  const token = tokenStore.get();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Normalise errors; broadcast a 401 so AuthContext can log the user out.
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      window.dispatchEvent(new CustomEvent("parkwatch:unauthorized"));
    }
    const detail = err.response?.data?.detail;
    const msg =
      detail ||
      (err.code === "ERR_NETWORK"
        ? "Cannot reach the API at " + API_BASE + " — is the FastAPI backend running?"
        : err.message);
    return Promise.reject(new Error(msg));
  }
);

export default api;

// ── Endpoint helpers ─────────────────────────────────────────────────────────
export const authApi = {
  login: (username, password) => api.post("/auth/login", { username, password }).then((r) => r.data),
  me: () => api.get("/auth/me").then((r) => r.data),
};

export const endpoints = {
  telemetrySummary: () => api.get("/telemetry/summary").then((r) => r.data),
  mapViolations: () => api.get("/map/violations").then((r) => r.data),
  mapPlayback: () => api.get("/map/playback").then((r) => r.data),
  analyticsCharts: () => api.get("/analytics/charts").then((r) => r.data),
  simulatorEvaluate: (params) => api.post("/simulator/evaluate", params).then((r) => r.data),
  simulatorLeaderboard: () => api.get("/simulator/leaderboard").then((r) => r.data),
  commanderChat: (message, history, session_id) =>
    api.post("/commander/chat", { message, history, session_id }).then((r) => r.data),
  commanderInsights: () => api.get("/commander/insights").then((r) => r.data),
  forecastPredict: (days = 7, station) =>
    api.get("/forecast/predict", { params: { days, station } }).then((r) => r.data),
  forecastDay: (day) => api.get("/forecast/predict", { params: { day } }).then((r) => r.data),
  alerts: () => api.get("/alerts").then((r) => r.data),
  // ── Advanced ML models ──
  mlStatus: () => api.get("/ml/status").then((r) => r.data),
  predictEta: (body) => api.post("/predict/eta", body).then((r) => r.data),
  predictPropensity: (station, hour = 18) => api.get("/predict/propensity", { params: { station, hour } }).then((r) => r.data),
  emergingHotspots: () => api.get("/predict/emerging").then((r) => r.data),
  economicForecast: (days = 30) => api.get("/forecast/economic", { params: { days } }).then((r) => r.data),
  dataPreview: () => api.get("/data/preview").then((r) => r.data),
  dataUpload: (file, columnMap) => {
    const fd = new FormData();
    fd.append("file", file);
    if (columnMap) fd.append("column_map", JSON.stringify(columnMap));
    return api
      .post("/data/upload", fd, { headers: { "Content-Type": "multipart/form-data" }, timeout: 120000 })
      .then((r) => r.data);
  },
  uploadStatus: (jobId) => api.get(`/data/upload/status/${jobId}`, { timeout: 120000 }).then((r) => r.data),
  emergencyOptions: () => api.get("/emergency/options").then((r) => r.data),
  emergencyResponse: (station, hospital) =>
    api.get("/emergency/response", { params: { station, hospital } }).then((r) => r.data),
  dataClean: (toggles) => api.post("/data/clean", toggles).then((r) => r.data),
  cctvCameras: () => api.get("/cctv/cameras").then((r) => r.data),
  cctvInfraction: (payload) => api.post("/cctv/infraction", payload).then((r) => r.data),
  dispatcherVrp: (fleet_size) => api.post("/dispatcher/vrp", { fleet_size }).then((r) => r.data),
  ortoolsSolve: (trucks, stations) =>
    api.get("/ortools/solve", { params: { trucks, stations } }).then((r) => r.data),
};

// ── Hooks ────────────────────────────────────────────────────────────────────

/**
 * useFetch — run an async loader on mount (and when deps change).
 * Returns { data, loading, error, refetch }.
 *   const { data, loading, error, refetch } = useFetch(endpoints.analyticsCharts, []);
 */
export function useFetch(loader, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const mounted = useRef(true);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await loader();
      if (mounted.current) setData(res);
    } catch (e) {
      if (mounted.current) setError(e.message || "Request failed");
    } finally {
      if (mounted.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    mounted.current = true;
    run();
    return () => {
      mounted.current = false;
    };
  }, [run]);

  // Refetch automatically when the active dataset changes (e.g. after a CSV upload),
  // so every mounted tab updates live without needing a tab switch.
  useEffect(() => {
    const onChanged = () => run();
    window.addEventListener("parkwatch:dataset-changed", onChanged);
    return () => window.removeEventListener("parkwatch:dataset-changed", onChanged);
  }, [run]);

  return { data, loading, error, refetch: run };
}

/**
 * useLazyRequest — for POST actions triggered by the user (sliders, buttons).
 * Returns { run, data, loading, error }.
 */
export function useLazyRequest(action) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = useCallback(
    async (...args) => {
      setLoading(true);
      setError(null);
      try {
        const res = await action(...args);
        setData(res);
        return res;
      } catch (e) {
        setError(e.message || "Request failed");
        throw e;
      } finally {
        setLoading(false);
      }
    },
    [action]
  );

  return { run, data, loading, error, setData };
}
