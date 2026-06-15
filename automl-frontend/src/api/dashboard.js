import api from "./axios";

/** GET /system/status → { database, minio, ollama, scheduler_running, active_models } */
export const getSystemStatus = () => api.get("/system/status");

/** GET /models → array of model objects */
export const getModels = () => api.get("/models");

/** GET /monitoring/signals?limit=50 → array of monitoring metrics */
export const getMonitoringSignals = (limit = 50) =>
    api.get("/monitoring/signals", { params: { limit } });

/** GET /automation/logs?limit=10 → array of automation log entries */
export const getAutomationLogs = (limit = 10) =>
    api.get("/automation/logs", { params: { limit } });
