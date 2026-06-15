import api from "./axios";

/** GET /inference/status → endpoint status and active version */
export const getInferenceStatus = async () => {
    const res = await api.get("/inference/status");
    return res.data;
};

/** GET /inference/metrics → KPI metrics (latency, rps, etc) */
export const getInferenceMetrics = async () => {
    const res = await api.get("/inference/metrics");
    return res.data;
};

/** GET /inference/logs → live request logs */
export const getInferenceLogs = async (limit = 50) => {
    const res = await api.get("/inference/logs", { params: { limit } });
    return res.data;
};

/** POST /inference/predict → run manual prediction (sandbox) */
export const runInference = async (payload) => {
    const res = await api.post("/inference/predict", payload);
    return res.data;
};

/** POST /inference/switch-version → switch active model version */
export const switchVersion = async (version) => {
    const res = await api.post("/inference/switch-version", { version });
    return res.data;
};
