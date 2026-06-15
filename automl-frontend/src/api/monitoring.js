import api from "./axios";

/** GET /monitoring/summary → aggregate stats */
export const getMonitoringSummary = async () => {
    const res = await api.get("/monitoring/summary");
    return res.data;
};

/** GET /monitoring/features → per-feature drift list */
export const getFeatureList = async () => {
    const res = await api.get("/monitoring/features");
    return res.data;
};

/** GET /monitoring/feature/{name} → detailed feature stats */
export const getFeatureDetails = async (featureName) => {
    const res = await api.get(`/monitoring/feature/${encodeURIComponent(featureName)}`);
    return res.data;
};
