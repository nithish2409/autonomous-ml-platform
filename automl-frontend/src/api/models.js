import api from "./axios";

/** GET /models → array of model objects */
export const getModels = async () => {
    const res = await api.get("/models");
    return res.data;
};

/** POST /models/register → registers a new model */
export const registerModel = async (payload) => {
    const res = await api.post("/models/register", payload);
    return res.data;
};

/** GET /models/{modelId}/versions → array of VersionResponse */
export const getModelVersions = async (modelId) => {
    const res = await api.get(`/models/${modelId}/versions`);
    return res.data;
};

/** POST /models/{modelId}/activate → activate a model */
export const activateModel = async (modelId) => {
    const res = await api.post(`/models/${modelId}/activate`);
    return res.data;
};

export const deleteModel = async (modelId) => {
    const res = await api.delete(`/models/${modelId}`);
    return res.data;
};

/** POST /models/{modelId}/promote → promote a specific version */
export const promoteVersion = async (modelId, version) => {
    const res = await api.post(`/models/${modelId}/promote`, { version });
    return res.data;
};
