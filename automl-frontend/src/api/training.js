import api from "./axios";

/** GET /training-jobs → list of all training jobs */
export const getTrainingJobs = async () => {
    const res = await api.get("/training-jobs");
    return res.data;
};

/** GET /training/{jobId} → job status and details */
export const getTrainingJobStatus = async (jobId) => {
    const res = await api.get(`/training/${jobId}`);
    return res.data;
};

/** GET /training/{jobId}/logs → container logs */
export const getTrainingJobLogs = async (jobId) => {
    const res = await api.get(`/training/${jobId}/logs`);
    return res.data;
};

/** POST /training/start → start a new training job */
export const startTraining = async (payload) => {
    const res = await api.post("/training/start", payload);
    return res.data;
};

/** DELETE /training/{jobId} → delete a training job */
export const deleteTrainingJob = async (jobId) => {
    const res = await api.delete(`/training/${jobId}`);
    return res.data;
};

/** POST /models/{modelId}/retrain → trigger manual retraining & evaluation */
export const retrainModel = async (modelId) => {
    const res = await api.post(`/models/${modelId}/retrain`);
    return res.data;
};
