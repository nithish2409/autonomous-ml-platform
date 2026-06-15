import api from "./axios";

/**
 * Upload a CSV dataset.
 * POST /datasets/upload  (multipart/form-data)
 */
export const uploadDataset = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await api.post("/datasets/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
};

export const deleteDataset = async (datasetId) => {
    const res = await api.delete(`/datasets/${datasetId}`);
    return res.data;
};

/**
 * List all datasets.
 * GET /datasets → array of dataset objects
 */
export const getDatasets = async () => {
    const res = await api.get("/datasets");
    return res.data;
};
