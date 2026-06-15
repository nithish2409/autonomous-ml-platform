import api from "./axios";

/** GET /policies → current policy config */
export const getPolicies = async () => {
    const res = await api.get("/policies");
    return res.data;
};

/** PUT /policies → update policy config */
export const updatePolicies = async (data) => {
    const res = await api.put("/policies", data);
    return res.data;
};

/** POST /policies/simulate/{id} → simulate policy against a decision */
export const simulatePolicy = async (decisionId) => {
    const res = await api.post(`/policies/simulate/${decisionId}`);
    return res.data;
};
