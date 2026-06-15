import api from "./axios";

/** GET /automation/status → aggregate summary (cards) */
export const getAutomationStatus = async () => {
    const res = await api.get("/automation/status");
    return res.data;
};

/** GET /automation/history → decision list */
export const getDecisionHistory = async () => {
    const res = await api.get("/automation/history");
    return res.data;
};

/** GET /automation/decision/{id} → single decision detail */
export const getDecisionDetails = async (id) => {
    const res = await api.get(`/automation/decision/${id}`);
    return res.data;
};

/** POST /automation/toggle → enable/disable autonomous mode */
export const toggleAutonomousMode = async (enabled) => {
    const res = await api.post("/automation/toggle", { enabled });
    return res.data;
};

/** POST /automation/approve/{id} → approve a pending decision */
export const approveDecision = async (id) => {
    const res = await api.post(`/automation/approve/${id}`);
    return res.data;
};

/** POST /automation/reject/{id} → reject a pending decision */
export const rejectDecision = async (id) => {
    const res = await api.post(`/automation/reject/${id}`);
    return res.data;
};

/** POST /automation/manual-train/{id} → trigger manual training */
export const triggerManualTrain = async (id) => {
    const res = await api.post(`/automation/manual-train/${id}`);
    return res.data;
};

/** POST /system/trigger-monitoring → manually trigger the monitoring cycle */
export const triggerMonitoringLoop = async () => {
    const res = await api.post("/system/trigger-monitoring");
    return res.data;
};
