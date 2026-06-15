import { useState, useEffect, useRef, useCallback } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";
import {
    getInferenceStatus,
    getInferenceMetrics,
    getInferenceLogs,
    runInference,
    switchVersion,
} from "../api/inference";
import { getModels, getModelVersions } from "../api/models";

/* ── Helpers ──────────────────────────────────────────────── */
function pct(v) {
    if (v == null) return "—";
    const n = typeof v === "number" ? v : parseFloat(v);
    if (isNaN(n)) return "—";
    return n >= 1 ? `${n.toFixed(2)}%` : `${(n * 100).toFixed(2)}%`;
}

function fmtDate(d) {
    if (!d) return "—";
    try {
        const dt = new Date(d);
        return dt.toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            fractionalSecondDigits: 3,
        });
    } catch {
        return d;
    }
}

/* ══════════════════════════════════════════════════════════
   INFERENCE PAGE
   ══════════════════════════════════════════════════════════ */
export default function Inference() {
    // Data state
    const [status, setStatus] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [logs, setLogs] = useState([]);
    const [versions, setVersions] = useState([]);
    const [chartData, setChartData] = useState([]);

    // UI state
    const [loadingStatus, setLoadingStatus] = useState(true);
    const [loadingMetrics, setLoadingMetrics] = useState(true);
    const [loadingLogs, setLoadingLogs] = useState(true);
    const [error, setError] = useState(null);
    const [toast, setToast] = useState(null);

    // Version management
    const [selectedVersion, setSelectedVersion] = useState("");
    const [deploying, setDeploying] = useState(false);

    // Sandbox
    const [sandboxInput, setSandboxInput] = useState(
        JSON.stringify(
            {
                transaction_id: "tx_99218",
                amount: 1240.5,
                currency: "USD",
                merchant_category: "electronics",
                user_id: "user_772",
                ip_address: "192.168.1.1",
                timestamp: "2023-11-01T10:20:00Z",
            },
            null,
            2
        )
    );
    const [sandboxError, setSandboxError] = useState(null);
    const [sandboxRunning, setSandboxRunning] = useState(false);
    const [sandboxResult, setSandboxResult] = useState(null);
    const [simulationProgress, setSimulationProgress] = useState(null);

    // Polling refs
    const metricsInterval = useRef(null);
    const logsInterval = useRef(null);

    /* ── Fetch Data ────────────────────────────────────────── */
    const fetchStatus = useCallback(async () => {
        try {
            const st = await getInferenceStatus();
            setStatus(st);

            // Auto-select version
            if (st.version && !selectedVersion) {
                setSelectedVersion(st.version);
            }

            // Try fetching version list
            try {
                const modelList = await getModels();
                const activeModel = modelList.find((m) => m.status === "active" || m.name === st.current_model);
                if (activeModel) {
                    const vList = await getModelVersions(activeModel.id);
                    setVersions(vList || []);
                    
                    // Auto-detect and set appropriate initial template for active model
                    if (activeModel.model_class && activeModel.model_class.toLowerCase().includes("classifier")) {
                        setSandboxInput(JSON.stringify({
                            age: 35,
                            income: 85000.0,
                            credit_score: 750,
                            loan_amount: 25000.0,
                            employment_years: 8,
                            debt_ratio: 0.25,
                            num_accounts: 4,
                            late_payments: 0,
                            credit_utilization: 0.3
                        }, null, 2));
                    }
                } else {
                    // fallback generic
                    setVersions([{ version: st.version, status: "active", metrics: {} }]);
                }
            } catch {
                // ignore model fetch errors, keep basic version
                setVersions([{ version: st.version, status: "active", metrics: {} }]);
            }
        } catch (err) {
            setError("Failed to load inference status.");
        } finally {
            setLoadingStatus(false);
        }
    }, [selectedVersion]);

    const fetchMetrics = useCallback(async () => {
        try {
            const m = await getInferenceMetrics();
            setMetrics(m);

            // Update chart rolling data
            setChartData((prev) => {
                const now = new Date().toLocaleTimeString();
                const pt = {
                    time: now,
                    latency: m.avg_latency_ms || 0,
                    rps: m.request_rate_rps || 0,
                };
                const next = [...prev, pt];
                return next.length > 20 ? next.slice(next.length - 20) : next;
            });
        } catch {
            // ignore silent failures for polling
        } finally {
            setLoadingMetrics(false);
        }
    }, []);

    const fetchLogs = useCallback(async () => {
        try {
            const l = await getInferenceLogs(50);
            setLogs(l || []);
        } catch {
            // ignore silent failures for polling
        } finally {
            setLoadingLogs(false);
        }
    }, []);

    /* ── Initial Load & Polling ───────────────────────────── */
    useEffect(() => {
        // Initial fetch
        fetchStatus();
        fetchMetrics();
        fetchLogs();

        // Start polling
        metricsInterval.current = setInterval(fetchMetrics, 5000);
        logsInterval.current = setInterval(fetchLogs, 5000);

        return () => {
            clearInterval(metricsInterval.current);
            clearInterval(logsInterval.current);
        };
    }, [fetchStatus, fetchMetrics, fetchLogs]);

    /* ── Handlers ───────────────────────────────────────────── */
    const showToast = (msg, isError = false) => {
        setToast({ msg, isError });
        setTimeout(() => setToast(null), 3000);
    };

    const loadTemplate = (key) => {
        const templates = {
            loan_healthy: {
                age: 35,
                income: 85000.0,
                credit_score: 750,
                loan_amount: 25000.0,
                employment_years: 8,
                debt_ratio: 0.25,
                num_accounts: 4,
                late_payments: 0,
                credit_utilization: 0.3,
                approved: 1
            },
            loan_drift: {
                age: 18,
                income: 1000.0,
                credit_score: 300,
                loan_amount: 99000.0,
                employment_years: 0,
                debt_ratio: 0.99,
                num_accounts: 20,
                late_payments: 15,
                credit_utilization: 0.99,
                approved: 0
            },
            fraud_healthy: {
                transaction_id: "tx_99218",
                amount: 1240.5,
                currency: "USD",
                merchant_category: "electronics",
                user_id: "user_772",
                ip_address: "192.168.1.1",
                timestamp: "2023-11-01T10:20:00Z"
            }
        };

        if (templates[key]) {
            setSandboxInput(JSON.stringify(templates[key], null, 2));
            setSandboxError(null);
            showToast(`Loaded ${key.replace("_", " ")} template`);
        }
    };

    const handleDeploy = async () => {
        if (!selectedVersion || selectedVersion === status?.version) return;
        setDeploying(true);
        try {
            await switchVersion(selectedVersion);
            showToast(`Successfully deployed version ${selectedVersion}`);
            await fetchStatus(); // refresh header
        } catch {
            showToast(`Failed to deploy version ${selectedVersion}`, true);
        } finally {
            setDeploying(false);
        }
    };

    const handleRunInference = async () => {
        setSandboxError(null);
        setSandboxResult(null);

        // Parse JSON
        let payload;
        try {
            payload = JSON.parse(sandboxInput);
        } catch (e) {
            setSandboxError("Invalid JSON payload.");
            return;
        }

        setSandboxRunning(true);
        try {
            const res = await runInference(payload);
            setSandboxResult(res);
            fetchLogs(); // refresh logs immediately
        } catch (err) {
            setSandboxError(err.response?.data?.detail || err.message || "Inference request failed.");
        } finally {
            setSandboxRunning(false);
        }
    };

    const handleSimulateTraffic = async (count = 100) => {
        setSandboxError(null);
        setSandboxResult(null);

        let payload;
        try {
            payload = JSON.parse(sandboxInput);
        } catch (e) {
            setSandboxError("Invalid JSON payload.");
            return;
        }

        setSandboxRunning(true);
        setSimulationProgress(0);
        try {
            const batchSize = 10;
            for (let i = 0; i < count; i += batchSize) {
                const promises = [];
                for (let j = 0; j < batchSize; j++) {
                    promises.push(runInference(payload));
                }
                await Promise.all(promises);
                setSimulationProgress(Math.round(((i + batchSize) / count) * 100));
            }
            showToast(`Successfully simulated ${count} requests!`);
            fetchLogs();
        } catch (err) {
            setSandboxError("Simulation failed midway: " + (err.message || "Unknown error"));
        } finally {
            setSandboxRunning(false);
            setSimulationProgress(null);
        }
    };

    /* ── UI Helpers ─────────────────────────────────────────── */
    const healthBadge = (h) => {
        const s = (h || "offline").toLowerCase();
        if (s === "healthy")
            return "bg-emerald-500/10 text-emerald-500 border-emerald-500/20";
        if (s === "degraded")
            return "bg-amber-500/10 text-amber-500 border-amber-500/20";
        return "bg-rose-500/10 text-rose-500 border-rose-500/20";
    };

    const healthDot = (h) => {
        const s = (h || "offline").toLowerCase();
        if (s === "healthy") return "bg-emerald-500 healthy-pulse";
        if (s === "degraded") return "bg-amber-500";
        return "bg-rose-500";
    };

    return (
        <>
            {/* Toast */}
            {toast && (
                <div
                    className={`fixed top-4 right-4 z-50 px-4 py-2.5 rounded-lg shadow-lg text-sm font-medium border transition-all ${toast.isError
                        ? "bg-rose-500/10 border-rose-500/30 text-rose-400"
                        : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                        }`}
                >
                    {toast.msg}
                </div>
            )}

            <div className="flex-1 overflow-y-auto w-full max-w-[1600px] mx-auto px-4 py-6 space-y-6">

                {/* Error banner */}
                {error && (
                    <div className="bg-rose-500/10 border border-rose-500/30 text-rose-400 px-6 py-3 rounded-xl text-sm flex items-center justify-between shadow-sm">
                        <div className="flex items-center gap-2">
                            <span className="material-symbols-outlined text-[18px]">error</span>
                            {error}
                        </div>
                        <button
                            onClick={() => { setError(null); fetchStatus(); fetchMetrics(); fetchLogs(); }}
                            className="text-xs font-bold hover:text-rose-300 underline"
                        >
                            Retry
                        </button>
                    </div>
                )}

                {/* ── Header Section ── */}
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-slate-200 dark:border-surface-border pb-6">
                    <div className="space-y-1">
                        <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
                            <span>Endpoints</span>
                            <span className="material-symbols-outlined text-xs">chevron_right</span>
                            <span className="text-slate-900 dark:text-slate-100">
                                {loadingStatus ? "..." : status?.endpoint_name || "inference-endpoint"}
                            </span>
                        </div>
                        <div className="flex flex-wrap items-center gap-4">
                            <h1 className="text-3xl font-bold tracking-tight">
                                {loadingStatus ? "Loading..." : status?.endpoint_name || "inference-endpoint"}
                            </h1>
                            {!loadingStatus && (
                                <div
                                    className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold border ${healthBadge(
                                        status?.health_status
                                    )}`}
                                >
                                    <span className={`size-2 rounded-full ${healthDot(status?.health_status)}`} />
                                    <span className="capitalize">{status?.health_status || "Offline"}</span>
                                </div>
                            )}
                        </div>

                        {!loadingStatus && (
                            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 pt-2 text-sm font-medium text-slate-500 dark:text-slate-400">
                                <div className="flex items-center gap-2">
                                    <span className="material-symbols-outlined text-lg">history</span>
                                    Version: <span className="text-slate-900 dark:text-slate-200 font-mono">{status?.version || "—"}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="material-symbols-outlined text-lg">layers</span>
                                    Replicas: <span className="text-slate-900 dark:text-slate-200">{status?.replicas || 0} pods</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="material-symbols-outlined text-lg">public</span>
                                    Region: <span className="text-slate-900 dark:text-slate-200">{status?.region || "—"}</span>
                                </div>
                                {status?.uptime && (
                                    <div className="flex items-center gap-2">
                                        <span className="material-symbols-outlined text-lg">timer</span>
                                        Uptime: <span className="text-slate-900 dark:text-slate-200">{status.uptime}</span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="flex gap-2">
                        <button className="bg-slate-200 dark:bg-surface-dark border border-slate-300 dark:border-surface-border hover:bg-slate-300 dark:hover:bg-slate-800 text-slate-900 dark:text-white px-4 py-2 rounded-lg font-bold text-sm transition-all flex items-center gap-2">
                            <span className="material-symbols-outlined text-xl">edit</span>
                            Edit Config
                        </button>
                        <button className="bg-primary hover:bg-primary-dark text-white px-4 py-2 rounded-lg font-bold text-sm transition-all flex items-center gap-2 shadow-lg shadow-primary/20">
                            <span className="material-symbols-outlined text-xl">rocket_launch</span>
                            Rollback
                        </button>
                    </div>
                </div>

                {/* ── KPI Grid ── */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <KpiCard
                        title="Avg Latency"
                        value={metrics?.avg_latency_ms != null ? `${metrics.avg_latency_ms.toFixed(1)}ms` : "—"}
                        subtitle="live"
                        loading={loadingMetrics}
                    />
                    <KpiCard
                        title="P95 Latency"
                        value={metrics?.p95_latency_ms != null ? `${metrics.p95_latency_ms.toFixed(1)}ms` : "—"}
                        subtitle="live"
                        loading={loadingMetrics}
                    />
                    <KpiCard
                        title="Request Rate"
                        value={metrics?.request_rate_rps != null ? `${metrics.request_rate_rps.toFixed(1)} RPS` : "—"}
                        subtitle="live"
                        loading={loadingMetrics}
                    />
                    <KpiCard
                        title="Error Rate"
                        value={metrics?.error_rate_percent != null ? `${metrics.error_rate_percent.toFixed(2)}%` : "—"}
                        subtitle="live"
                        loading={loadingMetrics}
                    />
                </div>

                {/* ── Performance Chart ── */}
                <div className="bg-white dark:bg-surface-dark border border-slate-200 dark:border-surface-border rounded-xl shadow-sm">
                    <div className="p-6 border-b border-slate-200 dark:border-surface-border flex justify-between items-center">
                        <h2 className="font-bold text-lg">Live Inference Performance</h2>
                    </div>
                    <div className="p-6 h-[300px] w-full">
                        {chartData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                                    <XAxis dataKey="time" stroke="#94a3b8" fontSize={10} tickMargin={10} />
                                    <YAxis yAxisId="left" stroke="#94a3b8" fontSize={10} domain={['auto', 'auto']} />
                                    <YAxis yAxisId="right" orientation="right" stroke="#137fec" fontSize={10} domain={['auto', 'auto']} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155", borderRadius: "8px", fontSize: "12px", color: "#fff" }}
                                        itemStyle={{ color: "#e2e8f0" }}
                                    />
                                    <Line yAxisId="left" type="monotone" dataKey="latency" name="Latency (ms)" stroke="#94a3b8" strokeWidth={2} dot={false} isAnimationActive={false} />
                                    <Line yAxisId="right" type="monotone" dataKey="rps" name="RPS" stroke="#137fec" strokeWidth={2} dot={false} isAnimationActive={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full w-full flex items-center justify-center text-slate-400 text-sm">Waiting for metric points...</div>
                        )}
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                    {/* ── Version Management ── */}
                    <div className="lg:col-span-1 space-y-6 flex flex-col h-full">
                        <div className="bg-white dark:bg-surface-dark border border-slate-200 dark:border-surface-border rounded-xl shadow-sm p-6 flex-1">
                            <h2 className="font-bold text-lg mb-4 flex items-center gap-2">
                                <span className="material-symbols-outlined text-primary">schema</span>
                                Version Management
                            </h2>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-2 uppercase tracking-wider">
                                        Select Version to Deploy
                                    </label>
                                    <div className="relative">
                                        <select
                                            value={selectedVersion}
                                            onChange={(e) => setSelectedVersion(e.target.value)}
                                            className="w-full bg-slate-100 dark:bg-[#111a22] border border-slate-200 dark:border-surface-border rounded-lg text-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-primary appearance-none transition-colors"
                                        >
                                            {versions.length === 0 && <option value="">Loading...</option>}
                                            {versions.map((v) => (
                                                <option key={v.version} value={v.version}>
                                                    {v.version} {v.version === status?.version ? "(Active)" : ""}
                                                </option>
                                            ))}
                                        </select>
                                        <span className="material-symbols-outlined absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500">
                                            expand_more
                                        </span>
                                    </div>
                                </div>

                                {selectedVersion && (
                                    <div className="p-4 bg-slate-50 dark:bg-[#111a22] rounded-lg border border-slate-100 dark:border-surface-border">
                                        <h4 className="text-xs font-bold mb-3 text-slate-500 uppercase">Version Details ({selectedVersion})</h4>
                                        <ul className="text-xs space-y-2 text-slate-600 dark:text-slate-400">
                                            <li className="flex justify-between">
                                                <span>Model Artifact:</span>
                                                <span className="font-mono text-primary truncate max-w-[150px]">
                                                    s3://models/{selectedVersion}
                                                </span>
                                            </li>
                                            <li className="flex justify-between">
                                                <span>Framework:</span>
                                                <span>{status?.framework || "Unknown"}</span>
                                            </li>
                                        </ul>
                                    </div>
                                )}

                                <button
                                    onClick={handleDeploy}
                                    disabled={!selectedVersion || selectedVersion === status?.version || deploying}
                                    className="w-full bg-primary hover:bg-primary-dark text-white font-bold py-3 rounded-lg transition-all shadow-lg shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                                >
                                    {deploying ? (
                                        <span className="material-symbols-outlined animate-spin">sync</span>
                                    ) : (
                                        "Deploy Version"
                                    )}
                                </button>
                            </div>
                        </div>

                        <div className="bg-white dark:bg-surface-dark border border-slate-200 dark:border-surface-border rounded-xl shadow-sm p-6 flex-none">
                            <h2 className="font-bold text-lg mb-4">Infrastructure Status</h2>
                            <div className="space-y-4">
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-slate-500 dark:text-slate-400">CPU Usage</span>
                                    <span className="font-mono">{status?.cpu_usage || "42%"}</span>
                                </div>
                                <div className="w-full bg-slate-100 dark:bg-[#111a22] h-1.5 rounded-full overflow-hidden">
                                    <div className="bg-primary h-full w-[42%]" style={{ width: status?.cpu_usage || '42%' }}></div>
                                </div>
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-slate-500 dark:text-slate-400">Memory Usage</span>
                                    <span className="font-mono">{status?.memory_usage || "2.4 GB"}</span>
                                </div>
                                <div className="w-full bg-slate-100 dark:bg-[#111a22] h-1.5 rounded-full overflow-hidden">
                                    <div className="bg-purple-500 h-full w-[30%]" style={{ width: '30%' }}></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* ── Manual Prediction Sandbox ── */}
                    <div className="lg:col-span-2 bg-white dark:bg-surface-dark border border-slate-200 dark:border-surface-border rounded-xl shadow-sm overflow-hidden flex flex-col h-full">
                        <div className="p-6 border-b border-slate-200 dark:border-surface-border flex justify-between items-center bg-slate-50 dark:bg-surface-dark">
                            <h2 className="font-bold text-lg flex items-center gap-2">
                                <span className="material-symbols-outlined text-primary">terminal</span>
                                Manual Prediction Test
                            </h2>
                        </div>
                        <div className="flex-1 grid grid-cols-1 md:grid-cols-2">

                            {/* Input */}
                            <div className="p-6 border-r border-slate-200 dark:border-surface-border flex flex-col">
                                <div className="flex flex-col gap-2 mb-3">
                                    <div className="flex items-center justify-between">
                                        <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                                            Request Payload (JSON)
                                        </span>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5 pt-1">
                                        <button
                                            type="button"
                                            onClick={() => loadTemplate("loan_healthy")}
                                            className="px-2 py-1 text-[11px] font-semibold rounded bg-emerald-500/10 border border-emerald-500/20 hover:bg-emerald-500/20 text-emerald-400 transition-colors"
                                        >
                                            Loan (Healthy)
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => loadTemplate("loan_drift")}
                                            className="px-2 py-1 text-[11px] font-semibold rounded bg-rose-500/10 border border-rose-500/20 hover:bg-rose-500/20 text-rose-400 transition-colors"
                                        >
                                            Loan (Drift)
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => loadTemplate("fraud_healthy")}
                                            className="px-2 py-1 text-[11px] font-semibold rounded bg-slate-100 hover:bg-slate-200 dark:bg-surface-dark dark:hover:bg-slate-800 dark:border dark:border-surface-border text-slate-700 dark:text-slate-300 transition-colors"
                                        >
                                            Fraud (Healthy)
                                        </button>
                                    </div>
                                </div>
                                <textarea
                                    value={sandboxInput}
                                    onChange={(e) => setSandboxInput(e.target.value)}
                                    spellCheck="false"
                                    className="flex-1 w-full bg-white dark:bg-[#111a22] border border-slate-200 dark:border-surface-border rounded-lg p-4 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 resize-none min-h-[250px]"
                                />

                                {sandboxError && (
                                    <div className="mt-3 p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-lg font-mono">
                                        {sandboxError}
                                    </div>
                                )}

                                <div className="flex gap-2 mt-4">
                                    <button
                                        onClick={handleRunInference}
                                        disabled={sandboxRunning}
                                        className="flex-1 bg-slate-900 dark:bg-white text-white dark:text-slate-900 font-bold py-3 rounded-lg flex items-center justify-center gap-2 hover:opacity-90 transition-opacity disabled:opacity-50"
                                    >
                                        {sandboxRunning && simulationProgress === null ? (
                                            <span className="material-symbols-outlined animate-spin text-[18px]">sync</span>
                                        ) : (
                                            <span className="material-symbols-outlined text-[18px]">play_arrow</span>
                                        )}
                                        Run Inference
                                    </button>
                                    <button
                                        onClick={() => handleSimulateTraffic(100)}
                                        disabled={sandboxRunning}
                                        className="flex-1 bg-amber-500 hover:bg-amber-600 text-white font-bold py-3 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 shadow-lg shadow-amber-500/10"
                                    >
                                        <span className={`material-symbols-outlined text-[18px] ${simulationProgress !== null ? "animate-spin" : ""}`}>
                                            {simulationProgress !== null ? "sync" : "speed"}
                                        </span>
                                        {simulationProgress !== null ? `Simulating ${simulationProgress}%` : "Simulate 100x"}
                                    </button>
                                </div>
                            </div>

                            {/* Output */}
                            <div className="p-6 bg-slate-50 dark:bg-[#111a22] flex flex-col">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                                        Inference Result
                                    </span>
                                    {sandboxResult && (
                                        <div className="flex gap-2">
                                            <span className="bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 text-[10px] font-bold px-2 py-0.5 rounded">
                                                200 OK
                                            </span>
                                            <span className="bg-slate-200 dark:bg-surface-dark border dark:border-surface-border text-[10px] font-bold px-2 py-0.5 rounded text-slate-600 dark:text-slate-300">
                                                {sandboxResult.latency_ms || "—"}ms
                                            </span>
                                        </div>
                                    )}
                                </div>

                                <div className="flex-1 bg-white dark:bg-[#0d131a] border border-slate-200 dark:border-slate-800 rounded-lg p-4 font-mono text-xs overflow-auto">
                                    {sandboxResult ? (
                                        <pre className="text-slate-600 dark:text-slate-300"
                                            dangerouslySetInnerHTML={{
                                                __html: JSON.stringify(sandboxResult, null, 2).replace(
                                                    /"prediction": |"status": |"latency_ms": /g,
                                                    (match) => `<span class="text-primary">${match}</span>`
                                                ).replace(
                                                    /"success"|true|false/g,
                                                    (match) => `<span class="text-emerald-400">${match}</span>`
                                                )
                                            }}
                                        />
                                    ) : (
                                        <div className="h-full flex items-center justify-center text-slate-400 text-sm italic">
                                            Run inference to view results here...
                                        </div>
                                    )}
                                </div>
                                <p className="mt-4 text-[11px] text-slate-500 leading-tight">
                                    Manual inference calls are marked with the tag{" "}
                                    <span className="font-mono text-primary">[sandbox]</span>.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── Live Request Logs ── */}
                <div className="bg-white dark:bg-surface-dark border border-slate-200 dark:border-surface-border rounded-xl shadow-sm overflow-hidden min-h-[300px] flex flex-col">
                    <div className="p-6 border-b border-slate-200 dark:border-surface-border flex justify-between items-center bg-slate-50 dark:bg-surface-dark">
                        <h2 className="font-bold text-lg flex items-center gap-2">
                            <span className="material-symbols-outlined text-primary">list_alt</span>
                            Live Request Logs
                        </h2>
                        <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
                                <span className="size-2 bg-emerald-500 rounded-full healthy-pulse"></span>
                                Streaming live...
                            </div>
                        </div>
                    </div>

                    <div className="overflow-x-auto flex-1">
                        <table className="w-full text-left text-sm border-collapse">
                            <thead className="bg-slate-50 dark:bg-[#151f28] sticky top-0">
                                <tr className="text-slate-500 dark:text-slate-400 font-bold uppercase text-[10px] tracking-widest border-b border-slate-200 dark:border-surface-border">
                                    <th className="px-6 py-4">Timestamp</th>
                                    <th className="px-6 py-4">Request ID</th>
                                    <th className="px-6 py-4">Latency</th>
                                    <th className="px-6 py-4">Status</th>
                                    <th className="px-6 py-4">Result</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100 dark:divide-surface-border font-mono text-xs">
                                {loadingLogs ? (
                                    <tr>
                                        <td colSpan="5" className="p-8 text-center text-slate-400 h-32">
                                            <span className="material-symbols-outlined animate-spin text-2xl">sync</span>
                                        </td>
                                    </tr>
                                ) : logs.length === 0 ? (
                                    <tr>
                                        <td colSpan="5" className="p-8 text-center text-slate-400 h-32 italic">
                                            No inference logs found.
                                        </td>
                                    </tr>
                                ) : (
                                    logs.map((log) => (
                                        <tr key={log.request_id} className="hover:bg-slate-50 dark:hover:bg-white/5 transition-colors">
                                            <td className="px-6 py-4 whitespace-nowrap text-slate-500">
                                                {fmtDate(log.timestamp)}
                                            </td>
                                            <td className="px-6 py-4 font-bold text-slate-900 dark:text-slate-300">
                                                {log.request_id?.slice(0, 12)}...
                                            </td>
                                            <td className="px-6 py-4 text-slate-600 dark:text-slate-400">
                                                {log.latency_ms?.toFixed(1) || 0}ms
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`font-bold ${(log.status_code >= 200 && log.status_code < 300) ? "text-emerald-500" : "text-rose-500"
                                                    }`}>
                                                    {log.status_code || 200}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4">
                                                {log.is_sandbox && (
                                                    <span className="bg-primary/10 text-primary px-1.5 py-0.5 rounded text-[10px] mr-2 font-display uppercase font-bold tracking-wider">Sandbox</span>
                                                )}
                                                <span className="text-slate-700 dark:text-slate-400">{JSON.stringify(log.prediction || {})}</span>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>
        </>
    );
}

/* ── KPI Card Component ── */
function KpiCard({ title, value, subtitle, loading }) {
    return (
        <div className="bg-white dark:bg-surface-dark border border-slate-200 dark:border-surface-border p-5 rounded-xl shadow-sm hover:border-primary/50 transition-colors group">
            <div className="flex justify-between items-start mb-2">
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{title}</p>
            </div>
            {loading ? (
                <div className="h-8 w-24 bg-slate-200 dark:bg-slate-800 rounded animate-pulse my-1" />
            ) : (
                <div className="flex items-baseline gap-2">
                    <h3 className="text-2xl font-bold tracking-tight">{value}</h3>
                    {(subtitle && value !== "—") && (
                        <span className="text-slate-400 text-xs font-mono bg-slate-100 dark:bg-[#111a22] px-1.5 py-0.5 rounded">
                            {subtitle}
                        </span>
                    )}
                </div>
            )}
        </div>
    );
}
