import { useState, useEffect } from "react";
import {
    ResponsiveContainer,
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
} from "recharts";
import {
    getSystemStatus,
    getModels,
    getMonitoringSignals,
    getAutomationLogs,
} from "../api/dashboard";

/* ──────────────────────────────────────────────────────────
   Helper: format an ISO timestamp to a short label
   ────────────────────────────────────────────────────────── */
function fmtTime(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function fmtDate(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
}
function timeAgo(iso) {
    if (!iso) return "";
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
}

/* ──────────────────────────────────────────────────────────
   Stat Card
   ────────────────────────────────────────────────────────── */
function StatCard({ icon, iconBg, label, value, loading }) {
    return (
        <div className="p-5 rounded-xl border border-slate-200 dark:border-[#2a3b4d] bg-white dark:bg-[#1a2634] shadow-sm">
            <div className="flex justify-between items-start mb-4">
                <div className={`p-2 rounded-lg ${iconBg}`}>
                    <span className="material-symbols-outlined">{icon}</span>
                </div>
            </div>
            <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400">
                {label}
            </h3>
            <p className="text-2xl font-bold text-slate-900 dark:text-white mt-1">
                {loading ? "…" : value}
            </p>
        </div>
    );
}

/* ──────────────────────────────────────────────────────────
   Alert severity styles
   ────────────────────────────────────────────────────────── */
const alertColors = {
    completed: { icon: "check_circle", color: "text-emerald-500" },
    failed: { icon: "error", color: "text-red-500" },
    trained_not_promoted: { icon: "info", color: "text-amber-500" },
    cooldown_blocked: { icon: "schedule", color: "text-slate-400" },
    strategy_skipped: { icon: "skip_next", color: "text-slate-400" },
    early_stopped: { icon: "stop_circle", color: "text-amber-500" },
};

/* ──────────────────────────────────────────────────────────
   Status badge for training jobs
   ────────────────────────────────────────────────────────── */
function StatusBadge({ status }) {
    const map = {
        running: { dot: "bg-blue-500 animate-pulse", text: "text-blue-500", label: "Running" },
        completed: { dot: "bg-emerald-500", text: "text-emerald-500", label: "Completed" },
        failed: { dot: "bg-red-500", text: "text-red-500", label: "Failed" },
        pending: { dot: "bg-amber-400", text: "text-amber-400", label: "Pending" },
    };
    const s = map[status] || map.pending;
    return (
        <span className={`inline-flex items-center gap-1.5 ${s.text} font-medium`}>
            <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
            {s.label}
        </span>
    );
}

/* ══════════════════════════════════════════════════════════
   DASHBOARD PAGE
   ══════════════════════════════════════════════════════════ */
export default function Dashboard() {
    const [loading, setLoading] = useState(true);
    const [systemStatus, setSystemStatus] = useState(null);
    const [models, setModels] = useState([]);
    const [monitoringSignals, setMonitoringSignals] = useState([]);
    const [automationLogs, setAutomationLogs] = useState([]);

    useEffect(() => {
        async function fetchAll() {
            try {
                const [statusRes, modelsRes, monRes, logsRes] = await Promise.allSettled([
                    getSystemStatus(),
                    getModels(),
                    getMonitoringSignals(30),
                    getAutomationLogs(5),
                ]);

                if (statusRes.status === "fulfilled") setSystemStatus(statusRes.value.data);
                if (modelsRes.status === "fulfilled") setModels(modelsRes.value.data || []);
                if (monRes.status === "fulfilled") setMonitoringSignals(monRes.value.data || []);
                if (logsRes.status === "fulfilled") setAutomationLogs(logsRes.value.data || []);
            } catch (err) {
                console.error("Dashboard fetch error:", err);
            } finally {
                setLoading(false);
            }
        }
        fetchAll();
        const interval = setInterval(fetchAll, 30000); // refresh every 30s
        return () => clearInterval(interval);
    }, []);

    /* ── Derived values ── */
    const dbOk = systemStatus?.database === "connected";
    const minioOk = systemStatus?.minio === "connected";
    const ollamaOk = systemStatus?.ollama === "connected";
    const systemOk = dbOk && minioOk;

    const activeModels = typeof systemStatus?.active_models === "number"
        ? systemStatus.active_models
        : models.filter((m) => m.status === "active").length;

    const totalModels = models.length;
    const totalSignals = monitoringSignals.length;

    /* Chart data — drift score over time (reversed so oldest first) */
    const chartData = [...monitoringSignals]
        .reverse()
        .map((s) => ({
            time: fmtTime(s.created_at),
            drift_score: s.drift_score != null ? parseFloat(s.drift_score) : 0,
        }));

    /* ── Critical alerts count from automation logs ── */
    const criticalCount = automationLogs.filter(
        (l) => l.status === "failed" || l.action === "alert"
    ).length;

    return (
        <>
            {/* ── Header ── */}
            <header className="h-16 flex items-center justify-between px-6 border-b border-slate-200 dark:border-[#2a3b4d] bg-white dark:bg-[#111a22] shrink-0 z-10">
                <div className="flex items-center gap-4">
                    <h1 className="text-xl font-semibold text-slate-900 dark:text-white tracking-tight">
                        Dashboard Overview
                    </h1>
                    <div className="h-6 w-px bg-slate-200 dark:bg-[#2a3b4d] mx-2" />
                    <div
                        className={`flex items-center gap-2 px-3 py-1 rounded-full border ${systemOk
                                ? "bg-emerald-500/10 border-emerald-500/20"
                                : "bg-red-500/10 border-red-500/20"
                            }`}
                    >
                        <span className="relative flex h-2 w-2">
                            {systemOk && (
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                            )}
                            <span
                                className={`relative inline-flex rounded-full h-2 w-2 ${systemOk ? "bg-emerald-500" : "bg-red-500"
                                    }`}
                            />
                        </span>
                        <span
                            className={`text-xs font-medium ${systemOk ? "text-emerald-500" : "text-red-500"
                                }`}
                        >
                            {loading
                                ? "Checking…"
                                : systemOk
                                    ? "System Operational"
                                    : "System Degraded"}
                        </span>
                    </div>
                </div>
            </header>

            {/* ── Scrollable content ── */}
            <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
                <div className="max-w-[1600px] mx-auto flex flex-col gap-6">
                    {/* ── Stat cards ── */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <StatCard
                            icon="deployed_code"
                            iconBg="bg-[#137fec]/10 text-[#137fec]"
                            label="Active Models"
                            value={activeModels}
                            loading={loading}
                        />
                        <StatCard
                            icon="hub"
                            iconBg="bg-indigo-500/10 text-indigo-500"
                            label="Total Models"
                            value={totalModels}
                            loading={loading}
                        />
                        <StatCard
                            icon="monitoring"
                            iconBg="bg-orange-500/10 text-orange-500"
                            label="Drift Signals"
                            value={totalSignals}
                            loading={loading}
                        />
                        <StatCard
                            icon="speed"
                            iconBg="bg-pink-500/10 text-pink-500"
                            label="Automation Runs"
                            value={automationLogs.length}
                            loading={loading}
                        />
                    </div>

                    {/* ── Chart + Right column ── */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Drift chart */}
                        <div className="lg:col-span-2 p-6 rounded-xl border border-slate-200 dark:border-[#2a3b4d] bg-white dark:bg-[#1a2634] shadow-sm flex flex-col h-[400px] overflow-hidden">
                            <div className="mb-4">
                                <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                                    Drift Score Trend
                                </h2>
                                <p className="text-sm text-slate-500 dark:text-slate-400">
                                    Recent monitoring signals
                                </p>
                            </div>
                            <div className="flex-1 min-h-0">
                                {chartData.length > 0 ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#2a3b4d" />
                                            <XAxis
                                                dataKey="time"
                                                tick={{ fill: "#94a3b8", fontSize: 11 }}
                                                axisLine={{ stroke: "#2a3b4d" }}
                                                tickLine={false}
                                            />
                                            <YAxis
                                                domain={[0, 1]}
                                                tick={{ fill: "#94a3b8", fontSize: 11 }}
                                                axisLine={{ stroke: "#2a3b4d" }}
                                                tickLine={false}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    background: "#1a2634",
                                                    border: "1px solid #2a3b4d",
                                                    borderRadius: 8,
                                                    color: "#f1f5f9",
                                                    fontSize: 12,
                                                }}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="drift_score"
                                                stroke="#137fec"
                                                strokeWidth={2}
                                                dot={{ r: 3, fill: "#137fec" }}
                                                activeDot={{ r: 5, fill: "#137fec" }}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="flex items-center justify-center h-full text-slate-500 text-sm">
                                        No monitoring data yet
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Right column */}
                        <div className="space-y-6">
                            {/* Cluster health */}
                            <div className="p-6 rounded-xl border border-slate-200 dark:border-[#2a3b4d] bg-white dark:bg-[#1a2634] shadow-sm">
                                <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                                    System Connectivity
                                </h2>
                                <div className="space-y-4">
                                    {[
                                        { label: "Database", ok: dbOk },
                                        { label: "MinIO Storage", ok: minioOk },
                                        { label: "Ollama LLM", ok: ollamaOk },
                                        { label: "Scheduler", ok: systemStatus?.scheduler_running },
                                    ].map((svc) => (
                                        <div
                                            key={svc.label}
                                            className="flex items-center justify-between text-sm"
                                        >
                                            <span className="text-slate-500 dark:text-slate-400">
                                                {svc.label}
                                            </span>
                                            <span
                                                className={`flex items-center gap-1.5 font-medium ${svc.ok ? "text-emerald-500" : "text-red-400"
                                                    }`}
                                            >
                                                <span
                                                    className={`h-2 w-2 rounded-full ${svc.ok ? "bg-emerald-500" : "bg-red-400"
                                                        }`}
                                                />
                                                {loading ? "…" : svc.ok ? "Connected" : "Disconnected"}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Alerts */}
                            <div className="p-6 rounded-xl border border-slate-200 dark:border-[#2a3b4d] bg-white dark:bg-[#1a2634] shadow-sm">
                                <div className="flex items-center justify-between mb-4">
                                    <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                                        Alerts
                                    </h2>
                                    {criticalCount > 0 && (
                                        <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-500/10 text-red-500">
                                            {criticalCount} Critical
                                        </span>
                                    )}
                                </div>
                                <div className="space-y-3">
                                    {automationLogs.length === 0 && !loading && (
                                        <p className="text-sm text-slate-500">No recent alerts</p>
                                    )}
                                    {automationLogs.slice(0, 3).map((log, i) => {
                                        const style =
                                            alertColors[log.status] || alertColors.completed;
                                        return (
                                            <div
                                                key={log.id || i}
                                                className="flex gap-3 items-start p-3 rounded-lg bg-slate-50 dark:bg-[#111a22] border border-slate-100 dark:border-slate-700/50"
                                            >
                                                <span
                                                    className={`material-symbols-outlined ${style.color} text-[20px] mt-0.5`}
                                                >
                                                    {style.icon}
                                                </span>
                                                <div className="min-w-0 flex-1">
                                                    <h4 className="text-sm font-medium text-slate-900 dark:text-slate-200 truncate">
                                                        {log.action?.toUpperCase()} — {log.status}
                                                    </h4>
                                                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 truncate">
                                                        {log.reason || "No reason"}
                                                    </p>
                                                    <p className="text-xs text-slate-400 mt-1">
                                                        {timeAgo(log.created_at)}
                                                    </p>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* ── Models table ── */}
                    <div className="rounded-xl border border-slate-200 dark:border-[#2a3b4d] bg-white dark:bg-[#1a2634] shadow-sm overflow-hidden">
                        <div className="px-6 py-4 border-b border-slate-200 dark:border-[#2a3b4d] flex items-center justify-between">
                            <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                                Registered Models
                            </h2>
                            <span className="text-sm text-slate-400">
                                {models.length} total
                            </span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left border-collapse">
                                <thead>
                                    <tr className="bg-slate-50 dark:bg-[#151e29] text-slate-500 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
                                        <th className="px-6 py-3">Model ID</th>
                                        <th className="px-6 py-3">Framework</th>
                                        <th className="px-6 py-3">Model Class</th>
                                        <th className="px-6 py-3">Status</th>
                                        <th className="px-6 py-3">Version</th>
                                        <th className="px-6 py-3">Created</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-200 dark:divide-slate-700/50 text-sm">
                                    {models.length === 0 && !loading && (
                                        <tr>
                                            <td
                                                colSpan={6}
                                                className="px-6 py-8 text-center text-slate-500"
                                            >
                                                No models registered yet
                                            </td>
                                        </tr>
                                    )}
                                    {models.map((m) => (
                                        <tr
                                            key={m.id}
                                            className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                                        >
                                            <td className="px-6 py-4 font-mono text-slate-600 dark:text-slate-400 text-xs">
                                                {m.id?.slice(0, 8)}…
                                            </td>
                                            <td className="px-6 py-4 font-medium text-slate-900 dark:text-white">
                                                {m.framework || "—"}
                                            </td>
                                            <td className="px-6 py-4 text-slate-500 dark:text-slate-400">
                                                {m.model_class || "—"}
                                            </td>
                                            <td className="px-6 py-4">
                                                <StatusBadge status={m.status} />
                                            </td>
                                            <td className="px-6 py-4 text-slate-500 dark:text-slate-400">
                                                {m.current_version || "—"}
                                            </td>
                                            <td className="px-6 py-4 text-slate-500 dark:text-slate-400">
                                                {fmtDate(m.created_at)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div className="h-10" />
                </div>
            </div>
        </>
    );
}
