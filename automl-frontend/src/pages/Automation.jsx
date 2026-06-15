import { useState, useEffect, useRef, useCallback } from "react";
import {
    getAutomationStatus,
    getDecisionHistory,
    getDecisionDetails,
    toggleAutonomousMode,
    approveDecision,
    rejectDecision,
    triggerManualTrain,
} from "../api/automation";

/* ── helpers ───────────────────────────────────────────────── */
function pct(v) {
    if (v == null) return "—";
    const n = typeof v === "number" ? v : parseFloat(v);
    if (isNaN(n)) return "—";
    return n >= 1 ? `${n.toFixed(0)}%` : `${(n * 100).toFixed(1)}%`;
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
        });
    } catch {
        return d;
    }
}
function shortId(id) {
    if (!id) return "—";
    return `#DEC-${id.slice(0, 4).toUpperCase()}`;
}

const SEV = {
    low: "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-400 border-slate-200 dark:border-slate-700",
    medium: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400 border-orange-200 dark:border-orange-900/50",
    high: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-900/50",
    critical: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-900/50 font-bold",
};
const STAT = {
    pending: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-900/50",
    pending_human_review: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-900/50 font-medium",
    scheduled: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-900/50",
    alert: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-900/50",
    executed: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/50",
    completed: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/50",
    success: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/50",
    rejected: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-500 border-slate-200 dark:border-slate-700",
    manual_training: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400 border-purple-200 dark:border-purple-900/50",
};
function isPending(s) {
    return ["pending", "alert", "scheduled", "pending_human_review"].includes(s);
}

/* ══════════════════════════════════════════════════════════
   AUTOMATION CONTROL CENTER
   ══════════════════════════════════════════════════════════ */
export default function Automation() {
    const [status, setStatus] = useState(null);
    const [history, setHistory] = useState([]);
    const [selected, setSelected] = useState(null);
    const [detail, setDetail] = useState(null);
    const [loading, setLoading] = useState(true);
    const [loadingDetail, setLoadingDetail] = useState(false);
    const [actionLoading, setActionLoading] = useState(null);
    const [autoEnabled, setAutoEnabled] = useState(true);
    const [toast, setToast] = useState(null);
    const [error, setError] = useState(null);
    const [countdown, setCountdown] = useState(null);
    const pollRef = useRef(null);
    const countdownRef = useRef(null);

    /* ── data loading ── */
    const loadData = useCallback(async () => {
        try {
            const [st, hist] = await Promise.all([
                getAutomationStatus(),
                getDecisionHistory(),
            ]);
            setStatus(st);
            setHistory(hist);
            setAutoEnabled(st.autonomous_enabled);
            setError(null);
        } catch (err) {
            setError("Failed to load automation data");
        } finally {
            setLoading(false);
        }
    }, []);

    /* ── on mount + polling ── */
    useEffect(() => {
        loadData();
        pollRef.current = setInterval(loadData, 15000);
        return () => {
            clearInterval(pollRef.current);
            clearInterval(countdownRef.current);
        };
    }, [loadData]);

    /* ── select decision ── */
    const selectDecision = useCallback(async (row) => {
        setSelected(row);
        setLoadingDetail(true);
        setCountdown(null);
        clearInterval(countdownRef.current);
        try {
            const d = await getDecisionDetails(row.id);
            setDetail(d);
            // Start countdown if pending
            if (d.auto_approval_seconds && isPending(d.status)) {
                let secs = d.auto_approval_seconds;
                setCountdown(secs);
                countdownRef.current = setInterval(() => {
                    secs -= 1;
                    if (secs <= 0) {
                        clearInterval(countdownRef.current);
                        setCountdown(null);
                        loadData();
                    } else {
                        setCountdown(secs);
                    }
                }, 1000);
            }
        } catch {
            setDetail(null);
        } finally {
            setLoadingDetail(false);
        }
    }, [loadData]);

    /* Auto-select first row when history loads and none selected */
    useEffect(() => {
        if (history.length > 0 && !selected) {
            selectDecision(history[0]);
        }
    }, [history, selected, selectDecision]);

    /* ── toggle autonomous mode ── */
    const handleToggle = useCallback(async () => {
        const next = !autoEnabled;
        setAutoEnabled(next); // optimistic
        try {
            await toggleAutonomousMode(next);
            showToast(`Autonomous mode ${next ? "enabled" : "disabled"}`);
        } catch {
            setAutoEnabled(!next); // rollback
            showToast("Failed to toggle mode", true);
        }
    }, [autoEnabled]);


    /* ── actions ── */
    const handleAction = useCallback(
        async (type) => {
            if (!detail) return;
            setActionLoading(type);
            try {
                if (type === "approve") await approveDecision(detail.id);
                else if (type === "reject") await rejectDecision(detail.id);
                else if (type === "manual") await triggerManualTrain(detail.id);
                clearInterval(countdownRef.current);
                setCountdown(null);
                showToast(
                    type === "approve"
                        ? "Decision approved"
                        : type === "reject"
                            ? "Decision rejected"
                            : "Manual training triggered"
                );
                await loadData();
                // Refresh detail
                const d = await getDecisionDetails(detail.id);
                setDetail(d);
            } catch {
                showToast(`Failed to ${type} decision`, true);
            } finally {
                setActionLoading(null);
            }
        },
        [detail, loadData]
    );

    const showToast = (msg, isError = false) => {
        setToast({ msg, isError });
        setTimeout(() => setToast(null), 3000);
    };

    /* ── countdown format ── */
    const fmtCountdown = (s) => {
        const m = Math.floor(s / 60);
        const sec = s % 60;
        return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
    };

    /* ── confidence bar color ── */
    const confBar = (c) => {
        const n = typeof c === "number" ? c : parseFloat(c) || 0;
        const v = n >= 1 ? n : n * 100;
        return { width: `${v}%`, bg: v >= 80 ? "bg-primary" : v >= 60 ? "bg-warning" : "bg-danger" };
    };

    return (
        <>
            {/* ── Toast ── */}
            {toast && (
                <div className={`fixed top-4 right-4 z-50 px-4 py-2.5 rounded-lg shadow-lg text-sm font-medium border transition-all ${toast.isError
                    ? "bg-red-500/10 border-red-500/30 text-red-400"
                    : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                    }`}>
                    {toast.msg}
                </div>
            )}

            {/* ── Error banner ── */}
            {error && (
                <div className="bg-red-500/10 border-b border-red-500/30 text-red-400 px-6 py-2.5 text-sm flex items-center gap-2">
                    <span className="material-symbols-outlined text-[18px]">error</span>
                    {error}
                    <button onClick={loadData} className="ml-auto text-xs underline hover:text-red-300">Retry</button>
                </div>
            )}

            {/* ── Header ── */}
            <header className="flex-none border-b border-slate-200 dark:border-surface-border bg-white dark:bg-[#111a22] px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="size-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                            <span className="material-symbols-outlined text-2xl">neurology</span>
                        </div>
                        <div>
                            <h1 className="text-xl font-bold leading-tight tracking-tight">Automation Control Center</h1>
                            <p className="text-slate-500 dark:text-slate-400 text-sm font-medium">
                                Autonomous retraining and strategy orchestration
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">

                        <div className="flex items-center gap-3 bg-slate-100 dark:bg-surface-dark px-4 py-2 rounded-lg border border-slate-200 dark:border-surface-border">
                            <span className="text-sm font-semibold text-slate-600 dark:text-slate-300">Autonomous Mode</span>
                            <label className="relative inline-flex items-center cursor-pointer">
                                <input
                                    type="checkbox"
                                    className="sr-only peer"
                                    checked={autoEnabled}
                                    onChange={handleToggle}
                                />
                                <div className="w-11 h-6 bg-slate-300 peer-focus:outline-none rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-primary" />
                            </label>
                            <span className={`text-xs font-bold uppercase tracking-wide ${autoEnabled ? "text-primary" : "text-slate-400"}`}>
                                {autoEnabled ? "Enabled" : "Disabled"}
                            </span>
                        </div>
                    </div>
                </div>
            </header>

            {/* ── Body ── */}
            <div className="flex flex-1 overflow-hidden">
                <main className="flex-1 flex flex-col min-w-0 overflow-y-auto p-6 gap-6">

                    {/* ── Loading ── */}
                    {loading && (
                        <div className="flex items-center justify-center py-20">
                            <span className="material-symbols-outlined text-3xl animate-spin text-slate-500">sync</span>
                        </div>
                    )}

                    {!loading && (
                        <>
                            {/* ── Retraining Alert Banner ── */}
                            {(() => {
                                const pendingDec = history.find(log => log.status === "pending_human_review" && log.action.toUpperCase() === "RETRAIN");
                                if (!pendingDec) return null;
                                return (
                                    <div className="bg-[#137fec]/10 border border-[#137fec]/20 text-[#137fec] rounded-xl p-4 flex items-center justify-between shadow-sm">
                                        <div className="flex items-start gap-3">
                                            <span className="material-symbols-outlined text-primary text-[24px] mt-0.5">info</span>
                                            <div>
                                                <h4 className="font-bold text-sm text-[#137fec]">Retraining Approval Required</h4>
                                                <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 font-medium leading-relaxed">
                                                    The LLM Decision Engine has recommended a <strong>Model Retraining</strong> strategy due to detected drift.
                                                    You can approve the retraining here by selecting decision <strong>{shortId(pendingDec.id)}</strong> and clicking <strong>"Approve RETRAIN"</strong> in the details panel, or on the Monitoring page.
                                                </p>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => selectDecision(pendingDec)}
                                            className="px-4 py-2 bg-[#137fec] hover:bg-[#0f66bd] text-white text-xs font-semibold rounded-lg shadow-md transition-colors whitespace-nowrap ml-4"
                                        >
                                            View & Approve
                                        </button>
                                    </div>
                                );
                            })()}

                            {/* ── Stat cards ── */}
                            <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                <StatCard
                                    icon="auto_awesome"
                                    iconColor="text-primary"
                                    iconBg="bg-primary/10"
                                    label="Total Automated Decisions"
                                    value={status?.total_decisions?.toLocaleString() ?? "0"}
                                />
                                <StatCard
                                    icon="model_training"
                                    iconColor="text-purple-500"
                                    iconBg="bg-purple-500/10"
                                    label="Retraining Executed"
                                    value={status?.retraining_executed?.toLocaleString() ?? "0"}
                                />
                                <StatCard
                                    icon="pending_actions"
                                    iconColor="text-warning"
                                    iconBg="bg-warning/10"
                                    label="Pending Approvals"
                                    value={status?.pending_approvals?.toString() ?? "0"}
                                    valueColor={status?.pending_approvals > 0 ? "text-warning" : ""}
                                    highlight={status?.pending_approvals > 0}
                                />
                                <StatCard
                                    icon="verified"
                                    iconColor="text-blue-500"
                                    iconBg="bg-blue-500/10"
                                    label="Avg Decision Confidence"
                                    value={pct(status?.avg_confidence)}
                                />
                            </section>

                            {/* ── Decision history table ── */}
                            <section className="flex flex-col flex-1 min-h-0 bg-white dark:bg-surface-dark rounded-lg border border-slate-200 dark:border-surface-border shadow-sm overflow-hidden">
                                <div className="p-4 border-b border-slate-200 dark:border-surface-border flex items-center justify-between">
                                    <h2 className="text-lg font-bold">Decision History</h2>
                                    <span className="text-xs text-slate-400">{history.length} total</span>
                                </div>

                                {history.length === 0 ? (
                                    <div className="flex-1 flex flex-col items-center justify-center py-16">
                                        <span className="material-symbols-outlined text-4xl text-slate-500 mb-3">smart_toy</span>
                                        <p className="text-slate-400 text-sm">No automation decisions yet</p>
                                    </div>
                                ) : (
                                    <div className="flex-1 overflow-auto">
                                        <table className="w-full text-left border-collapse">
                                            <thead className="bg-slate-50 dark:bg-[#151f28] sticky top-0 z-10">
                                                <tr>
                                                    <th className="px-5 py-3 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">ID</th>
                                                    <th className="px-5 py-3 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Model</th>
                                                    <th className="px-5 py-3 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Drift Type</th>
                                                    <th className="px-5 py-3 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Severity</th>
                                                    <th className="px-5 py-3 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Strategy</th>
                                                    <th className="px-5 py-3 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Res. Profile</th>
                                                    <th className="px-5 py-3 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Conf.</th>
                                                    <th className="px-5 py-3 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Status</th>
                                                    <th className="px-5 py-3 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Timestamp</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-slate-200 dark:divide-surface-border text-sm">
                                                {history.map((row) => {
                                                    const isSelected = selected?.id === row.id;
                                                    const cb = confBar(row.confidence);
                                                    return (
                                                        <tr
                                                            key={row.id}
                                                            onClick={() => selectDecision(row)}
                                                            className={`cursor-pointer transition-colors border-l-4 ${isSelected
                                                                ? "bg-primary/5 dark:bg-primary/10 border-l-primary"
                                                                : "hover:bg-slate-50 dark:hover:bg-white/5 border-l-transparent"
                                                                }`}
                                                        >
                                                            <td className={`px-5 py-3.5 font-mono text-xs font-medium ${isSelected ? "text-primary" : "text-slate-500 dark:text-slate-400"}`}>
                                                                {shortId(row.id)}
                                                            </td>
                                                            <td className="px-5 py-3.5 font-medium">{row.model_name}</td>
                                                            <td className="px-5 py-3.5 text-slate-500 dark:text-slate-400 capitalize">{row.drift_type}</td>
                                                            <td className="px-5 py-3.5">
                                                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${SEV[row.severity] || SEV.low}`}>
                                                                    {row.severity}
                                                                </span>
                                                            </td>
                                                            <td className="px-5 py-3.5 font-mono text-xs">{row.strategy}</td>
                                                            <td className="px-5 py-3.5 text-slate-500 dark:text-slate-400 text-xs">{row.resource_profile}</td>
                                                            <td className="px-5 py-3.5">
                                                                <div className="flex items-center gap-2">
                                                                    <div className="w-16 h-1.5 bg-slate-200 dark:bg-surface-border rounded-full overflow-hidden">
                                                                        <div className={`h-full ${cb.bg}`} style={{ width: cb.width }} />
                                                                    </div>
                                                                    <span className="text-xs font-medium">{pct(row.confidence)}</span>
                                                                </div>
                                                            </td>
                                                            <td className="px-5 py-3.5">
                                                                <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${STAT[row.status] || STAT.pending}`}>
                                                                    {isPending(row.status) && (
                                                                        <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                                                                    )}
                                                                    {row.status}
                                                                </span>
                                                            </td>
                                                            <td className="px-5 py-3.5 text-slate-500 dark:text-slate-400 font-mono text-xs">
                                                                {fmtDate(row.created_at)}
                                                            </td>
                                                        </tr>
                                                    );
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </section>
                        </>
                    )}
                </main>

                {/* ── Detail panel (aside) ── */}
                <aside className="w-[420px] flex-none border-l border-slate-200 dark:border-surface-border bg-white dark:bg-[#151f28] flex flex-col h-full shadow-2xl z-20">
                    {!selected ? (
                        <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
                            <span className="material-symbols-outlined text-4xl text-slate-500 mb-3">neurology</span>
                            <p className="text-slate-400 text-sm">Select a decision to view details</p>
                        </div>
                    ) : loadingDetail ? (
                        <div className="flex-1 flex items-center justify-center">
                            <span className="material-symbols-outlined text-3xl animate-spin text-slate-500">sync</span>
                        </div>
                    ) : detail ? (
                        <>
                            {/* Panel header */}
                            <div className="flex-none p-6 border-b border-slate-200 dark:border-surface-border">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <div className="flex items-center gap-2 mb-1">
                                            <h3 className="text-xl font-bold">Decision Details</h3>
                                            <span className="bg-primary/10 text-primary text-xs px-2 py-0.5 rounded font-mono">
                                                {shortId(detail.id)}
                                            </span>
                                        </div>
                                        <p className="text-sm text-slate-500 dark:text-slate-400">{detail.model_name}</p>
                                    </div>
                                    <button
                                        onClick={() => { setSelected(null); setDetail(null); clearInterval(countdownRef.current); }}
                                        className="text-slate-400 hover:text-white transition-colors"
                                    >
                                        <span className="material-symbols-outlined">close</span>
                                    </button>
                                </div>
                            </div>

                            {/* Scrollable content */}
                            <div className="flex-1 overflow-y-auto p-6 space-y-6">

                                {/* ── Drift Context ── */}
                                <div className="space-y-3">
                                    <div className="flex items-center justify-between">
                                        <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                                            Drift Context
                                        </h4>
                                        <span className="text-xs font-mono text-danger capitalize">
                                            {detail.drift_context?.drift_type}
                                        </span>
                                    </div>
                                    <div className="bg-slate-50 dark:bg-[#111a22] p-4 rounded-lg border border-slate-200 dark:border-surface-border">
                                        <div className="flex justify-between items-end mb-2">
                                            <span className="text-xs font-medium">Feature Drift Score</span>
                                            <span className={`text-sm font-bold ${(detail.drift_context?.drift_score || 0) > 0.5 ? "text-danger" : "text-warning"
                                                }`}>
                                                {detail.drift_context?.severity} ({detail.drift_context?.drift_score?.toFixed(2) ?? "—"})
                                            </span>
                                        </div>
                                        <div className="h-2 w-full bg-slate-200 dark:bg-surface-border rounded-full overflow-hidden mb-4">
                                            <div
                                                className="h-full bg-gradient-to-r from-warning to-danger rounded-full transition-all"
                                                style={{ width: `${Math.min((detail.drift_context?.drift_score || 0) * 100, 100)}%` }}
                                            />
                                        </div>
                                        {(detail.drift_context?.training_mean != null || detail.drift_context?.current_mean != null) && (
                                            <div className="flex gap-2">
                                                <div className="flex-1 bg-white dark:bg-surface-dark p-2 rounded border border-slate-200 dark:border-surface-border text-center">
                                                    <div className="text-[10px] text-slate-500 uppercase">Training Mean</div>
                                                    <div className="font-mono text-sm">{detail.drift_context?.training_mean?.toFixed(3) ?? "—"}</div>
                                                </div>
                                                <div className="flex-1 bg-white dark:bg-surface-dark p-2 rounded border border-slate-200 dark:border-surface-border text-center">
                                                    <div className="text-[10px] text-slate-500 uppercase">Current Mean</div>
                                                    <div className="font-mono text-sm text-danger">{detail.drift_context?.current_mean?.toFixed(3) ?? "—"}</div>
                                                </div>
                                            </div>
                                        )}
                                        {detail.drift_context?.affected_features?.length > 0 && (
                                            <div className="mt-3 pt-3 border-t border-slate-200 dark:border-surface-border">
                                                <div className="text-[10px] text-slate-500 uppercase mb-1.5">Affected Features</div>
                                                <div className="flex flex-wrap gap-1.5">
                                                    {detail.drift_context.affected_features.map((f, i) => (
                                                        <span
                                                            key={i}
                                                            className="bg-warning/10 text-warning border border-warning/20 px-2 py-0.5 rounded text-xs font-mono"
                                                        >
                                                            {f}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* ── LLM Decision Output ── */}
                                <div className="space-y-3">
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                                        LLM Decision Output
                                    </h4>
                                    <div className="bg-[#0d131a] p-4 rounded-lg border border-slate-800 font-mono text-xs leading-relaxed text-slate-300">
                                        <p className="mb-2"><span className="text-primary">&gt;&gt;&gt; ANALYSIS_START</span></p>
                                        <p className="mb-2 whitespace-pre-wrap">{detail.llm_output?.raw_reasoning}</p>
                                        <p className="mb-2"><span className="text-primary">&gt;&gt;&gt; STRATEGY_RECOMMENDATION</span></p>
                                        <p>Strategy: <span className="text-success">{detail.llm_output?.strategy?.toUpperCase()}</span></p>
                                        <p className="mt-2"><span className="text-primary">&gt;&gt;&gt; ANALYSIS_END</span></p>
                                    </div>
                                    <div className="grid grid-cols-2 gap-2">
                                        <InfoCell label="Resource Profile" value={detail.llm_output?.resource_profile} />
                                        <InfoCell label="Strategy" value={detail.llm_output?.strategy} />
                                        <InfoCell
                                            label="Early Stopping"
                                            value={detail.llm_output?.early_stopping ? "Enabled" : "Disabled"}
                                            color={detail.llm_output?.early_stopping ? "text-success" : "text-slate-400"}
                                        />
                                        <InfoCell
                                            label="Ensemble"
                                            value={detail.llm_output?.ensemble ? "Enabled" : "Disabled"}
                                            color={detail.llm_output?.ensemble ? "text-success" : "text-slate-400"}
                                        />
                                    </div>
                                </div>

                                {/* ── Projected Impact ── */}
                                <div className="space-y-3">
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                                        Projected Impact
                                    </h4>
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="bg-slate-50 dark:bg-[#111a22] p-3 rounded-lg border border-slate-200 dark:border-surface-border">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="material-symbols-outlined text-lg text-slate-400">payments</span>
                                                <span className="text-xs font-medium text-slate-500">Est. Cost</span>
                                            </div>
                                            <div className="text-lg font-bold">
                                                {detail.impact?.estimated_cost != null
                                                    ? `$${typeof detail.impact.estimated_cost === "number"
                                                        ? detail.impact.estimated_cost.toFixed(2)
                                                        : detail.impact.estimated_cost}`
                                                    : "—"}
                                            </div>
                                        </div>
                                        <div className="bg-slate-50 dark:bg-[#111a22] p-3 rounded-lg border border-slate-200 dark:border-surface-border">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="material-symbols-outlined text-lg text-slate-400">timer</span>
                                                <span className="text-xs font-medium text-slate-500">Downtime</span>
                                            </div>
                                            <div className="text-lg font-bold">{detail.impact?.estimated_downtime ?? "—"}</div>
                                        </div>
                                        {detail.impact?.expected_improvement != null && (
                                            <div className="col-span-2 bg-slate-50 dark:bg-[#111a22] p-3 rounded-lg border border-slate-200 dark:border-surface-border">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <span className="material-symbols-outlined text-lg text-emerald-400">trending_up</span>
                                                    <span className="text-xs font-medium text-slate-500">Expected Improvement</span>
                                                </div>
                                                <div className="text-lg font-bold text-emerald-400">
                                                    +{(detail.impact.expected_improvement * 100).toFixed(1)}%
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* ── Actions footer ── */}
                            <div className="flex-none p-6 border-t border-slate-200 dark:border-surface-border bg-slate-50 dark:bg-[#111a22]">
                                <div className="grid grid-cols-2 gap-3 mb-3">
                                    {detail.llm_output?.strategy?.toUpperCase() === "RETRAIN" ? (
                                        <button
                                            onClick={() => handleAction("approve")}
                                            disabled={!isPending(detail.status) || actionLoading === "approve"}
                                            className="col-span-2 bg-primary hover:bg-primary-dark disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold py-2.5 px-4 rounded-lg shadow-lg shadow-primary/20 flex items-center justify-center gap-2 transition-all"
                                        >
                                            {actionLoading === "approve" ? (
                                                <span className="material-symbols-outlined text-xl animate-spin">sync</span>
                                            ) : (
                                                <span className="material-symbols-outlined text-xl">check_circle</span>
                                            )}
                                            Approve RETRAIN
                                        </button>
                                    ) : (
                                        <div className="col-span-2 bg-slate-100 dark:bg-[#151f28] text-slate-500 dark:text-slate-400 text-sm font-medium py-2.5 px-4 rounded-lg border border-slate-200 dark:border-surface-border flex items-center justify-center gap-2">
                                            <span className="material-symbols-outlined text-lg">info</span>
                                            Monitoring alert — no manual action required
                                        </div>
                                    )}
                                    <button
                                        onClick={() => handleAction("reject")}
                                        disabled={!isPending(detail.status) || actionLoading === "reject"}
                                        className="border border-slate-300 dark:border-surface-border hover:bg-slate-100 dark:hover:bg-surface-dark disabled:opacity-40 disabled:cursor-not-allowed text-slate-700 dark:text-slate-300 font-medium py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
                                    >
                                        {actionLoading === "reject" ? (
                                            <span className="material-symbols-outlined text-lg animate-spin">sync</span>
                                        ) : (
                                            <span className="material-symbols-outlined text-lg">cancel</span>
                                        )}
                                        Reject
                                    </button>
                                    <button
                                        onClick={() => handleAction("manual")}
                                        disabled={!isPending(detail.status) || actionLoading === "manual"}
                                        className="border border-slate-300 dark:border-surface-border hover:bg-slate-100 dark:hover:bg-surface-dark disabled:opacity-40 disabled:cursor-not-allowed text-slate-700 dark:text-slate-300 font-medium py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
                                    >
                                        {actionLoading === "manual" ? (
                                            <span className="material-symbols-outlined text-lg animate-spin">sync</span>
                                        ) : (
                                            <span className="material-symbols-outlined text-lg">build</span>
                                        )}
                                        Manual Train
                                    </button>
                                </div>
                                {isPending(detail.status) && countdown != null && (
                                    <p className="text-center text-xs text-slate-400 dark:text-slate-500">
                                        Auto-approval in <span className="font-mono text-slate-300">{fmtCountdown(countdown)}</span> if no action taken.
                                    </p>
                                )}
                                {!isPending(detail.status) && (
                                    <p className="text-center text-xs text-slate-400 dark:text-slate-500 capitalize">
                                        Decision {detail.status}
                                    </p>
                                )}
                            </div>
                        </>
                    ) : null}
                </aside>
            </div>
        </>
    );
}

/* ── Stat card component ── */
function StatCard({ icon, iconColor, iconBg, label, value, valueColor, highlight }) {
    return (
        <div className={`bg-white dark:bg-surface-dark border border-slate-200 dark:border-surface-border rounded-lg p-5 shadow-sm hover:border-primary/50 transition-colors group relative overflow-hidden ${highlight ? "" : ""
            }`}>
            {highlight && (
                <div className="absolute right-0 top-0 w-16 h-16 bg-gradient-to-br from-warning/20 to-transparent rounded-bl-full -mr-4 -mt-4" />
            )}
            <div className="flex justify-between items-start mb-2 relative z-10">
                <div className={`p-2 rounded-md ${iconBg} ${iconColor} group-hover:brightness-125 transition-all`}>
                    <span className="material-symbols-outlined text-xl">{icon}</span>
                </div>
            </div>
            <p className="text-slate-500 dark:text-slate-400 text-sm font-medium">{label}</p>
            <p className={`text-3xl font-bold tracking-tight mt-1 ${valueColor || ""}`}>{value}</p>
        </div>
    );
}

/* ── Info cell for detail panel ── */
function InfoCell({ label, value, color }) {
    return (
        <div className="bg-slate-50 dark:bg-[#111a22] p-2.5 rounded-lg border border-slate-200 dark:border-surface-border">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
            <div className={`text-sm font-medium font-mono mt-0.5 ${color || "text-white"}`}>{value ?? "—"}</div>
        </div>
    );
}
