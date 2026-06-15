import { useState, useEffect } from "react";
import {
    ResponsiveContainer,
    AreaChart,
    Area,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    Cell,
} from "recharts";
import {
    getMonitoringSummary,
    getFeatureList,
    getFeatureDetails,
} from "../api/monitoring";
import {
    getDecisionHistory,
    approveDecision,
} from "../api/automation";
import {
    retrainModel,
} from "../api/training";

/* ── helpers ──────────────────────────────────────────────── */
function pct(v) {
    return v != null ? `${(v * 100).toFixed(1)}%` : "—";
}
function shortId(id) {
    if (!id) return "—";
    return `#DEC-${id.slice(0, 4).toUpperCase()}`;
}
function driftColor(d) {
    if (d == null) return "text-slate-400";
    if (d > 0.2) return "text-red-400";
    if (d > 0.1) return "text-amber-400";
    return "text-emerald-400";
}
function driftBg(d) {
    if (d > 0.2) return "bg-red-500/10 border-red-500/20";
    if (d > 0.1) return "bg-amber-500/10 border-amber-500/20";
    return "bg-emerald-500/10 border-emerald-500/20";
}
function severityBadge(sev) {
    const m = {
        low: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
        medium: "bg-amber-500/10 text-amber-400 border-amber-500/20",
        high: "bg-red-500/10 text-red-400 border-red-500/20",
    };
    return m[sev] || m.low;
}

/* ── Bar color helper ── */
function barFill(v) {
    if (v > 0.2) return "#ef4444";
    if (v > 0.15) return "#f59e0b";
    return "#22c55e";
}

/* ══════════════════════════════════════════════════════════
   MONITORING PAGE
   ══════════════════════════════════════════════════════════ */
export default function Monitoring() {
    const [summary, setSummary] = useState(null);
    const [features, setFeatures] = useState([]);
    const [selectedFeature, setSelectedFeature] = useState(null);
    const [featureDetails, setFeatureDetails] = useState(null);
    const [loading, setLoading] = useState(true);
    const [loadingDetail, setLoadingDetail] = useState(false);
    const [history, setHistory] = useState([]);

    /* ── load on mount ── */
    useEffect(() => {
        (async () => {
            try {
                const [sum, feats, hist] = await Promise.all([
                    getMonitoringSummary(),
                    getFeatureList(),
                    getDecisionHistory(),
                ]);
                setSummary(sum);
                setFeatures(feats);
                setHistory(hist);
                // Auto-select first feature
                if (feats.length > 0) {
                    selectFeature(feats[0]);
                }
            } catch (err) {
                console.error("Failed to load monitoring data:", err);
            } finally {
                setLoading(false);
            }
        })();
    }, []);

    const selectFeature = async (feat) => {
        setSelectedFeature(feat);
        setLoadingDetail(true);
        try {
            const detail = await getFeatureDetails(feat.feature_name);
            setFeatureDetails(detail);
        } catch (err) {
            console.error("Failed to load feature details:", err);
            setFeatureDetails(null);
        } finally {
            setLoadingDetail(false);
        }
    };
    

    /* ── derived values ── */
    const accuracyDelta =
        summary?.accuracy != null && summary?.baseline_accuracy != null
            ? summary.accuracy - summary.baseline_accuracy
            : null;

    /* ── distribution chart data ── */
    const distributionData =
        featureDetails?.distribution
            ? featureDetails.distribution.baseline.map((b, i) => ({
                idx: i,
                baseline: b,
                current: featureDetails.distribution.current[i] ?? 0,
            }))
            : [];

    return (
        <>
            {/* ── Header ── */}
            <header className="flex items-center justify-between px-6 py-3 border-b border-slate-200 dark:border-[#2d3f50] bg-white dark:bg-[#1a2632] shrink-0 z-10">
                <div className="flex flex-col gap-0.5">
                    <h1 className="text-xl font-bold tracking-tight dark:text-white text-slate-900">
                        Drift Intelligence
                    </h1>
                    <p className="text-xs dark:text-slate-400 text-slate-500">
                        {summary?.model_name
                            ? `Monitoring ${summary.model_name} · ${summary.window} window`
                            : "Real-time feature drift analysis"}
                    </p>
                </div>
            </header>

            <div className="flex-1 overflow-y-auto p-6 lg:p-10">
                <div className="max-w-7xl mx-auto flex flex-col gap-6">

                    {/* ── Loading state ── */}
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
                                    <div className="bg-[#137fec]/10 border border-[#137fec]/20 text-[#137fec] rounded-xl p-4 flex items-center justify-between shadow-sm mb-4">
                                        <div className="flex items-start gap-3">
                                            <span className="material-symbols-outlined text-primary text-[24px] mt-0.5">info</span>
                                            <div>
                                                <h4 className="font-bold text-sm text-[#137fec]">Retraining Approval Required</h4>
                                                <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 font-medium leading-relaxed">
                                                    The LLM Decision Engine has recommended a <strong>Model Retraining</strong> strategy due to detected drift.
                                                    Please review and approve this action on the Automation Control Center.
                                                </p>
                                            </div>
                                        </div>
                                        <a
                                            href="/automation"
                                            className="px-4 py-2 bg-[#137fec] hover:bg-[#0f66bd] text-white text-xs font-semibold rounded-lg shadow-md transition-colors whitespace-nowrap ml-4 flex items-center gap-1"
                                        >
                                            Go to Automation
                                            <span className="material-symbols-outlined text-xs">arrow_forward</span>
                                        </a>
                                    </div>
                                );
                            })()}

                            {/* ── Stat cards ── */}
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                {/* Drift score */}
                                <div className="dark:bg-[#1a2632] bg-white p-5 rounded-xl border dark:border-[#2d3f50] border-slate-200 shadow-sm">
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <p className="text-xs font-medium dark:text-slate-400 text-slate-500 uppercase tracking-wider">
                                                Global Drift
                                            </p>
                                            <h3 className={`text-2xl font-bold mt-1 ${driftColor(summary?.global_drift)}`}>
                                                {pct(summary?.global_drift)}
                                            </h3>
                                        </div>
                                        <span className={`material-symbols-outlined p-2 rounded-lg ${(summary?.global_drift || 0) > 0.2
                                            ? "text-red-400 bg-red-500/10"
                                            : (summary?.global_drift || 0) > 0.1
                                                ? "text-amber-400 bg-amber-500/10"
                                                : "text-emerald-400 bg-emerald-500/10"
                                            }`}>
                                            trending_up
                                        </span>
                                    </div>
                                </div>

                                {/* Accuracy */}
                                <div className="dark:bg-[#1a2632] bg-white p-5 rounded-xl border dark:border-[#2d3f50] border-slate-200 shadow-sm">
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <p className="text-xs font-medium dark:text-slate-400 text-slate-500 uppercase tracking-wider">
                                                Accuracy
                                            </p>
                                            <h3 className="text-2xl font-bold dark:text-white text-slate-900 mt-1">
                                                {pct(summary?.accuracy)}
                                            </h3>
                                            {accuracyDelta != null && (
                                                <span className={`text-xs font-medium ${accuracyDelta < 0 ? "text-red-400" : "text-emerald-400"}`}>
                                                    {accuracyDelta >= 0 ? "+" : ""}{(accuracyDelta * 100).toFixed(2)}% vs baseline
                                                </span>
                                            )}
                                        </div>
                                        <span className="material-symbols-outlined text-[#137fec] bg-[#137fec]/10 p-2 rounded-lg">
                                            query_stats
                                        </span>
                                    </div>
                                </div>

                                {/* Data quality */}
                                <div className="dark:bg-[#1a2632] bg-white p-5 rounded-xl border dark:border-[#2d3f50] border-slate-200 shadow-sm">
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <p className="text-xs font-medium dark:text-slate-400 text-slate-500 uppercase tracking-wider">
                                                Data Quality
                                            </p>
                                            <h3 className="text-2xl font-bold text-emerald-400 mt-1">
                                                {pct(summary?.data_quality)}
                                            </h3>
                                        </div>
                                        <span className="material-symbols-outlined text-emerald-400 bg-emerald-500/10 p-2 rounded-lg">
                                            verified
                                        </span>
                                    </div>
                                </div>

                                {/* Alerts */}
                                <div className="dark:bg-[#1a2632] bg-white p-5 rounded-xl border dark:border-[#2d3f50] border-slate-200 shadow-sm">
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <p className="text-xs font-medium dark:text-slate-400 text-slate-500 uppercase tracking-wider">
                                                Active Alerts
                                            </p>
                                            <h3 className="text-2xl font-bold dark:text-white text-slate-900 mt-1">
                                                {summary?.alerts ?? 0}
                                            </h3>
                                        </div>
                                        <span className={`material-symbols-outlined p-2 rounded-lg ${(summary?.alerts || 0) > 0
                                            ? "text-red-400 bg-red-500/10"
                                            : "text-slate-400 bg-slate-500/10"
                                            }`}>
                                            notifications_active
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* ── Main grid ── */}
                            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 items-start">

                                {/* ── Left: Feature table ── */}
                                <div className="lg:col-span-2 dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm overflow-hidden">
                                    <div className="px-6 py-4 border-b dark:border-[#2d3f50] border-slate-200">
                                        <h2 className="dark:text-white text-slate-900 text-base font-bold flex items-center gap-2">
                                            <span className="material-symbols-outlined text-[18px] text-[#137fec]">
                                                swap_vert
                                            </span>
                                            Feature Drift Signals
                                        </h2>
                                    </div>

                                    {features.length === 0 ? (
                                        <div className="px-6 py-12 text-center">
                                            <span className="material-symbols-outlined text-3xl text-slate-500 mb-2 block">
                                                monitoring
                                            </span>
                                            <p className="text-sm text-slate-400">No feature drift data available</p>
                                        </div>
                                    ) : (
                                        <div className="max-h-[540px] overflow-y-auto divide-y dark:divide-[#2d3f50] divide-slate-100">
                                            {features.map((f) => (
                                                <button
                                                    key={f.feature_name}
                                                    onClick={() => selectFeature(f)}
                                                    className={`w-full text-left px-5 py-3.5 transition-colors ${selectedFeature?.feature_name === f.feature_name
                                                        ? "bg-[#137fec]/5 dark:bg-[#137fec]/10 border-l-2 border-[#137fec]"
                                                        : "hover:bg-slate-50 dark:hover:bg-white/5 border-l-2 border-transparent"
                                                        }`}
                                                >
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-sm font-semibold dark:text-white text-slate-900 truncate">
                                                            {f.feature_name}
                                                        </span>
                                                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${severityBadge(f.severity)}`}>
                                                            {f.severity}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center justify-between text-xs dark:text-slate-400 text-slate-500">
                                                        <span className="capitalize">{f.type}</span>
                                                        <span className={`font-mono font-medium ${driftColor(f.drift_score)}`}>
                                                            {pct(f.drift_score)}
                                                        </span>
                                                    </div>
                                                    {/* Drift bar */}
                                                    <div className="mt-1.5 w-full h-1 rounded-full bg-slate-700/30 overflow-hidden">
                                                        <div
                                                            className={`h-1 rounded-full transition-all ${f.drift_score > 0.2 ? "bg-red-500" : f.drift_score > 0.1 ? "bg-amber-500" : "bg-emerald-500"
                                                                }`}
                                                            style={{ width: `${Math.min(f.drift_score * 100, 100)}%` }}
                                                        />
                                                    </div>
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                {/* ── Right: Detail panel ── */}
                                <div className="lg:col-span-3">
                                    {!selectedFeature ? (
                                        <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-12 text-center">
                                            <span className="material-symbols-outlined text-4xl text-slate-500 mb-3 block">
                                                monitoring
                                            </span>
                                            <p className="text-slate-400 text-sm">Select a feature to view drift details</p>
                                        </div>
                                    ) : loadingDetail ? (
                                        <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-12 text-center">
                                            <span className="material-symbols-outlined text-2xl animate-spin text-slate-500">sync</span>
                                            <p className="mt-2 text-sm text-slate-400">Loading details…</p>
                                        </div>
                                    ) : (
                                        <div className="flex flex-col gap-4">

                                            {/* ── Recommendation card ── */}
                                            {featureDetails?.recommendation && (
                                                <div className={`rounded-xl border shadow-sm p-5 ${featureDetails.recommendation.action === "retrain"
                                                    ? "dark:bg-red-500/5 bg-red-50 dark:border-red-500/20 border-red-200"
                                                    : "dark:bg-[#1a2632] bg-white dark:border-[#2d3f50] border-slate-200"
                                                    }`}>
                                                    <div className="flex items-start justify-between mb-3">
                                                        <div className="flex items-center gap-2">
                                                            <span className={`material-symbols-outlined text-[20px] ${featureDetails.recommendation.action === "retrain"
                                                                ? "text-red-400"
                                                                : "text-[#137fec]"
                                                                }`}>
                                                                smart_toy
                                                            </span>
                                                            <h3 className="text-sm font-bold dark:text-white text-slate-900">
                                                                Autonomous Recommendation
                                                            </h3>
                                                        </div>
                                                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium border ${featureDetails.recommendation.action === "retrain"
                                                            ? "bg-red-500/10 text-red-400 border-red-500/20"
                                                            : "bg-[#137fec]/10 text-[#137fec] border-[#137fec]/20"
                                                            }`}>
                                                            {featureDetails.recommendation.action}
                                                        </span>
                                                    </div>

                                                    <div className="grid grid-cols-3 gap-3 mb-4">
                                                        <div className="dark:bg-[#101922]/50 bg-white/70 p-3 rounded-lg">
                                                            <p className="text-xs dark:text-slate-400 text-slate-500">Confidence</p>
                                                            <p className="text-sm font-bold dark:text-white text-slate-900 mt-0.5">
                                                                {pct(featureDetails.recommendation.confidence)}
                                                            </p>
                                                        </div>
                                                        <div className="dark:bg-[#101922]/50 bg-white/70 p-3 rounded-lg">
                                                            <p className="text-xs dark:text-slate-400 text-slate-500">Est. Cost</p>
                                                            <p className="text-sm font-bold dark:text-white text-slate-900 mt-0.5">
                                                                ${featureDetails.recommendation.estimated_cost?.toFixed(2)}
                                                            </p>
                                                        </div>
                                                        <div className="dark:bg-[#101922]/50 bg-white/70 p-3 rounded-lg">
                                                            <p className="text-xs dark:text-slate-400 text-slate-500">Duration</p>
                                                            <p className="text-sm font-bold dark:text-white text-slate-900 mt-0.5">
                                                                {featureDetails.recommendation.estimated_duration_minutes} min
                                                            </p>
                                                        </div>
                                                    </div>

                                                    {featureDetails.recommendation.action === "retrain" ? (
                                                        <div className="text-center py-2">
                                                            <p className="text-xs dark:text-slate-400 text-slate-500 mb-2 font-medium">
                                                                Retraining approval is required. Please review and approve the action on the Automation Control Center.
                                                            </p>
                                                            <a
                                                                href="/automation"
                                                                className="w-full inline-flex items-center justify-center gap-1.5 bg-[#137fec] hover:bg-[#0f66bd] text-white font-semibold py-2.5 px-4 rounded-lg transition-colors shadow-lg shadow-[#137fec]/20"
                                                            >
                                                                Go to Automation
                                                                <span className="material-symbols-outlined text-xs">arrow_forward</span>
                                                            </a>
                                                        </div>
                                                    ) : (
                                                        <p className="text-xs dark:text-slate-400 text-slate-500 text-center py-2">
                                                            No action required — continue monitoring.
                                                        </p>
                                                    )}
                                                </div>
                                            )}

                                            {/* ── Distribution shift chart ── */}
                                            {distributionData.length > 0 && (
                                                <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-5">
                                                    <h3 className="text-sm font-bold dark:text-white text-slate-900 mb-3 flex items-center gap-2">
                                                        <span className="material-symbols-outlined text-[16px] text-indigo-400">
                                                            stacked_line_chart
                                                        </span>
                                                        Distribution Shift
                                                    </h3>
                                                    <ResponsiveContainer width="100%" height={200}>
                                                        <AreaChart data={distributionData}>
                                                            <CartesianGrid strokeDasharray="3 3" stroke="#2d3f50" />
                                                            <XAxis dataKey="idx" hide />
                                                            <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} width={50} />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: "#1a2632",
                                                                    border: "1px solid #2d3f50",
                                                                    borderRadius: 8,
                                                                    color: "#f1f5f9",
                                                                    fontSize: 12,
                                                                }}
                                                            />
                                                            <Area
                                                                type="monotone"
                                                                dataKey="baseline"
                                                                stroke="#6366f1"
                                                                fill="#6366f1"
                                                                fillOpacity={0.15}
                                                                name="Baseline"
                                                            />
                                                            <Area
                                                                type="monotone"
                                                                dataKey="current"
                                                                stroke="#f59e0b"
                                                                fill="#f59e0b"
                                                                fillOpacity={0.15}
                                                                name="Current"
                                                            />
                                                        </AreaChart>
                                                    </ResponsiveContainer>
                                                    <div className="flex items-center justify-center gap-6 mt-2 text-xs dark:text-slate-400">
                                                        <span className="flex items-center gap-1.5">
                                                            <span className="w-3 h-0.5 bg-indigo-500 rounded" /> Baseline
                                                        </span>
                                                        <span className="flex items-center gap-1.5">
                                                            <span className="w-3 h-0.5 bg-amber-500 rounded" /> Current
                                                        </span>
                                                    </div>
                                                </div>
                                            )}

                                            {/* ── Drift trend chart ── */}
                                            {featureDetails?.trend?.length > 0 && (
                                                <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-5">
                                                    <h3 className="text-sm font-bold dark:text-white text-slate-900 mb-3 flex items-center gap-2">
                                                        <span className="material-symbols-outlined text-[16px] text-amber-400">
                                                            bar_chart
                                                        </span>
                                                        Drift Trend (14 days)
                                                    </h3>
                                                    <ResponsiveContainer width="100%" height={180}>
                                                        <BarChart data={featureDetails.trend}>
                                                            <CartesianGrid strokeDasharray="3 3" stroke="#2d3f50" />
                                                            <XAxis
                                                                dataKey="day"
                                                                tick={{ fill: "#94a3b8", fontSize: 10 }}
                                                                tickFormatter={(d) => d.slice(5)}
                                                            />
                                                            <YAxis
                                                                tick={{ fill: "#94a3b8", fontSize: 10 }}
                                                                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                                                                width={40}
                                                                domain={[0, "auto"]}
                                                            />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: "#1a2632",
                                                                    border: "1px solid #2d3f50",
                                                                    borderRadius: 8,
                                                                    color: "#f1f5f9",
                                                                    fontSize: 12,
                                                                }}
                                                                formatter={(v) => [pct(v), "Drift"]}
                                                            />
                                                            <Bar dataKey="drift" radius={[4, 4, 0, 0]}>
                                                                {featureDetails.trend.map((entry, i) => (
                                                                    <Cell key={i} fill={barFill(entry.drift)} />
                                                                ))}
                                                            </Bar>
                                                        </BarChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            )}

                                            {/* ── Statistics cards ── */}
                                            {featureDetails?.statistics && (
                                                <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-5">
                                                    <h3 className="text-sm font-bold dark:text-white text-slate-900 mb-3 flex items-center gap-2">
                                                        <span className="material-symbols-outlined text-[16px] text-[#137fec]">
                                                            calculate
                                                        </span>
                                                        Feature Statistics
                                                    </h3>
                                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                                        <StatMini
                                                            label="Mean"
                                                            value={featureDetails.statistics.mean?.toFixed(2)}
                                                            delta={featureDetails.statistics.mean_delta}
                                                        />
                                                        <StatMini
                                                            label="Std Dev"
                                                            value={featureDetails.statistics.std?.toFixed(2)}
                                                            delta={featureDetails.statistics.std_delta}
                                                        />
                                                        <StatMini
                                                            label="Min"
                                                            value={featureDetails.statistics.min?.toFixed(2)}
                                                        />
                                                        <StatMini
                                                            label="Max"
                                                            value={featureDetails.statistics.max?.toFixed(2)}
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </>
    );
}

/* ── Stat mini card ── */
function StatMini({ label, value, delta }) {
    return (
        <div className="dark:bg-[#101922] bg-slate-50 p-3 rounded-lg border dark:border-[#2d3f50] border-slate-200">
            <p className="text-xs dark:text-slate-400 text-slate-500 uppercase tracking-wider mb-0.5">
                {label}
            </p>
            <p className="text-base font-bold dark:text-white text-slate-900 font-mono">
                {value ?? "—"}
            </p>
            {delta != null && (
                <p className={`text-xs font-medium mt-0.5 ${delta > 0 ? "text-amber-400" : "text-emerald-400"}`}>
                    Δ {delta > 0 ? "+" : ""}{typeof delta === "number" ? delta.toFixed(4) : delta}
                </p>
            )}
        </div>
    );
}
