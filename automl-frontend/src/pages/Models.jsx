import { useState, useEffect } from "react";
import { getModels, getModelVersions, activateModel, promoteVersion, deleteModel } from "../api/models";

/* ── helpers ──────────────────────────────────────────────── */
function fmtDate(iso) {
    if (!iso) return "—";
    return new Date(iso).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
    });
}

function StatusBadge({ status }) {
    const map = {
        active: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
        training: "bg-blue-500/10 text-blue-500 border-blue-500/20",
        inactive: "bg-slate-500/10 text-slate-400 border-slate-500/20",
        deprecated: "bg-slate-500/10 text-slate-400 border-slate-500/20",
    };
    const cls = map[status] || map.inactive;
    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${cls}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${status === "active" ? "bg-emerald-500" : status === "training" ? "bg-blue-500" : "bg-slate-400"}`} />
            {status}
        </span>
    );
}

/* ══════════════════════════════════════════════════════════
   MODELS PAGE
   ══════════════════════════════════════════════════════════ */
export default function Models() {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedModel, setSelectedModel] = useState(null);
    const [versions, setVersions] = useState([]);
    const [versionLoading, setVersionLoading] = useState(false);
    const [promoting, setPromoting] = useState(false);

    /* ── load models on mount ── */
    useEffect(() => { loadModels(); }, []);

    const loadModels = async () => {
        try {
            const data = await getModels();
            setModels(data);
        } catch (err) {
            console.error("Failed to load models:", err);
        } finally {
            setLoading(false);
        }
    };

    /* ── select model → fetch versions ── */
    const handleSelectModel = async (model) => {
        setSelectedModel(model);
        setVersions([]);
        setVersionLoading(true);
        try {
            const data = await getModelVersions(model.id);
            setVersions(data);
        } catch (err) {
            console.error("Failed to load versions:", err);
        } finally {
            setVersionLoading(false);
        }
    };

    /* ── promote a version ── */
    const handlePromote = async (versionNumber) => {
        if (!selectedModel) return;
        setPromoting(true);
        try {
            await promoteVersion(selectedModel.id, versionNumber);
            await loadModels();
            // refresh the selected model reference
            const updated = (await getModels()).find((m) => m.id === selectedModel.id);
            if (updated) {
                setSelectedModel(updated);
                const v = await getModelVersions(updated.id);
                setVersions(v);
            }
        } catch (err) {
            console.error("Promote failed:", err);
        } finally {
            setPromoting(false);
        }
    };

    /* ── activate model ── */
    const handleActivate = async (modelId) => {
        try {
            await activateModel(modelId);
            await loadModels();
        } catch (err) {
            console.error("Activate failed:", err);
        }
    };

    /* ── delete model ── */
    const handleDelete = async (modelId) => {
        if (!window.confirm("Are you sure you want to delete this model and all its versions?")) {
            return;
        }
        try {
            await deleteModel(modelId);
            if (selectedModel?.id === modelId) {
                setSelectedModel(null);
                setVersions([]);
            }
            await loadModels();
        } catch (err) {
            console.error("Delete failed:", err);
            alert("Failed to delete model. Check console for details.");
        }
    };

    /* ── derived ── */
    const activeCount = models.filter((m) => m.status === "active").length;

    return (
        <>
            {/* ── Header ── */}
            <header className="flex items-center justify-between px-6 py-3 border-b border-slate-200 dark:border-[#2d3f50] bg-white dark:bg-[#1a2632] shrink-0 z-10">
                <div className="flex flex-col gap-0.5">
                    <h1 className="text-xl font-bold tracking-tight dark:text-white text-slate-900">
                        Model Registry
                    </h1>
                    <p className="text-xs dark:text-slate-400 text-slate-500">
                        Manage models, versions, and deployments
                    </p>
                </div>
            </header>

            {/* ── Content ── */}
            <div className="flex-1 overflow-y-auto p-6 lg:p-10">
                <div className="max-w-7xl mx-auto flex flex-col gap-6">

                    {/* ── Stat cards ── */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <StatCard
                            label="Total Models"
                            value={loading ? "…" : models.length}
                            icon="deployed_code"
                            iconClass="dark:text-[#137fec] text-[#137fec]/80 bg-[#137fec]/10"
                        />
                        <StatCard
                            label="Active"
                            value={loading ? "…" : activeCount}
                            icon="check_circle"
                            iconClass="dark:text-emerald-400 text-emerald-600 bg-emerald-500/10"
                        />
                        <StatCard
                            label="Total Versions"
                            value={loading ? "…" : models.reduce((s, m) => s + (m.all_versions ? m.all_versions.length : (m.current_version ? 1 : 0)), 0)}
                            icon="history"
                            iconClass="dark:text-indigo-400 text-indigo-600 bg-indigo-500/10"
                        />
                    </div>

                    {/* ── Models Table ── */}
                    <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm overflow-hidden">
                        <div className="px-6 py-5 border-b dark:border-[#2d3f50] border-slate-200 flex items-center justify-between">
                            <h2 className="dark:text-white text-slate-900 text-lg font-bold">
                                Registered Models
                            </h2>
                            <span className="text-xs text-slate-400">{models.length} total</span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left border-collapse">
                                <thead>
                                    <tr className="dark:bg-[#101922]/50 bg-slate-50 border-b dark:border-[#2d3f50] border-slate-200">
                                        <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Model</th>
                                        <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Framework</th>
                                        <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Class</th>
                                        <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Version</th>
                                        <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Status</th>
                                        <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Created</th>
                                        <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider text-right">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y dark:divide-[#2d3f50] divide-slate-100">
                                    {loading && (
                                        <tr>
                                            <td colSpan={7} className="px-6 py-12 text-center text-slate-500">
                                                <span className="material-symbols-outlined text-2xl animate-spin">sync</span>
                                                <p className="mt-2 text-sm">Loading models…</p>
                                            </td>
                                        </tr>
                                    )}
                                    {!loading && models.length === 0 && (
                                        <tr>
                                            <td colSpan={7} className="px-6 py-12 text-center text-slate-500">
                                                <span className="material-symbols-outlined text-3xl mb-2 block">deployed_code</span>
                                                <p className="text-sm">No models registered yet</p>
                                            </td>
                                        </tr>
                                    )}
                                    {models.map((model) => (
                                        <tr
                                            key={model.id}
                                            onClick={() => handleSelectModel(model)}
                                            className={`group cursor-pointer transition-colors ${selectedModel?.id === model.id
                                                ? "bg-[#137fec]/5 dark:bg-[#137fec]/10"
                                                : "hover:bg-slate-50 dark:hover:bg-white/5"
                                                }`}
                                        >
                                            <td className="px-6 py-4">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-1.5 rounded bg-indigo-500/10 text-indigo-500">
                                                        <span className="material-symbols-outlined text-[18px] block">deployed_code</span>
                                                    </div>
                                                    <div>
                                                        <div className="text-sm font-semibold dark:text-white text-slate-900">
                                                            {model.model_class || model.name || "Unnamed"}
                                                        </div>
                                                        <div className="text-xs dark:text-slate-400 text-slate-500 font-mono">
                                                            {String(model.id).slice(0, 8)}…
                                                        </div>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 text-sm dark:text-slate-300 text-slate-600 capitalize">
                                                {model.framework || "—"}
                                            </td>
                                            <td className="px-6 py-4 text-sm dark:text-slate-400 text-slate-500">
                                                {model.model_class || "—"}
                                            </td>
                                            <td className="px-6 py-4 text-sm font-mono dark:text-slate-300 text-slate-600">
                                                {model.all_versions && model.all_versions.length > 0 ? (
                                                    <div className="flex flex-wrap gap-1.5">
                                                        {model.all_versions.map(v => (
                                                            <span key={v} className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider ${v === model.current_version ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' : 'bg-slate-100 dark:bg-slate-800 text-slate-500 border border-slate-200 dark:border-slate-700'}`}>
                                                                {v}
                                                            </span>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    model.current_version || "—"
                                                )}
                                            </td>
                                            <td className="px-6 py-4">
                                                <StatusBadge status={model.status || "inactive"} />
                                            </td>
                                            <td className="px-6 py-4 text-xs dark:text-slate-400 text-slate-500">
                                                {fmtDate(model.created_at)}
                                            </td>
                                            <td className="px-6 py-4 text-right">
                                                <div className="flex items-center justify-end gap-2">
                                                    {model.status !== "active" && (
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); handleActivate(model.id); }}
                                                            className="text-xs font-medium text-emerald-500 hover:text-emerald-400 transition-colors px-2 py-1 rounded hover:bg-emerald-500/10"
                                                        >
                                                            Activate
                                                        </button>
                                                    )}
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); handleSelectModel(model); }}
                                                        className="text-xs font-semibold text-[#137fec] hover:text-blue-400 transition-colors px-2 py-1 rounded hover:bg-[#137fec]/10"
                                                    >
                                                        Versions
                                                    </button>
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); handleDelete(model.id); }}
                                                        className="text-xs font-medium text-red-500 hover:text-red-400 transition-colors px-2 py-1 rounded hover:bg-red-500/10"
                                                    >
                                                        Delete
                                                    </button>
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* ── Version Panel ── */}
                    {selectedModel && (
                        <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm overflow-hidden">
                            <div className="px-6 py-5 border-b dark:border-[#2d3f50] border-slate-200 flex items-center justify-between">
                                <div>
                                    <h2 className="dark:text-white text-slate-900 text-lg font-bold flex items-center gap-2">
                                        <span className="material-symbols-outlined text-[#137fec] text-[20px]">history</span>
                                        Versions — {selectedModel.model_class || selectedModel.name || "Model"}
                                    </h2>
                                    <p className="text-xs dark:text-slate-400 text-slate-500 mt-0.5 font-mono">
                                        {selectedModel.id}
                                    </p>
                                </div>
                                <button
                                    onClick={() => { setSelectedModel(null); setVersions([]); }}
                                    className="text-slate-400 hover:text-white transition-colors p-1 rounded hover:bg-slate-700"
                                >
                                    <span className="material-symbols-outlined text-[20px]">close</span>
                                </button>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="w-full text-left border-collapse">
                                    <thead>
                                        <tr className="dark:bg-[#101922]/50 bg-slate-50 border-b dark:border-[#2d3f50] border-slate-200">
                                            <th className="px-5 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Version</th>
                                            <th className="px-5 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Accuracy</th>
                                            <th className="px-5 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">F1 Score</th>
                                            <th className="px-5 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Drift</th>
                                            <th className="px-5 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Ensemble</th>
                                            <th className="px-5 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Created</th>
                                            <th className="px-5 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider text-right">Action</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y dark:divide-[#2d3f50] divide-slate-100">
                                        {versionLoading && (
                                            <tr>
                                                <td colSpan={7} className="px-5 py-10 text-center text-slate-500">
                                                    <span className="material-symbols-outlined text-xl animate-spin">sync</span>
                                                    <p className="mt-2 text-sm">Loading versions…</p>
                                                </td>
                                            </tr>
                                        )}
                                        {!versionLoading && versions.length === 0 && (
                                            <tr>
                                                <td colSpan={7} className="px-5 py-10 text-center text-slate-500 text-sm">
                                                    No versions found
                                                </td>
                                            </tr>
                                        )}
                                        {versions.map((ver) => {
                                            const isCurrent =
                                                selectedModel.current_version === ver.version_number;
                                            return (
                                                <tr
                                                    key={ver.version_id || ver.version_number}
                                                    className={`transition-colors ${isCurrent
                                                        ? "bg-emerald-500/5 dark:bg-emerald-500/5"
                                                        : "hover:bg-slate-50 dark:hover:bg-white/5"
                                                        }`}
                                                >
                                                    <td className="px-5 py-3.5 font-mono text-sm dark:text-white text-slate-900">
                                                        <div className="flex items-center gap-2">
                                                            {ver.version_number}
                                                            {isCurrent && (
                                                                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                                                                    CURRENT
                                                                </span>
                                                            )}
                                                        </div>
                                                    </td>
                                                    <td className="px-5 py-3.5 text-sm dark:text-slate-300 text-slate-600">
                                                        {ver.metrics?.accuracy != null
                                                            ? `${(ver.metrics.accuracy * 100).toFixed(2)}%`
                                                            : "—"}
                                                    </td>
                                                    <td className="px-5 py-3.5 text-sm dark:text-slate-300 text-slate-600">
                                                        {ver.metrics?.f1_score != null
                                                            ? ver.metrics.f1_score.toFixed(4)
                                                            : "—"}
                                                    </td>
                                                    <td className="px-5 py-3.5 text-sm dark:text-slate-300 text-slate-600">
                                                        {ver.metrics?.drift_score ?? "—"}
                                                    </td>
                                                    <td className="px-5 py-3.5 text-sm dark:text-slate-300 text-slate-600">
                                                        {ver.metrics?.ensemble_used != null
                                                            ? ver.metrics.ensemble_used
                                                                ? "Yes"
                                                                : "No"
                                                            : "—"}
                                                    </td>
                                                    <td className="px-5 py-3.5 text-xs dark:text-slate-400 text-slate-500">
                                                        {fmtDate(ver.created_at)}
                                                    </td>
                                                    <td className="px-5 py-3.5 text-right">
                                                        {!isCurrent && (
                                                            <button
                                                                onClick={() => handlePromote(ver.version_number)}
                                                                disabled={promoting}
                                                                className="bg-[#137fec] hover:bg-[#0f66bd] disabled:opacity-50 text-white text-xs font-medium px-3 py-1.5 rounded-lg transition-colors shadow-sm"
                                                            >
                                                                {promoting ? "…" : "Promote"}
                                                            </button>
                                                        )}
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
}

/* ── Stat card ── */
function StatCard({ label, value, icon, iconClass }) {
    return (
        <div className="dark:bg-[#1a2632] bg-white p-5 rounded-xl border dark:border-[#2d3f50] border-slate-200 shadow-sm">
            <div className="flex justify-between items-start">
                <div>
                    <p className="text-xs font-medium dark:text-slate-400 text-slate-500 uppercase tracking-wider">{label}</p>
                    <h3 className="text-2xl font-bold dark:text-white text-slate-900 mt-1">{value}</h3>
                </div>
                <span className={`material-symbols-outlined ${iconClass} p-2 rounded-lg`}>{icon}</span>
            </div>
        </div>
    );
}
