import { useState, useEffect, useRef } from "react";
import { uploadDataset, getDatasets, deleteDataset } from "../api/datasets";

/* ── helpers ──────────────────────────────────────────────── */
function fmtDate(iso) {
    if (!iso) return "—";
    return new Date(iso).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
    });
}

function colCount(schema) {
    if (Array.isArray(schema)) return schema.length;
    if (typeof schema === "string") {
        try { return JSON.parse(schema).length; } catch { return "—"; }
    }
    return "—";
}

/* ══════════════════════════════════════════════════════════
   DATASET MANAGEMENT PAGE
   ══════════════════════════════════════════════════════════ */
export default function DatasetUpload() {
    /* ── state ── */
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const fileInputRef = useRef(null);

    /* ── load datasets on mount ── */
    useEffect(() => { loadDatasets(); }, []);

    const loadDatasets = async () => {
        try {
            const data = await getDatasets();
            setDatasets(data);
        } catch (err) {
            console.error("Failed to load datasets:", err);
        } finally {
            setLoading(false);
        }
    };

    /* ── upload handler ── */
    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        setError(null);
        setSuccess(null);
        try {
            const result = await uploadDataset(file);
            setSuccess(`Uploaded "${result.name}" — ${result.rows} rows`);
            setFile(null);
            if (fileInputRef.current) fileInputRef.current.value = "";
            await loadDatasets();
        } catch (err) {
            setError(err.response?.data?.detail || "Upload failed. Check if the file is a valid CSV.");
        } finally {
            setUploading(false);
        }
    };

    /* ── drag & drop ── */
    const onDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        const dropped = e.dataTransfer.files?.[0];
        if (dropped) setFile(dropped);
    };

    /* ── delete handler ── */
    const handleDeleteDataset = async (datasetId) => {
        if (!window.confirm("Are you sure you want to delete this dataset?")) {
            return;
        }
        try {
            await deleteDataset(datasetId);
            setSuccess("Dataset deleted successfully.");
            await loadDatasets();
        } catch (err) {
            console.error("Delete failed:", err);
            setError(err.response?.data?.detail || "Failed to delete dataset. Models might still be using it.");
        }
    };

    /* ── derived stats ── */
    const totalDatasets = datasets.length;

    return (
        <>
            {/* ── Header ── */}
            <header className="flex items-center justify-between px-6 py-3 border-b border-slate-200 dark:border-[#2d3f50] bg-white dark:bg-[#1a2632] shrink-0 z-10">
                <div className="flex flex-col gap-0.5">
                    <h1 className="text-xl font-bold tracking-tight dark:text-white text-slate-900">
                        Dataset Management
                    </h1>
                    <p className="text-xs dark:text-slate-400 text-slate-500">
                        Upload and manage CSV training data
                    </p>
                </div>
            </header>

            {/* ── Content ── */}
            <div className="flex-1 overflow-y-auto p-6 lg:p-10">
                <div className="max-w-7xl mx-auto flex flex-col gap-8">

                    {/* ── Stat cards ── */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <StatCard
                            label="Total Datasets"
                            value={loading ? "…" : totalDatasets}
                            icon="database"
                            iconClass="dark:text-[#137fec] text-[#137fec]/80 bg-[#137fec]/10"
                        />
                        <StatCard
                            label="Total Columns"
                            value={loading ? "…" : datasets.reduce((sum, d) => {
                                const c = colCount(d.schema);
                                return sum + (typeof c === "number" ? c : 0);
                            }, 0)}
                            icon="view_column"
                            iconClass="dark:text-emerald-400 text-emerald-600 bg-emerald-500/10"
                        />
                        <StatCard
                            label="With Stats"
                            value={loading ? "…" : datasets.filter((d) => d.baseline_stats && Object.keys(d.baseline_stats).length > 0).length}
                            icon="analytics"
                            iconClass="dark:text-amber-400 text-amber-600 bg-amber-500/10"
                        />
                    </div>

                    {/* ── Main grid: Upload + Table ── */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
                        {/* Upload panel */}
                        <section className="lg:col-span-1 flex flex-col gap-4">
                            <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl p-6 shadow-sm">
                                <h3 className="dark:text-white text-slate-900 text-lg font-bold mb-4 flex items-center gap-2">
                                    <span className="material-symbols-outlined text-[#137fec]">cloud_upload</span>
                                    Upload Dataset
                                </h3>

                                {/* Drop zone */}
                                <div
                                    className={`group flex flex-col items-center justify-center w-full h-56 border-2 border-dashed rounded-lg transition-all cursor-pointer relative overflow-hidden
                    ${dragOver
                                            ? "border-[#137fec] bg-[#137fec]/5"
                                            : "dark:border-[#2d3f50] border-slate-300 dark:bg-[#101922]/50 bg-slate-50/50 hover:border-[#137fec] dark:hover:border-[#137fec]"
                                        }`}
                                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                                    onDragLeave={() => setDragOver(false)}
                                    onDrop={onDrop}
                                >
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept=".csv"
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    />
                                    <div className="flex flex-col items-center justify-center text-center px-4">
                                        <div className="mb-3 p-3 rounded-full bg-slate-100 dark:bg-[#1a2632] text-slate-400 dark:text-slate-500 group-hover:text-[#137fec] group-hover:bg-[#137fec]/10 transition-colors">
                                            <span className="material-symbols-outlined text-4xl">upload_file</span>
                                        </div>
                                        {file ? (
                                            <p className="text-sm font-medium dark:text-white text-slate-900">
                                                {file.name}
                                                <span className="block text-xs text-slate-400 mt-0.5">
                                                    {(file.size / 1024).toFixed(1)} KB
                                                </span>
                                            </p>
                                        ) : (
                                            <>
                                                <p className="mb-1 text-sm font-medium dark:text-white text-slate-900">
                                                    <span className="font-bold text-[#137fec]">Click to upload</span> or drag and drop
                                                </p>
                                                <p className="text-xs dark:text-slate-400 text-slate-500">CSV files (MAX. 500MB)</p>
                                            </>
                                        )}
                                    </div>
                                </div>

                                {/* Messages */}
                                {error && (
                                    <div className="mt-3 p-3 rounded-lg bg-red-500/10 text-red-400 text-xs border border-red-500/20">
                                        {error}
                                    </div>
                                )}
                                {success && (
                                    <div className="mt-3 p-3 rounded-lg bg-emerald-500/10 text-emerald-400 text-xs border border-emerald-500/20">
                                        {success}
                                    </div>
                                )}

                                {/* Upload button */}
                                <button
                                    onClick={handleUpload}
                                    disabled={!file || uploading}
                                    className="w-full mt-4 flex items-center justify-center gap-2 bg-[#137fec] hover:bg-[#0f66bd] disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-2.5 px-4 rounded-lg transition-colors shadow-lg shadow-[#137fec]/20"
                                >
                                    {uploading ? (
                                        <>
                                            <span className="material-symbols-outlined text-[18px] animate-spin">sync</span>
                                            Uploading…
                                        </>
                                    ) : (
                                        <>
                                            <span className="material-symbols-outlined text-[18px]">cloud_upload</span>
                                            Start Upload
                                        </>
                                    )}
                                </button>
                            </div>

                            {/* Tips */}
                            <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl p-6 shadow-sm">
                                <h4 className="dark:text-white text-slate-900 text-sm font-bold mb-3">Quick Tips</h4>
                                <ul className="text-xs dark:text-slate-400 text-slate-500 space-y-2 list-disc list-inside">
                                    <li>Ensure your CSV has headers in the first row.</li>
                                    <li>Date columns should be ISO 8601 formatted.</li>
                                    <li>Remove empty rows to speed up processing.</li>
                                </ul>
                            </div>
                        </section>

                        {/* Datasets table */}
                        <section className="lg:col-span-2 flex flex-col gap-4">
                            <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm flex flex-col">
                                {/* Table header */}
                                <div className="px-6 py-5 border-b dark:border-[#2d3f50] border-slate-200 flex items-center justify-between">
                                    <h3 className="dark:text-white text-slate-900 text-lg font-bold">Registered Datasets</h3>
                                    <span className="text-xs text-slate-400">{totalDatasets} total</span>
                                </div>

                                {/* Table */}
                                <div className="overflow-x-auto">
                                    <table className="w-full text-left border-collapse">
                                        <thead>
                                            <tr className="dark:bg-[#101922]/50 bg-slate-50 border-b dark:border-[#2d3f50] border-slate-200">
                                                <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">ID</th>
                                                <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider">Dataset Name</th>
                                                <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider text-right">Columns</th>
                                                <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider text-right">Stats Keys</th>
                                                <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider text-right">Created</th>
                                                <th className="px-6 py-3 text-xs font-semibold dark:text-slate-400 text-slate-500 uppercase tracking-wider text-right">Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y dark:divide-[#2d3f50] divide-slate-100">
                                            {loading && (
                                                <tr>
                                                    <td colSpan={5} className="px-6 py-12 text-center text-slate-500">
                                                        <span className="material-symbols-outlined text-2xl animate-spin">sync</span>
                                                        <p className="mt-2 text-sm">Loading datasets…</p>
                                                    </td>
                                                </tr>
                                            )}
                                            {!loading && datasets.length === 0 && (
                                                <tr>
                                                    <td colSpan={5} className="px-6 py-12 text-center text-slate-500">
                                                        <span className="material-symbols-outlined text-3xl mb-2 block">folder_open</span>
                                                        <p className="text-sm">No datasets uploaded yet</p>
                                                    </td>
                                                </tr>
                                            )}
                                            {datasets.map((ds) => (
                                                <tr
                                                    key={ds.id}
                                                    className="group hover:bg-slate-50 dark:hover:bg-white/5 transition-colors"
                                                >
                                                    <td className="px-6 py-4 text-xs font-medium dark:text-[#137fec] text-[#137fec] font-mono">
                                                        {String(ds.id).slice(0, 8)}
                                                    </td>
                                                    <td className="px-6 py-4">
                                                        <div className="flex items-center gap-3">
                                                            <div className="p-1.5 rounded bg-blue-500/10 text-blue-500">
                                                                <span className="material-symbols-outlined text-[18px] block">description</span>
                                                            </div>
                                                            <div>
                                                                <div className="text-sm font-medium dark:text-white text-slate-900">
                                                                    {ds.name}
                                                                </div>
                                                                {ds.minio_path && (
                                                                    <div className="text-xs dark:text-slate-400 text-slate-500">
                                                                        {ds.minio_path}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </td>
                                                    <td className="px-6 py-4 text-sm dark:text-slate-400 text-slate-600 text-right font-mono">
                                                        {colCount(ds.schema)}
                                                    </td>
                                                    <td className="px-6 py-4 text-sm dark:text-slate-400 text-slate-600 text-right font-mono">
                                                        {ds.baseline_stats ? Object.keys(ds.baseline_stats).length : "—"}
                                                    </td>
                                                    <td className="px-6 py-4 text-xs dark:text-slate-400 text-slate-500 text-right">
                                                        {fmtDate(ds.created_at)}
                                                    </td>
                                                    <td className="px-6 py-4 text-right">
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); handleDeleteDataset(ds.id); }}
                                                            className="text-xs font-medium text-red-500 hover:text-red-400 transition-colors px-2 py-1 rounded hover:bg-red-500/10"
                                                        >
                                                            Delete
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>

                                {/* Footer */}
                                {datasets.length > 0 && (
                                    <div className="flex items-center justify-between px-6 py-4 border-t dark:border-[#2d3f50] border-slate-200">
                                        <div className="text-xs dark:text-slate-400 text-slate-500">
                                            Showing <span className="font-medium dark:text-white text-slate-900">{datasets.length}</span> datasets
                                        </div>
                                    </div>
                                )}
                            </div>
                        </section>
                    </div>
                </div>
            </div>
        </>
    );
}

/* ── Stat card component ── */
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
