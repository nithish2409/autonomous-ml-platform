import { useState, useEffect, useRef } from "react";
import { getTrainingJobs, getTrainingJobStatus, getTrainingJobLogs, startTraining, deleteTrainingJob } from "../api/training";
import { getDatasets } from "../api/datasets";
import { registerModel } from "../api/models";

/* ── helpers ──────────────────────────────────────────────── */
function fmtDate(iso) {
    if (!iso) return "—";
    return new Date(iso).toLocaleDateString("en-US", {
        month: "short", day: "numeric", year: "numeric",
    });
}
function fmtTime(iso) {
    if (!iso) return "";
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/* ── Status badge ── */
function StatusBadge({ status }) {
    const map = {
        running: { cls: "bg-blue-500/10 text-blue-500 border-blue-500/20", dot: "bg-blue-500 animate-pulse", icon: null },
        completed: { cls: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20", dot: "bg-emerald-500", icon: null },
        failed: { cls: "bg-red-500/10 text-red-500 border-red-500/20", dot: "bg-red-500", icon: null },
        pending: { cls: "bg-amber-500/10 text-amber-500 border-amber-500/20", dot: "bg-amber-400", icon: null },
        paused: { cls: "bg-slate-500/10 text-slate-400 border-slate-500/20", dot: "bg-slate-400", icon: null },
    };
    const s = map[status] || map.pending;
    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${s.cls}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
            {status}
        </span>
    );
}

/* ── Progress bar ── */
function ProgressBar({ value, color = "bg-primary", height = "h-2" }) {
    const pct = Math.min(Math.max((value || 0) * 100, 0), 100);
    return (
        <div className={`w-full ${height} rounded-full bg-slate-700/50 overflow-hidden`}>
            <div className={`${height} rounded-full ${color} transition-all duration-500`} style={{ width: `${pct}%` }} />
        </div>
    );
}

/* ── Metric mini card ── */
function MetricCard({ label, value, icon, iconClass }) {
    return (
        <div className="dark:bg-background-dark bg-slate-50 p-4 rounded-lg border dark:border-[#2d3f50] border-slate-200">
            <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium dark:text-slate-400 text-slate-500 uppercase tracking-wider">{label}</span>
                <span className={`material-symbols-outlined text-[16px] ${iconClass}`}>{icon}</span>
            </div>
            <p className="text-lg font-bold dark:text-white text-slate-900">{value}</p>
        </div>
    );
}

/* ── New Training Job Modal ── */
function NewTrainingModal({ onClose, onStart }) {
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [form, setForm] = useState({ datasetId: "", algorithm: "RandomForestClassifier", targetColumn: "" });
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState(null);

    // Algorithm catalogue — key is the model_class sent to the backend
    const ALGORITHMS = [
        { group: "scikit-learn — Ensemble", options: [
            { value: "RandomForestClassifier", label: "Random Forest", framework: "sklearn" },
            { value: "GradientBoostingClassifier", label: "Gradient Boosting", framework: "sklearn" },
            { value: "AdaBoostClassifier", label: "AdaBoost", framework: "sklearn" },
            { value: "ExtraTreesClassifier", label: "Extra Trees", framework: "sklearn" },
        ]},
        { group: "scikit-learn — Linear", options: [
            { value: "LogisticRegression", label: "Logistic Regression", framework: "sklearn" },
        ]},
        { group: "scikit-learn — Tree", options: [
            { value: "DecisionTreeClassifier", label: "Decision Tree", framework: "sklearn" },
        ]},
        { group: "scikit-learn — SVM", options: [
            { value: "SVC", label: "Support Vector Machine (SVC)", framework: "sklearn" },
        ]},
        { group: "scikit-learn — Neighbors", options: [
            { value: "KNeighborsClassifier", label: "K-Nearest Neighbors", framework: "sklearn" },
        ]},
        { group: "XGBoost", options: [
            { value: "XGBClassifier", label: "XGBoost", framework: "xgboost" },
        ]},
        { group: "LightGBM", options: [
            { value: "LGBMClassifier", label: "LightGBM", framework: "lightgbm" },
        ]},
    ];

    // Derive framework from selected algorithm
    const getFramework = (algoValue) => {
        for (const group of ALGORITHMS) {
            const found = group.options.find(o => o.value === algoValue);
            if (found) return found.framework;
        }
        return "sklearn";
    };

    useEffect(() => {
        getDatasets()
            .then(data => setDatasets(data))
            .catch(err => setError("Failed to load datasets"))
            .finally(() => setLoading(false));
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!form.datasetId) return setError("Please select a dataset.");
        setSubmitting(true);
        setError(null);
        try {
            await onStart({
                datasetId: form.datasetId,
                framework: getFramework(form.algorithm),
                targetColumn: form.targetColumn || null,
                modelClass: form.algorithm
            });
            onClose();
        } catch (err) {
            setError(err.response?.data?.detail || err.message || "Failed to start training.");
            setSubmitting(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm">
            <div className="bg-white dark:bg-[#1a2632] border border-slate-200 dark:border-[#2d3f50] rounded-xl shadow-2xl w-full max-w-md overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-200 dark:border-[#2d3f50] flex items-center justify-between">
                    <h3 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
                        <span className="material-symbols-outlined text-primary">rocket_launch</span>
                        New Training Job
                    </h3>
                    <button onClick={onClose} className="text-slate-400 hover:text-slate-600 dark:hover:text-white transition-colors">
                        <span className="material-symbols-outlined">close</span>
                    </button>
                </div>
                <form onSubmit={handleSubmit} className="p-6 flex flex-col gap-4">
                    {error && (
                        <div className="p-3 rounded-lg bg-red-500/10 text-red-500 text-sm border border-red-500/20">
                            {error}
                        </div>
                    )}
                    <div>
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Dataset</label>
                        <select
                            value={form.datasetId}
                            onChange={(e) => setForm({ ...form, datasetId: e.target.value })}
                            className="w-full bg-slate-50 dark:bg-background-dark border border-slate-200 dark:border-[#2d3f50] text-slate-900 dark:text-white text-sm rounded-lg focus:ring-primary focus:border-primary block p-2.5 outline-none"
                            disabled={loading || submitting}
                        >
                            <option value="">{loading ? "Loading datasets..." : "Select a dataset"}</option>
                            {datasets.map(ds => (
                                <option key={ds.id} value={ds.id}>{ds.name} ({String(ds.id).slice(0, 8)})</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Algorithm</label>
                        <select
                            value={form.algorithm}
                            onChange={(e) => setForm({ ...form, algorithm: e.target.value })}
                            className="w-full bg-slate-50 dark:bg-background-dark border border-slate-200 dark:border-[#2d3f50] text-slate-900 dark:text-white text-sm rounded-lg focus:ring-primary focus:border-primary block p-2.5 outline-none"
                            disabled={submitting}
                        >
                            {ALGORITHMS.map(group => (
                                <optgroup key={group.group} label={group.group}>
                                    {group.options.map(opt => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </optgroup>
                            ))}
                        </select>
                        <p className="mt-1 text-xs text-slate-400">
                            Framework: <span className="font-medium text-slate-500 dark:text-slate-300">{getFramework(form.algorithm)}</span>
                        </p>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                            Target Column <span className="text-xs text-slate-400 font-normal">(optional, auto-detected)</span>
                        </label>
                        <input
                            type="text"
                            value={form.targetColumn}
                            onChange={(e) => setForm({ ...form, targetColumn: e.target.value })}
                            placeholder="Leave blank for auto-detect"
                            className="w-full bg-slate-50 dark:bg-background-dark border border-slate-200 dark:border-[#2d3f50] text-slate-900 dark:text-white text-sm rounded-lg focus:ring-primary focus:border-primary block p-2.5 outline-none"
                            disabled={submitting}
                        />
                    </div>

                    <div className="pt-2 mt-2 border-t border-slate-200 dark:border-[#2d3f50] flex justify-end gap-3">
                        <button
                            type="button"
                            onClick={onClose}
                            disabled={submitting}
                            className="px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-[#2d3f50] rounded-lg transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={submitting}
                            className="px-4 py-2 text-sm font-medium text-white bg-primary hover:bg-[#0f66bd] rounded-lg transition-colors shadow-lg shadow-primary/20 flex items-center gap-2"
                        >
                            {submitting ? <span className="material-symbols-outlined text-[18px] animate-spin">sync</span> : null}
                            {submitting ? "Starting..." : "Start Training"}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}

/* ══════════════════════════════════════════════════════════
   TRAINING PAGE
   ══════════════════════════════════════════════════════════ */
export default function Training() {
    const [jobs, setJobs] = useState([]);
    const [loadingJobs, setLoadingJobs] = useState(true);
    const [selectedJob, setSelectedJob] = useState(null);
    const [jobDetails, setJobDetails] = useState(null);
    const [jobLogs, setJobLogs] = useState([]);
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [showNewJobModal, setShowNewJobModal] = useState(false);
    const logEndRef = useRef(null);

    /* ── load jobs on mount ── */
    useEffect(() => { loadJobs(); }, []);

    const loadJobs = async () => {
        try {
            const data = await getTrainingJobs();
            setJobs(data);
        } catch (err) {
            console.error("Failed to load training jobs:", err);
        } finally {
            setLoadingJobs(false);
        }
    };

    /* ── start new job ── */
    const handleStartNewJob = async (payload) => {
        // 1. Register the model using the dataset
        const regRes = await registerModel({
            dataset_id: payload.datasetId,
            framework: payload.framework,
            model_class: payload.modelClass
        });

        // 2. Start training using the generated model_id
        await startTraining({
            model_id: regRes.model_id,
            target_column: payload.targetColumn,
            hyperparameters: null,
            split_ratio: 0.2,
            random_seed: 42
        });

        // 3. Reload list
        await loadJobs();
    };

    /* ── select job → fetch details + logs ── */
    const handleSelectJob = async (job) => {
        setSelectedJob(job);
        setJobDetails(null);
        setJobLogs([]);
        setLoadingDetails(true);
        try {
            const [detailRes, logsRes] = await Promise.allSettled([
                getTrainingJobStatus(job.id),
                getTrainingJobLogs(job.id),
            ]);
            if (detailRes.status === "fulfilled") setJobDetails(detailRes.value);
            if (logsRes.status === "fulfilled") setJobLogs(logsRes.value?.logs || []);
        } catch (err) {
            console.error("Failed to load job details:", err);
        } finally {
            setLoadingDetails(false);
        }
    };

    /* ── delete job ── */
    const handleDeleteJob = async (jobId) => {
        if (!window.confirm("Are you sure you want to delete this training job and its logs?")) return;
        try {
            await deleteTrainingJob(jobId);
            if (selectedJob?.id === jobId) {
                setSelectedJob(null);
                setJobDetails(null);
                setJobLogs([]);
            }
            await loadJobs();
        } catch (err) {
            console.error("Failed to delete job:", err);
            alert("Failed to delete job.");
        }
    };

    /* auto-scroll logs */
    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [jobLogs]);

    /* ── derived stats ── */
    const runningCount = jobs.filter((j) => j.status === "running").length;
    const completedCount = jobs.filter((j) => j.status === "completed").length;
    const failedCount = jobs.filter((j) => j.status === "failed").length;

    return (
        <>
            {/* ── Header ── */}
            <header className="flex items-center justify-between px-6 py-3 border-b border-slate-200 dark:border-[#2d3f50] bg-white dark:bg-[#1a2632] shrink-0 z-10">
                <div className="flex flex-col gap-0.5">
                    <h1 className="text-xl font-bold tracking-tight dark:text-white text-slate-900">
                        Training Jobs
                    </h1>
                    <p className="text-xs dark:text-slate-400 text-slate-500">
                        Monitor and inspect training pipelines
                    </p>
                </div>
                <button
                    onClick={() => setShowNewJobModal(true)}
                    className="flex items-center gap-2 bg-primary hover:bg-[#0f66bd] text-white text-sm font-semibold py-2 px-4 rounded-lg transition-colors shadow-lg shadow-primary/20"
                >
                    <span className="material-symbols-outlined text-[18px]">add</span>
                    New Training Job
                </button>
            </header>

            {showNewJobModal && (
                <NewTrainingModal
                    onClose={() => setShowNewJobModal(false)}
                    onStart={handleStartNewJob}
                />
            )}

            {/* ── Content ── */}
            <div className="flex-1 overflow-y-auto p-6 lg:p-10">
                <div className="max-w-7xl mx-auto flex flex-col gap-6">

                    {/* ── Stat cards ── */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <StatCard label="Total Jobs" value={loadingJobs ? "…" : jobs.length}
                            icon="work" iconClass="dark:text-primary text-primary/80 bg-primary/10" />
                        <StatCard label="Running" value={loadingJobs ? "…" : runningCount}
                            icon="play_circle" iconClass="dark:text-blue-400 text-blue-600 bg-blue-500/10" />
                        <StatCard label="Completed" value={loadingJobs ? "…" : completedCount}
                            icon="check_circle" iconClass="dark:text-emerald-400 text-emerald-600 bg-emerald-500/10" />
                        <StatCard label="Failed" value={loadingJobs ? "…" : failedCount}
                            icon="error" iconClass="dark:text-red-400 text-red-600 bg-red-500/10" />
                    </div>

                    {/* ── Main grid: Job list + Detail panel ── */}
                    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 items-start">

                        {/* Job list — 2 cols */}
                        <div className="lg:col-span-2 dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm overflow-hidden">
                            <div className="px-6 py-4 border-b dark:border-[#2d3f50] border-slate-200">
                                <h2 className="dark:text-white text-slate-900 text-base font-bold">Jobs</h2>
                            </div>
                            <div className="divide-y dark:divide-[#2d3f50] divide-slate-100 max-h-[600px] overflow-y-auto">
                                {loadingJobs && (
                                    <div className="px-6 py-12 text-center text-slate-500">
                                        <span className="material-symbols-outlined text-2xl animate-spin">sync</span>
                                        <p className="mt-2 text-sm">Loading jobs…</p>
                                    </div>
                                )}
                                {!loadingJobs && jobs.length === 0 && (
                                    <div className="px-6 py-12 text-center text-slate-500">
                                        <span className="material-symbols-outlined text-3xl mb-2 block">work</span>
                                        <p className="text-sm">No training jobs yet</p>
                                    </div>
                                )}
                                {jobs.map((job) => (
                                    <button
                                        key={job.id}
                                        onClick={() => handleSelectJob(job)}
                                        className={`w-full text-left px-5 py-4 transition-colors ${selectedJob?.id === job.id
                                            ? "bg-primary/5 dark:bg-primary/10 border-l-2 border-primary"
                                            : "hover:bg-slate-50 dark:hover:bg-white/5 border-l-2 border-transparent"
                                            }`}
                                    >
                                        <div className="flex items-center justify-between mb-1.5">
                                            <span className="text-sm font-semibold dark:text-white text-slate-900 truncate">
                                                {job.model_name || job.model_class || "Unnamed Model"}
                                            </span>
                                            <StatusBadge status={job.status || "pending"} />
                                        </div>
                                        <div className="flex items-center gap-3 text-xs dark:text-slate-400 text-slate-500">
                                            <span className="font-mono">{String(job.id).slice(0, 8)}…</span>
                                            {job.framework && <span className="capitalize">{job.framework}</span>}
                                            <span>{fmtDate(job.created_at)}</span>
                                        </div>
                                        {job.result_metrics?.accuracy != null && (
                                            <div className="mt-1.5 text-xs">
                                                <span className="text-emerald-500 font-medium">
                                                    {(job.result_metrics.accuracy * 100).toFixed(1)}% accuracy
                                                </span>
                                            </div>
                                        )}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Detail panel — 3 cols */}
                        <div className="lg:col-span-3">
                            {!selectedJob ? (
                                <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-12 text-center">
                                    <span className="material-symbols-outlined text-4xl text-slate-500 mb-3 block">analytics</span>
                                    <p className="text-slate-400 text-sm">Select a training job to view details</p>
                                </div>
                            ) : loadingDetails ? (
                                <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-12 text-center">
                                    <span className="material-symbols-outlined text-2xl animate-spin text-slate-500">sync</span>
                                    <p className="mt-2 text-sm text-slate-400">Loading details…</p>
                                </div>
                            ) : (
                                <div className="flex flex-col gap-4">

                                    {/* Job header card */}
                                    <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-6">
                                        <div className="flex items-center justify-between mb-4">
                                            <div>
                                                <h2 className="text-lg font-bold dark:text-white text-slate-900">
                                                    {selectedJob.model_name || selectedJob.model_class || "Training Job"}
                                                </h2>
                                                <p className="text-xs dark:text-slate-400 text-slate-500 font-mono mt-0.5">
                                                    {selectedJob.id}
                                                </p>
                                            </div>
                                            <div className="flex items-center gap-3">
                                                <StatusBadge status={selectedJob.status || "pending"} />
                                                <button
                                                    onClick={() => handleDeleteJob(selectedJob.id)}
                                                    className="w-8 h-8 flex items-center justify-center rounded-lg text-red-500 hover:bg-red-500/10 transition-colors"
                                                    title="Delete Job"
                                                >
                                                    <span className="material-symbols-outlined text-[20px]">delete</span>
                                                </button>
                                            </div>
                                        </div>

                                        {/* Progress bar */}
                                        {jobDetails?.progress != null && (
                                            <div className="mb-4">
                                                <div className="flex items-center justify-between mb-1.5">
                                                    <span className="text-xs dark:text-slate-400 text-slate-500">
                                                        {jobDetails.epoch && jobDetails.total_epochs
                                                            ? `Epoch ${jobDetails.epoch} / ${jobDetails.total_epochs}`
                                                            : "Progress"}
                                                    </span>
                                                    <span className="text-xs font-medium dark:text-white text-slate-900">
                                                        {(jobDetails.progress * 100).toFixed(0)}%
                                                    </span>
                                                </div>
                                                <ProgressBar value={jobDetails.progress} />
                                            </div>
                                        )}

                                        {/* Info row */}
                                        <div className="flex flex-wrap gap-4 text-xs dark:text-slate-400 text-slate-500">
                                            {selectedJob.framework && (
                                                <span className="flex items-center gap-1">
                                                    <span className="material-symbols-outlined text-[14px]">code</span>
                                                    {selectedJob.framework}
                                                </span>
                                            )}
                                            <span className="flex items-center gap-1">
                                                <span className="material-symbols-outlined text-[14px]">schedule</span>
                                                {fmtDate(selectedJob.created_at)} {fmtTime(selectedJob.created_at)}
                                            </span>
                                            {selectedJob.config?.strategy && (
                                                <span className="flex items-center gap-1">
                                                    <span className="material-symbols-outlined text-[14px]">tune</span>
                                                    {selectedJob.config.strategy}
                                                </span>
                                            )}
                                            {selectedJob.config?.drift_type && (
                                                <span className="flex items-center gap-1">
                                                    <span className="material-symbols-outlined text-[14px]">trending_up</span>
                                                    {selectedJob.config.drift_type}
                                                </span>
                                            )}
                                        </div>
                                    </div>

                                    {/* Metrics cards */}
                                    {(selectedJob.result_metrics || jobDetails?.metrics) && (
                                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                            {(() => {
                                                const m = jobDetails?.metrics || selectedJob.result_metrics || {};
                                                return Object.entries(m).map(([key, val]) => (
                                                    <MetricCard
                                                        key={key}
                                                        label={key.replace(/_/g, " ")}
                                                        value={typeof val === "number" ? (val < 1 && val > 0 ? `${(val * 100).toFixed(2)}%` : val.toFixed(4)) : String(val)}
                                                        icon="analytics"
                                                        iconClass="text-primary"
                                                    />
                                                ));
                                            })()}
                                        </div>
                                    )}

                                    {/* Resources */}
                                    {jobDetails?.resources && (
                                        <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-5">
                                            <h3 className="text-sm font-bold dark:text-white text-slate-900 mb-3 flex items-center gap-2">
                                                <span className="material-symbols-outlined text-[16px] text-amber-400">memory</span>
                                                Resource Allocation
                                            </h3>
                                            <div className="space-y-3">
                                                {jobDetails.resources.gpu_utilization != null && (
                                                    <div>
                                                        <div className="flex justify-between text-xs mb-1">
                                                            <span className="dark:text-slate-400 text-slate-500">GPU Utilization</span>
                                                            <span className="dark:text-white text-slate-900 font-medium">{jobDetails.resources.gpu_utilization}%</span>
                                                        </div>
                                                        <ProgressBar value={jobDetails.resources.gpu_utilization / 100} color="bg-amber-500" />
                                                    </div>
                                                )}
                                                {jobDetails.resources.memory_used != null && (
                                                    <div>
                                                        <div className="flex justify-between text-xs mb-1">
                                                            <span className="dark:text-slate-400 text-slate-500">Memory</span>
                                                            <span className="dark:text-white text-slate-900 font-medium">
                                                                {jobDetails.resources.memory_used}GB / {jobDetails.resources.memory_total || "?"}GB
                                                            </span>
                                                        </div>
                                                        <ProgressBar value={jobDetails.resources.memory_used / (jobDetails.resources.memory_total || 32)} color="bg-indigo-500" />
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    {/* Hyperparameters */}
                                    {(selectedJob.config || jobDetails?.hyperparameters) && (
                                        <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm p-5">
                                            <h3 className="text-sm font-bold dark:text-white text-slate-900 mb-3 flex items-center gap-2">
                                                <span className="material-symbols-outlined text-[16px] text-indigo-400">tune</span>
                                                Configuration
                                            </h3>
                                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                                {Object.entries(jobDetails?.hyperparameters || selectedJob.config || {}).map(([key, val]) => (
                                                    <div key={key} className="dark:bg-background-dark bg-slate-50 p-3 rounded-lg border dark:border-[#2d3f50] border-slate-200">
                                                        <p className="text-xs dark:text-slate-400 text-slate-500 uppercase tracking-wider mb-0.5">
                                                            {key.replace(/_/g, " ")}
                                                        </p>
                                                        <p className="text-sm font-medium dark:text-white text-slate-900 font-mono truncate">
                                                            {typeof val === "object" ? JSON.stringify(val) : String(val)}
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Logs */}
                                    {jobLogs.length > 0 && (
                                        <div className="dark:bg-[#1a2632] bg-white border dark:border-[#2d3f50] border-slate-200 rounded-xl shadow-sm overflow-hidden">
                                            <div className="px-5 py-3 border-b dark:border-[#2d3f50] border-slate-200 flex items-center gap-2">
                                                <span className="material-symbols-outlined text-[16px] text-slate-400">terminal</span>
                                                <h3 className="text-sm font-bold dark:text-white text-slate-900">Training Logs</h3>
                                            </div>
                                            <div className="max-h-56 overflow-y-auto p-4 font-mono text-xs dark:bg-background-dark bg-slate-50 space-y-0.5">
                                                {jobLogs.map((line, i) => (
                                                    <div key={i} className="dark:text-slate-300 text-slate-600 leading-relaxed">
                                                        <span className="text-slate-500 select-none mr-3">{String(i + 1).padStart(3, "0")}</span>
                                                        {line}
                                                    </div>
                                                ))}
                                                <div ref={logEndRef} />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
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
