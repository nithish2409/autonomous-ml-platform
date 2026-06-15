import { useState, useEffect, useCallback, useMemo } from "react";
import { getPolicies, updatePolicies, simulatePolicy } from "../api/policies";
import { getDecisionHistory } from "../api/automation";

/* ── Severity options ───────────────────────────────────── */
const SEVERITY_OPTIONS = ["low", "medium", "high", "critical"];

/* ── Deep equality check (for dirty tracking) ──────────── */
function deepEqual(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
}

/* ══════════════════════════════════════════════════════════
   POLICIES PAGE
   ══════════════════════════════════════════════════════════ */
export default function Policies() {
    const [saved, setSaved] = useState(null);       // last-saved config
    const [form, setForm] = useState(null);         // local editable
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState(null);
    const [toast, setToast] = useState(null);

    // simulation
    const [history, setHistory] = useState([]);
    const [simDecision, setSimDecision] = useState("");
    const [simLoading, setSimLoading] = useState(false);
    const [simResult, setSimResult] = useState(null);

    const isDirty = useMemo(
        () => saved && form && !deepEqual(saved, form),
        [saved, form]
    );

    /* ── Load policy + history on mount ── */
    useEffect(() => {
        (async () => {
            try {
                const [pol, hist] = await Promise.all([
                    getPolicies(),
                    getDecisionHistory(),
                ]);
                setSaved(pol);
                setForm(pol);
                setHistory(hist || []);
            } catch {
                setError("Failed to load policies");
            } finally {
                setLoading(false);
            }
        })();
    }, []);

    /* ── Save ── */
    const handleSave = useCallback(async () => {
        if (!form || !isDirty) return;
        setSaving(true);
        const previous = saved;
        setSaved(form); // optimistic
        try {
            const result = await updatePolicies(form);
            setSaved(result);
            setForm(result);
            showToast("Policies saved successfully");
        } catch {
            setSaved(previous); // rollback
            showToast("Failed to save policies", true);
        } finally {
            setSaving(false);
        }
    }, [form, isDirty, saved]);

    /* ── Reset ── */
    const handleReset = useCallback(() => {
        if (saved) setForm(saved);
    }, [saved]);

    /* ── Simulate ── */
    const handleSimulate = useCallback(async () => {
        if (!simDecision) return;
        setSimLoading(true);
        setSimResult(null);
        try {
            const res = await simulatePolicy(simDecision);
            setSimResult(res);
        } catch {
            showToast("Simulation failed", true);
        } finally {
            setSimLoading(false);
        }
    }, [simDecision]);

    /* ── Field updaters ── */
    const setAutoApproval = (key, value) =>
        setForm((f) => ({ ...f, auto_approval: { ...f.auto_approval, [key]: value } }));
    const setGuardrails = (key, value) =>
        setForm((f) => ({ ...f, guardrails: { ...f.guardrails, [key]: value } }));
    const setEscalation = (key, value) =>
        setForm((f) => ({ ...f, escalation: { ...f.escalation, [key]: value } }));

    const toggleSeverity = (sev) => {
        const current = form.auto_approval.allowed_severity || [];
        const next = current.includes(sev)
            ? current.filter((s) => s !== sev)
            : [...current, sev];
        setAutoApproval("allowed_severity", next);
    };

    const showToast = (msg, isError = false) => {
        setToast({ msg, isError });
        setTimeout(() => setToast(null), 3000);
    };

    /* ── Sim result color ── */
    const simColor = simResult?.simulation
        ? simResult.simulation.approved
            ? "border-emerald-500/30 bg-emerald-500/5"
            : simResult.simulation.blocked
                ? "border-red-500/30 bg-red-500/5"
                : "border-amber-500/30 bg-amber-500/5"
        : "";
    const simIcon = simResult?.simulation
        ? simResult.simulation.approved
            ? "check_circle"
            : simResult.simulation.blocked
                ? "block"
                : "warning"
        : "";
    const simIconColor = simResult?.simulation
        ? simResult.simulation.approved
            ? "text-emerald-400"
            : simResult.simulation.blocked
                ? "text-red-400"
                : "text-amber-400"
        : "";
    const simLabel = simResult?.simulation
        ? simResult.simulation.approved
            ? "Auto-Approved"
            : simResult.simulation.blocked
                ? "Blocked"
                : "Requires Human Review"
        : "";

    return (
        <>
            {/* Toast */}
            {toast && (
                <div className={`fixed top-4 right-4 z-50 px-4 py-2.5 rounded-lg shadow-lg text-sm font-medium border transition-all ${toast.isError
                        ? "bg-red-500/10 border-red-500/30 text-red-400"
                        : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                    }`}>
                    {toast.msg}
                </div>
            )}

            {/* Error banner */}
            {error && (
                <div className="bg-red-500/10 border-b border-red-500/30 text-red-400 px-6 py-2.5 text-sm flex items-center gap-2">
                    <span className="material-symbols-outlined text-[18px]">error</span>
                    {error}
                </div>
            )}

            {/* Header */}
            <header className="flex-none border-b border-slate-200 dark:border-surface-border bg-white dark:bg-[#111a22] px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="size-10 rounded-lg bg-purple-500/10 flex items-center justify-center text-purple-500">
                            <span className="material-symbols-outlined text-2xl">policy</span>
                        </div>
                        <div>
                            <h1 className="text-xl font-bold leading-tight tracking-tight">Policy Engine</h1>
                            <p className="text-slate-500 dark:text-slate-400 text-sm font-medium">
                                Configure auto-approval rules, guardrails, and escalation policies
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={handleReset}
                            disabled={!isDirty}
                            className="px-4 py-2 border border-slate-300 dark:border-surface-border rounded-lg text-sm font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-surface-dark disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                        >
                            Reset
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={!isDirty || saving}
                            className="px-5 py-2 bg-primary hover:bg-primary-dark disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold rounded-lg shadow-lg shadow-primary/20 flex items-center gap-2 transition-all text-sm"
                        >
                            {saving ? (
                                <span className="material-symbols-outlined text-lg animate-spin">sync</span>
                            ) : (
                                <span className="material-symbols-outlined text-lg">save</span>
                            )}
                            Save Policies
                        </button>
                    </div>
                </div>
            </header>

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-6">
                {loading ? (
                    <div className="flex items-center justify-center py-20">
                        <span className="material-symbols-outlined text-3xl animate-spin text-slate-500">sync</span>
                    </div>
                ) : form ? (
                    <div className="max-w-4xl mx-auto space-y-6">

                        {/* ── Section 1: Auto-Approval Rules ── */}
                        <SectionCard icon="auto_awesome" iconColor="text-primary" title="Auto-Approval Rules">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                                <NumberInput
                                    label="Minimum Confidence (%)"
                                    value={form.auto_approval.min_confidence}
                                    onChange={(v) => setAutoApproval("min_confidence", v)}
                                    min={0}
                                    max={100}
                                    unit="%"
                                />
                                <NumberInput
                                    label="Maximum Cost ($)"
                                    value={form.auto_approval.max_cost}
                                    onChange={(v) => setAutoApproval("max_cost", v)}
                                    min={0}
                                    step={10}
                                    unit="$"
                                />
                            </div>
                            <div className="mt-5">
                                <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-2 block">
                                    Allowed Severity Levels
                                </label>
                                <div className="flex flex-wrap gap-2">
                                    {SEVERITY_OPTIONS.map((sev) => {
                                        const active = form.auto_approval.allowed_severity?.includes(sev);
                                        return (
                                            <button
                                                key={sev}
                                                onClick={() => toggleSeverity(sev)}
                                                className={`px-4 py-1.5 rounded-full text-xs font-medium border capitalize transition-all ${active
                                                        ? sev === "low"
                                                            ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                                                            : sev === "medium"
                                                                ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                                                                : sev === "high"
                                                                    ? "bg-orange-500/10 border-orange-500/30 text-orange-400"
                                                                    : "bg-red-500/10 border-red-500/30 text-red-400"
                                                        : "border-slate-300 dark:border-surface-border text-slate-400 hover:border-slate-400"
                                                    }`}
                                            >
                                                {active && <span className="mr-1">✓</span>}
                                                {sev}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                            <div className="mt-5">
                                <ToggleRow
                                    label="Block Production Models"
                                    description="Require manual approval for changes to production models"
                                    checked={form.auto_approval.block_production}
                                    onChange={(v) => setAutoApproval("block_production", v)}
                                />
                            </div>
                        </SectionCard>

                        {/* ── Section 2: Guardrails ── */}
                        <SectionCard icon="shield" iconColor="text-amber-500" title="Guardrails">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                                <NumberInput
                                    label="Max GPU per Job"
                                    value={form.guardrails.max_gpu_per_job}
                                    onChange={(v) => setGuardrails("max_gpu_per_job", v)}
                                    min={1}
                                    max={64}
                                />
                                <NumberInput
                                    label="Max Daily Cost ($)"
                                    value={form.guardrails.max_daily_cost}
                                    onChange={(v) => setGuardrails("max_daily_cost", v)}
                                    min={0}
                                    step={100}
                                    unit="$"
                                />
                                <NumberInput
                                    label="Max Retrains / 24h"
                                    value={form.guardrails.max_retrains_24h}
                                    onChange={(v) => setGuardrails("max_retrains_24h", v)}
                                    min={1}
                                    max={100}
                                />
                            </div>
                            <div className="mt-5">
                                <ToggleRow
                                    label="Freeze Window"
                                    description="Block all automated actions (maintenance mode)"
                                    checked={form.guardrails.freeze_window}
                                    onChange={(v) => setGuardrails("freeze_window", v)}
                                    danger
                                />
                            </div>
                        </SectionCard>

                        {/* ── Section 3: Escalation ── */}
                        <SectionCard icon="notifications_active" iconColor="text-red-500" title="Escalation">
                            <ToggleRow
                                label="Notify on Critical"
                                description="Send alerts when critical-severity decisions are detected"
                                checked={form.escalation.notify_on_critical}
                                onChange={(v) => setEscalation("notify_on_critical", v)}
                            />
                            <div className="mt-5 grid grid-cols-1 gap-5">
                                <div>
                                    <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1.5 block">
                                        Webhook URL
                                    </label>
                                    <input
                                        type="url"
                                        value={form.escalation.webhook_url || ""}
                                        onChange={(e) => setEscalation("webhook_url", e.target.value || null)}
                                        placeholder="https://hooks.example.com/alerts"
                                        className="w-full bg-slate-50 dark:bg-[#111a22] border border-slate-200 dark:border-surface-border rounded-lg px-3.5 py-2 text-sm font-mono placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/40 transition-all"
                                    />
                                </div>
                                <div>
                                    <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1.5 block">
                                        Email Alerts (comma-separated)
                                    </label>
                                    <input
                                        type="text"
                                        value={(form.escalation.email_alerts || []).join(", ")}
                                        onChange={(e) =>
                                            setEscalation(
                                                "email_alerts",
                                                e.target.value
                                                    .split(",")
                                                    .map((s) => s.trim())
                                                    .filter(Boolean)
                                            )
                                        }
                                        placeholder="admin@company.com, devops@company.com"
                                        className="w-full bg-slate-50 dark:bg-[#111a22] border border-slate-200 dark:border-surface-border rounded-lg px-3.5 py-2 text-sm font-mono placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/40 transition-all"
                                    />
                                </div>
                            </div>
                        </SectionCard>

                        {/* ── Section 4: Policy Simulation ── */}
                        <SectionCard icon="science" iconColor="text-cyan-500" title="Policy Simulation">
                            <p className="text-sm text-slate-400 mb-4">
                                Select a past automation decision to simulate the current policy against it.
                            </p>
                            <div className="flex gap-3 items-end">
                                <div className="flex-1">
                                    <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1.5 block">
                                        Decision
                                    </label>
                                    <select
                                        value={simDecision}
                                        onChange={(e) => {
                                            setSimDecision(e.target.value);
                                            setSimResult(null);
                                        }}
                                        className="w-full bg-slate-50 dark:bg-[#111a22] border border-slate-200 dark:border-surface-border rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 transition-all"
                                    >
                                        <option value="">— Select a decision —</option>
                                        {history.map((d) => (
                                            <option key={d.id} value={d.id}>
                                                #{d.id?.slice(0, 6)} — {d.model_name} ({d.drift_type}, {d.severity})
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <button
                                    onClick={handleSimulate}
                                    disabled={!simDecision || simLoading}
                                    className="px-5 py-2 bg-cyan-600 hover:bg-cyan-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold rounded-lg shadow-lg shadow-cyan-600/20 flex items-center gap-2 transition-all text-sm whitespace-nowrap"
                                >
                                    {simLoading ? (
                                        <span className="material-symbols-outlined text-lg animate-spin">sync</span>
                                    ) : (
                                        <span className="material-symbols-outlined text-lg">play_arrow</span>
                                    )}
                                    Simulate
                                </button>
                            </div>

                            {/* Sim result */}
                            {simResult && (
                                <div className={`mt-5 p-5 rounded-lg border ${simColor}`}>
                                    <div className="flex items-center gap-3 mb-3">
                                        <span className={`material-symbols-outlined text-2xl ${simIconColor}`}>{simIcon}</span>
                                        <div>
                                            <p className={`text-sm font-bold ${simIconColor}`}>{simLabel}</p>
                                            {simResult.model_name && (
                                                <p className="text-xs text-slate-400">Model: {simResult.model_name}</p>
                                            )}
                                        </div>
                                    </div>
                                    <div className="bg-slate-50 dark:bg-[#0d131a] p-3 rounded-lg border border-slate-200 dark:border-slate-800">
                                        <p className="text-sm font-mono text-slate-300 leading-relaxed">
                                            {simResult.simulation?.reason}
                                        </p>
                                    </div>
                                    <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
                                        <SimFlag
                                            label="Auto-Approved"
                                            value={simResult.simulation?.approved}
                                        />
                                        <SimFlag
                                            label="Requires Human"
                                            value={simResult.simulation?.requires_human}
                                        />
                                        <SimFlag
                                            label="Blocked"
                                            value={simResult.simulation?.blocked}
                                        />
                                    </div>
                                </div>
                            )}
                        </SectionCard>
                    </div>
                ) : null}
            </div>
        </>
    );
}

/* ── Section card wrapper ── */
function SectionCard({ icon, iconColor, title, children }) {
    return (
        <section className="bg-white dark:bg-surface-dark rounded-lg border border-slate-200 dark:border-surface-border shadow-sm overflow-hidden">
            <div className="p-5 border-b border-slate-200 dark:border-surface-border flex items-center gap-3">
                <span className={`material-symbols-outlined text-xl ${iconColor}`}>{icon}</span>
                <h2 className="text-base font-bold">{title}</h2>
            </div>
            <div className="p-5">{children}</div>
        </section>
    );
}

/* ── Number input ── */
function NumberInput({ label, value, onChange, min, max, step = 1, unit }) {
    return (
        <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1.5 block">
                {label}
            </label>
            <div className="relative">
                {unit && (
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 font-mono">
                        {unit === "$" ? "$" : ""}
                    </span>
                )}
                <input
                    type="number"
                    value={value ?? ""}
                    onChange={(e) => {
                        const v = e.target.value === "" ? 0 : Number(e.target.value);
                        onChange(v);
                    }}
                    min={min}
                    max={max}
                    step={step}
                    className={`w-full bg-slate-50 dark:bg-[#111a22] border border-slate-200 dark:border-surface-border rounded-lg py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary/40 transition-all ${unit === "$" ? "pl-7 pr-3.5" : "px-3.5"
                        }`}
                />
                {unit === "%" && (
                    <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 font-mono">
                        %
                    </span>
                )}
            </div>
        </div>
    );
}

/* ── Toggle row ── */
function ToggleRow({ label, description, checked, onChange, danger }) {
    return (
        <div className="flex items-center justify-between py-2">
            <div>
                <p className={`text-sm font-medium ${danger ? "text-red-400" : ""}`}>{label}</p>
                {description && <p className="text-xs text-slate-400 mt-0.5">{description}</p>}
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
                <input
                    type="checkbox"
                    className="sr-only peer"
                    checked={checked}
                    onChange={(e) => onChange(e.target.checked)}
                />
                <div className={`w-11 h-6 rounded-full peer after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all ${danger
                        ? "bg-slate-300 dark:bg-slate-700 peer-checked:bg-red-500"
                        : "bg-slate-300 dark:bg-slate-700 peer-checked:bg-primary"
                    } peer-checked:after:translate-x-full peer-checked:after:border-white`} />
            </label>
        </div>
    );
}

/* ── Sim flag pill ── */
function SimFlag({ label, value }) {
    return (
        <div className={`text-center p-2 rounded border ${value
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                : "bg-slate-100 dark:bg-slate-800/30 border-slate-200 dark:border-slate-700 text-slate-400"
            }`}>
            <p className="font-bold text-sm">{value ? "YES" : "NO"}</p>
            <p className="text-[10px] uppercase tracking-wider mt-0.5">{label}</p>
        </div>
    );
}
