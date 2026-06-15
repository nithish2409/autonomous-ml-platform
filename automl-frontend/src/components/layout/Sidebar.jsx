import { NavLink } from "react-router-dom";

const sections = [
    {
        items: [
            { path: "/", label: "Dashboard", icon: "dashboard", filled: true },
        ],
    },
    {
        heading: "Resources",
        items: [
            { path: "/datasets", label: "Datasets", icon: "database" },
            { path: "/models", label: "Models", icon: "deployed_code" },
        ],
    },
    {
        heading: "Operations",
        items: [
            { path: "/training", label: "Training", icon: "memory" },
            { path: "/inference", label: "Inference", icon: "online_prediction" },
            { path: "/monitoring", label: "Monitoring", icon: "monitoring" },
            { path: "/automation", label: "Automation", icon: "account_tree" },
            { path: "/policies", label: "Policies", icon: "policy" },
        ],
    },
];


export default function Sidebar() {
    return (
        <aside className="w-64 flex flex-col border-r border-slate-200 dark:border-[#2a3b4d] bg-white dark:bg-[#111a22] shrink-0">
            {/* Logo */}
            <div className="h-16 flex items-center gap-3 px-6 border-b border-slate-200 dark:border-[#2a3b4d]">
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-[#137fec] to-blue-600 flex items-center justify-center text-white shadow-lg shadow-[#137fec]/20">
                    <span className="material-symbols-outlined text-[20px]">smart_toy</span>
                </div>
                <span className="font-semibold text-lg tracking-tight dark:text-white">Autonomous ML</span>
            </div>

            {/* Nav */}
            <nav className="flex-1 overflow-y-auto py-6 px-3 space-y-1">
                {sections.map((section, si) => (
                    <div key={si}>
                        {section.heading && (
                            <div className="pt-4 pb-2 px-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                {section.heading}
                            </div>
                        )}
                        {section.items.map((item) => (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                end={item.path === "/"}
                                className={({ isActive }) =>
                                    `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors group ${isActive
                                        ? "bg-[#137fec]/10 text-[#137fec] font-medium"
                                        : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200"
                                    }`
                                }
                            >
                                <span className={`material-symbols-outlined${item.filled ? " filled" : ""} group-hover:text-[#137fec] transition-colors`}>
                                    {item.icon}
                                </span>
                                <span>{item.label}</span>
                            </NavLink>
                        ))}
                    </div>
                ))}
            </nav>
        </aside>
    );
}
