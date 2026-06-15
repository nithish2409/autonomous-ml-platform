import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";

export default function Layout() {
    return (
        <div className="flex h-screen overflow-hidden bg-[#101922] text-slate-100 font-[Inter,sans-serif]">
            <Sidebar />
            <main className="flex-1 flex flex-col h-screen overflow-hidden relative">
                <Outlet />
            </main>
        </div>
    );
}
