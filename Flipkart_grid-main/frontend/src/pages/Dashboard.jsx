import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Home,
  BarChart3,
  Target,
  Bot,
  Database,
  Camera,
  Truck,
  ChevronLeft,
  LogOut,
} from "lucide-react";
import { accentClass, cx } from "../lib/accents.js";
import Badge from "../components/ui/Badge.jsx";
import { useAuth } from "../context/AuthContext.jsx";
import { useFetch, endpoints } from "../lib/api.js";

import CommandCenter from "../tabs/CommandCenter.jsx";
import CongestionAnalytics from "../tabs/CongestionAnalytics.jsx";
import IntelligentDispatch from "../tabs/IntelligentDispatch.jsx";
import TacticalCommander from "../tabs/TacticalCommander.jsx";
import DataInspector from "../tabs/DataInspector.jsx";
import LiveCCTV from "../tabs/LiveCCTV.jsx";
import FleetOptimizer from "../tabs/FleetOptimizer.jsx";
import InteractionLogs from "../tabs/InteractionLogs.jsx";

// `roles` controls visibility. Users see a limited subset; admins see everything.
const TABS = [
  { id: "command", label: "Command Center", icon: Home, accent: "violet", Comp: CommandCenter, roles: ["user", "admin"] },
  { id: "analytics", label: "Congestion Analytics", icon: BarChart3, accent: "cyan", Comp: CongestionAnalytics, roles: ["user", "admin"] },
  { id: "dispatch", label: "Intelligent Dispatch", icon: Target, accent: "emerald", Comp: IntelligentDispatch, roles: ["admin"] },
  { id: "tactical", label: "Tactical AI Commander", icon: Bot, accent: "violet", Comp: TacticalCommander, roles: ["admin"] },
  { id: "data", label: "Data Inspector", icon: Database, accent: "cyan", Comp: DataInspector, roles: ["admin"] },
  { id: "cctv", label: "Live CCTV Vision", icon: Camera, accent: "rose", Comp: LiveCCTV, roles: ["user", "admin"] },
  { id: "fleet", label: "OR-Tools Fleet", icon: Truck, accent: "amber", Comp: FleetOptimizer, roles: ["admin"] },
  { id: "logs", label: "Interaction Logs", icon: Database, accent: "cyan", Comp: InteractionLogs, roles: ["admin"] },
];

export default function Dashboard() {
  const navigate = useNavigate();
  const { user, logout, role } = useAuth();

  // Require auth — bounce to the landing page (which hosts the login dialog).
  useEffect(() => {
    if (!user) navigate("/", { replace: true });
  }, [user, navigate]);

  // Tabs visible to the current role.
  const visibleTabs = useMemo(
    () => TABS.filter((t) => t.roles.includes(role ?? "user")),
    [role]
  );

  // Live KPI strip in the top bar — refetches automatically on dataset change.
  const { data: kpi } = useFetch(endpoints.telemetrySummary, []);

  // Honour ?tab=<id> deep-links from the landing bento grid.
  const [searchParams] = useSearchParams();
  const [active, setActive] = useState(() => searchParams.get("tab") || "command");
  const [collapsed, setCollapsed] = useState(false);

  // If the active tab isn't permitted for this role, fall back to the first one.
  useEffect(() => {
    if (visibleTabs.length && !visibleTabs.some((t) => t.id === active)) {
      setActive(visibleTabs[0].id);
    }
  }, [visibleTabs, active]);

  if (!user) return null;

  const activeTab = visibleTabs.find((t) => t.id === active) ?? visibleTabs[0];
  const ActiveComp = activeTab?.Comp ?? CommandCenter;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.35 }}
      className="flex min-h-screen"
    >
      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside
        className={cx(
          "sticky top-0 z-30 flex h-screen flex-col border-r border-subtle bg-elevated/80 backdrop-blur-xl transition-all duration-300",
          collapsed ? "w-[76px]" : "w-[260px]"
        )}
      >
        <div className="flex items-center gap-2.5 px-5 py-5">
          <button onClick={() => navigate("/")} className="grid h-9 w-9 shrink-0 place-items-center rounded-inner bg-gradient-to-br from-violet to-cyan font-display text-white shadow-glow-violet">
            P
          </button>
          {!collapsed && (
            <div className="leading-tight">
              <div className="font-display text-base text-ink-primary">
                SmartPark<span className="text-violet"> AI</span>
              </div>
              <div className="text-[0.62rem] uppercase tracking-wider text-ink-faint">
                {role === "admin" ? "Admin · all modules" : "Officer · limited access"}
              </div>
            </div>
          )}
        </div>

        <nav className="flex-1 space-y-1 px-3 py-2">
          {visibleTabs.map((t) => {
            const a = accentClass[t.accent];
            const on = active === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActive(t.id)}
                className={cx(
                  "group relative flex w-full items-center gap-3 rounded-inner px-3 py-2.5 text-sm font-medium transition-all",
                  on ? "text-ink-primary" : "text-ink-body hover:text-ink-primary hover:bg-white/[0.03]"
                )}
                style={on ? { background: `${a.hex}14` } : undefined}
              >
                {on && (
                  <motion.span
                    layoutId="side-active"
                    className="absolute left-0 top-1/2 h-7 w-[3px] -translate-y-1/2 rounded-r"
                    style={{ background: a.hex, boxShadow: `0 0 12px ${a.hex}` }}
                  />
                )}
                <t.icon
                  size={19}
                  className="shrink-0 transition-transform group-hover:scale-110"
                  style={{ color: on ? a.hex : undefined, filter: on ? `drop-shadow(0 0 6px ${a.hex})` : "none" }}
                />
                {!collapsed && <span className="truncate">{t.label}</span>}
              </button>
            );
          })}
        </nav>

        <div className="px-3 pb-4">
          {!collapsed && (
            <div className="mb-3 rounded-inner border border-subtle px-3 py-2.5">
              <div className="flex items-center justify-between">
                <span className="label-caps">System</span>
                <Badge accent="emerald" pulse>Online</Badge>
              </div>
              <p className="mt-1.5 font-mono text-[0.65rem] text-ink-faint">
                edge nodes · 38/38 healthy
              </p>
            </div>
          )}
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="flex w-full items-center justify-center gap-2 rounded-inner border border-subtle py-2 text-xs text-ink-body transition-colors hover:text-ink-primary"
          >
            <ChevronLeft size={15} className={cx("transition-transform", collapsed && "rotate-180")} />
            {!collapsed && "Collapse"}
          </button>
        </div>
      </aside>

      {/* ── Main content ────────────────────────────────────── */}
      <main className="relative flex-1 overflow-x-hidden">
        {/* Top bar */}
        <div className="sticky top-0 z-20 flex items-center justify-between border-b border-subtle bg-base/70 px-6 py-3.5 backdrop-blur-xl">
          <div>
            <h1 className="font-display text-lg text-ink-primary">
              {activeTab?.label ?? "Command Center"}
            </h1>
            <p className="text-xs text-ink-faint">Bengaluru Traffic Police · Live operations</p>
          </div>
          <div className="flex items-center gap-2.5">
            <Badge accent="rose" pulse>{kpi ? `${kpi.activeViolations.toLocaleString()} active` : "— active"}</Badge>
            <Badge accent={kpi && kpi.dataset !== "demo" ? "emerald" : "cyan"}>
              {kpi ? (kpi.dataset === "demo" ? "demo data" : "live: " + String(kpi.dataset).replace(/\.csv$/i, "").slice(0, 18)) : "…"}
            </Badge>
            <div className="hidden items-center gap-2 rounded-inner border border-subtle px-2.5 py-1.5 sm:flex">
              <div className="grid h-7 w-7 place-items-center rounded-full bg-gradient-to-br from-violet to-cyan text-[0.6rem] font-bold uppercase text-white">
                {user.username.slice(0, 2)}
              </div>
              <span className="text-xs text-ink-body">
                {user.name} · <span className="font-semibold capitalize text-violet">{role}</span>
              </span>
            </div>
            <button
              onClick={() => {
                logout();
                navigate("/");
              }}
              title="Log out"
              className="grid h-9 w-9 place-items-center rounded-inner border border-subtle text-ink-body transition-colors hover:border-rose/50 hover:text-rose"
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>

        {/* Tab content with crossfade */}
        <div className="p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={active}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <ActiveComp />
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </motion.div>
  );
}
