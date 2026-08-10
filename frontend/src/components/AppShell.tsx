import {
  Activity,
  BarChart3,
  ClipboardCheck,
  FlaskConical,
  History,
  LayoutDashboard,
  Menu,
  PlusCircle,
  X,
} from "lucide-react";
import type { PropsWithChildren } from "react";
import { useState } from "react";
import { NavLink } from "react-router-dom";

const groups = [
  {
    label: "Workspace",
    items: [{ to: "/dashboard", label: "Dashboard", icon: LayoutDashboard }],
  },
  {
    label: "Research",
    items: [
      { to: "/research/new", label: "New run", icon: PlusCircle },
      { to: "/research/runs", label: "Run history", icon: FlaskConical },
    ],
  },
  {
    label: "Paper",
    items: [
      { to: "/paper", label: "Account", icon: Activity },
      { to: "/paper/proposals", label: "Approvals", icon: ClipboardCheck },
      { to: "/paper/history", label: "History", icon: History },
    ],
  },
  {
    label: "Analysis",
    items: [{ to: "/scorecard", label: "Scorecard", icon: BarChart3 }],
  },
];

export function AppShell({ children }: PropsWithChildren) {
  const [open, setOpen] = useState(false);
  return (
    <div className="app-frame">
      <aside className={open ? "sidebar sidebar-open" : "sidebar"}>
        <div className="brand">
          <div className="brand-mark">QP</div>
          <div>
            <strong>QUANT PLATFORM</strong>
            <span>Research workstation</span>
          </div>
          <button className="icon-button mobile-only" onClick={() => setOpen(false)}>
            <X size={18} />
          </button>
        </div>
        <nav>
          {groups.map((group) => (
            <div className="nav-group" key={group.label}>
              <div className="nav-label">{group.label}</div>
              {group.items.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}
                  onClick={() => setOpen(false)}
                >
                  <Icon size={17} />
                  {label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className="live-dot" /> Loopback session
          <small>No live broker connected</small>
        </div>
      </aside>
      <div className="main-column">
        <header className="topbar">
          <button className="icon-button mobile-only" onClick={() => setOpen(true)}>
            <Menu size={19} />
          </button>
          <span className="topbar-context">LOCAL / US EQUITIES / 20D EXCESS RETURN</span>
          <span className="paper-pill">PAPER ONLY</span>
        </header>
        <main className="workspace">{children}</main>
      </div>
    </div>
  );
}
