import { createBrowserRouter, Navigate } from "react-router-dom";

import { App } from "./App";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    hydrateFallbackElement: <section className="state-panel">Loading workstation view…</section>,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      { path: "dashboard", lazy: async () => ({ Component: (await import("../features/dashboard/DashboardPage")).DashboardPage }) },
      { path: "research/new", lazy: async () => ({ Component: (await import("../features/research/NewRunPage")).NewRunPage }) },
      { path: "research/runs", lazy: async () => ({ Component: (await import("../features/research/RunListPage")).RunListPage }) },
      { path: "research/runs/:runId", lazy: async () => ({ Component: (await import("../features/research/RunDetailPage")).RunDetailPage }) },
      { path: "paper", lazy: async () => ({ Component: (await import("../features/paper/PaperDashboardPage")).PaperDashboardPage }) },
      { path: "paper/proposals", lazy: async () => ({ Component: (await import("../features/paper/ProposalsPage")).ProposalsPage }) },
      { path: "paper/history", lazy: async () => ({ Component: (await import("../features/paper/PaperHistoryPage")).PaperHistoryPage }) },
      { path: "scorecard", lazy: async () => ({ Component: (await import("../features/scorecard/ScorecardPage")).ScorecardPage }) },
      {
        path: "*",
        element: <section className="state-panel"><strong>Route not found</strong><a href="/dashboard">Return to dashboard</a></section>,
      },
    ],
  },
]);
