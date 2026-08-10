export const queryKeys = {
  health: ["health"] as const,
  metadata: ["metadata"] as const,
  snapshots: ["snapshots"] as const,
  runs: (page = 1, state = "") => ["runs", page, state] as const,
  run: (runId: string) => ["run", runId] as const,
  summary: (runId: string) => ["run", runId, "summary"] as const,
  modelMetrics: (runId: string, model = "") =>
    ["run", runId, "modelMetrics", model] as const,
  strategyMetrics: (runId: string) => ["run", runId, "strategyMetrics"] as const,
  equity: (runId: string, filters: object, page: number) =>
    ["run", runId, "equity", filters, page] as const,
  trades: (runId: string, filters: object, page: number) =>
    ["run", runId, "trades", filters, page] as const,
  predictions: (runId: string, filters: object, page: number) =>
    ["run", runId, "predictions", filters, page] as const,
  weights: (runId: string, filters: object, page: number) =>
    ["run", runId, "weights", filters, page] as const,
  diagnostics: (runId: string) => ["run", runId, "diagnostics"] as const,
  overfitting: (runId: string) => ["run", runId, "overfitting"] as const,
  artifacts: (runId: string) => ["run", runId, "artifacts"] as const,
  reports: (runId: string) => ["run", runId, "reports"] as const,
  paper: ["paper"] as const,
  proposals: ["paper", "proposals"] as const,
  positions: ["paper", "positions"] as const,
  paperHistory: ["paper", "history"] as const,
};
