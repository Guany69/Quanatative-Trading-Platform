import createClient from "openapi-fetch";

import type { paths } from "./generated/schema";
import type { CreateRunInput } from "./types";

const client = createClient<paths>({ baseUrl: "" });

type ErrorPayload = {
  error?: { code?: string; message?: string; details?: Record<string, unknown> };
  detail?: unknown;
};

type FetchResult<T> = {
  data?: T;
  error?: unknown;
  response: Response;
};

export class ApiClientError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: Record<string, unknown>;

  constructor(code: string, message: string, status: number, details = {}) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

async function unwrap<T>(resultPromise: Promise<FetchResult<T>>): Promise<T> {
  const result = await resultPromise;
  if (result.data !== undefined) return result.data;
  const payload = (result.error ?? {}) as ErrorPayload;
  throw new ApiClientError(
    payload.error?.code ?? `HTTP_${result.response.status}`,
    payload.error?.message ?? "The server could not complete the request.",
    result.response.status,
    payload.error?.details ?? {},
  );
}

export const api = {
  health: () => unwrap(client.GET("/api/health")),
  metadata: () => unwrap(client.GET("/api/meta")),
  snapshots: () => unwrap(client.GET("/api/snapshots")),
  runs: (page = 1, state?: string) =>
    unwrap(
      client.GET("/api/runs", {
        params: { query: { page, limit: 25, state: state || undefined } },
      }),
    ),
  run: (runId: string) =>
    unwrap(client.GET("/api/runs/{run_id}", { params: { path: { run_id: runId } } })),
  runSummary: (runId: string) =>
    unwrap(
      client.GET("/api/runs/{run_id}/summary", {
        params: { path: { run_id: runId } },
      }),
    ),
  createRun: (body: CreateRunInput) => unwrap(client.POST("/api/runs", { body })),
  modelMetrics: (runId: string, model?: string) =>
    unwrap(
      client.GET("/api/runs/{run_id}/model-metrics", {
        params: {
          path: { run_id: runId },
          query: { limit: 500, model: model || undefined },
        },
      }),
    ),
  folds: (runId: string) =>
    unwrap(
      client.GET("/api/runs/{run_id}/folds", {
        params: { path: { run_id: runId }, query: { limit: 500 } },
      }),
    ),
  strategyMetrics: (runId: string) =>
    unwrap(
      client.GET("/api/runs/{run_id}/strategy-metrics", {
        params: { path: { run_id: runId }, query: { limit: 500 } },
      }),
    ),
  equity: (
    runId: string,
    filters: { model?: string; strategy?: string; costScenario?: string },
    page = 1,
  ) =>
    unwrap(
      client.GET("/api/runs/{run_id}/equity", {
        params: { path: { run_id: runId }, query: { ...filters, page, limit: 500 } },
      }),
    ),
  trades: (
    runId: string,
    filters: { model?: string; strategy?: string; costScenario?: string; security?: string },
    page = 1,
  ) =>
    unwrap(
      client.GET("/api/runs/{run_id}/trades", {
        params: { path: { run_id: runId }, query: { ...filters, page, limit: 50 } },
      }),
    ),
  predictions: (
    runId: string,
    filters: { model?: string; fold?: number; security?: string; startDate?: string; endDate?: string },
    page = 1,
  ) =>
    unwrap(
      client.GET("/api/runs/{run_id}/predictions", {
        params: { path: { run_id: runId }, query: { ...filters, page, limit: 50 } },
      }),
    ),
  weights: (
    runId: string,
    filters: { model?: string; strategy?: string; security?: string; startDate?: string; endDate?: string },
    page = 1,
  ) =>
    unwrap(
      client.GET("/api/runs/{run_id}/weights", {
        params: { path: { run_id: runId }, query: { ...filters, page, limit: 50 } },
      }),
    ),
  diagnostics: (runId: string) =>
    unwrap(
      client.GET("/api/runs/{run_id}/optimizer-diagnostics", {
        params: { path: { run_id: runId }, query: { limit: 500 } },
      }),
    ),
  overfitting: (runId: string) =>
    unwrap(
      client.GET("/api/runs/{run_id}/overfitting", {
        params: { path: { run_id: runId } },
      }),
    ),
  artifacts: (runId: string) =>
    unwrap(
      client.GET("/api/runs/{run_id}/artifacts", {
        params: { path: { run_id: runId } },
      }),
    ),
  reports: (runId: string) =>
    unwrap(
      client.GET("/api/runs/{run_id}/reports", {
        params: { path: { run_id: runId } },
      }),
    ),
  evaluateHoldout: (
    runId: string,
    body: { model: string; acknowledgedBy: string; acknowledgeOneShotEvaluation: boolean },
  ) =>
    unwrap(
      client.POST("/api/runs/{run_id}/holdout-evaluations", {
        params: { path: { run_id: runId } },
        body,
      }),
    ),
  paper: () => unwrap(client.GET("/api/paper")),
  initializePaper: (body: { initialCapital: number; accountId: string }) =>
    unwrap(client.POST("/api/paper/init", { body })),
  designateModel: (body: { runId: string; model: string; designatedBy: string }) =>
    unwrap(client.POST("/api/paper/production-model", { body })),
  createRebalance: (strategy: string) =>
    unwrap(client.POST("/api/paper/rebalances", { body: { strategy } })),
  proposals: (status?: string) =>
    unwrap(client.GET("/api/paper/proposals", { params: { query: { status } } })),
  decideProposal: (
    orderId: string,
    body: { decision: "APPROVE" | "REJECT"; actor: string; reason?: string },
  ) =>
    unwrap(
      client.POST("/api/paper/proposals/{order_id}/decision", {
        params: { path: { order_id: orderId } },
        body,
      }),
    ),
  executeRebalance: (rebalanceId: string) =>
    unwrap(
      client.POST("/api/paper/rebalances/{rebalance_id}/execute", {
        params: { path: { rebalance_id: rebalanceId } },
      }),
    ),
  positions: () => unwrap(client.GET("/api/paper/positions")),
  paperHistory: () => unwrap(client.GET("/api/paper/history")),
};
