import { expect, test, type BrowserContext } from "@playwright/test";

const completedRun = {
  runId: "run-complete-001",
  state: "COMPLETED",
  stage: "REPORTING",
  completedStages: 6,
  totalStages: 6,
  createdAt: "2026-08-10T12:00:00Z",
  startedAt: "2026-08-10T12:00:01Z",
  completedAt: "2026-08-10T12:00:09Z",
  updatedAt: "2026-08-10T12:00:09Z",
  failure: null,
  snapshotId: "snapshot-fixture-001",
  charterHash: "charter-abc123",
  codeVersion: "deadbeef",
  codeDirty: false,
  seed: 42,
  models: ["factor_composite", "no_skill"],
  strategies: ["equal_weight"],
  costScenarios: ["base"],
  foldScheduleHash: "folds-abc123",
  durationSeconds: 8,
} as const;

const failedRun = {
  ...completedRun,
  runId: "run-failed-001",
  state: "FAILED",
  stage: "MODEL_TRAINING",
  completedStages: 2,
  failure: { code: "MODEL_EXECUTION_FAILED", message: "Fixture model refused invalid inputs.", stage: "MODEL_TRAINING" },
} as const;

const metadata = {
  models: [
    { id: "factor_composite", name: "Factor Composite", description: "Governed factor model", productionEligible: true },
    { id: "no_skill", name: "No Skill", description: "Control", productionEligible: false },
  ],
  strategies: [{ id: "equal_weight", name: "Equal Weight", description: "Equal-weight long book", productionEligible: null }],
  costScenarios: [{ id: "base", name: "Base", description: "Base transaction costs", productionEligible: null }],
  features: [{ name: "momentum_12_1", family: "momentum", lookbackSessions: 252, minObservations: 126, missingPolicy: "drop", version: "1", description: "Momentum" }],
  portfolioModels: ["factor_composite"],
  defaultCharterId: "default",
};

const paper = {
  accountId: "paper-001",
  createdAt: "2026-08-10T12:00:00Z",
  updatedAt: "2026-08-10T12:01:00Z",
  initialCapital: 1_000_000,
  totalValue: 1_010_000,
  cash: 750_000,
  positionsValue: 260_000,
  nPositions: 2,
  totalReturn: 0.01,
  realizedPnl: 1_000,
  totalCosts: 50,
  nRebalances: 1,
  pendingApprovals: 0,
  productionModel: { runId: completedRun.runId, model: "factor_composite", designatedAt: "2026-08-10T12:00:00Z", designatedBy: "operator" },
  metrics: {},
};

type MockOptions = {
  pollRun?: boolean;
  holdoutRefusal?: boolean;
};

async function installApi(context: BrowserContext, options: MockOptions = {}) {
  let runPolls = 0;
  let holdoutPosts = 0;
  let proposals = [
    proposal("order-1", "SEC-ALPHA"),
    proposal("order-2", "SEC-BETA"),
  ];

  await context.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();
    if (!path.startsWith("/api/")) return route.continue();
    const json = (body: unknown, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

    if (path === "/api/health") return json({ status: "ok" });
    if (path === "/api/meta") return json(metadata);
    if (path === "/api/snapshots") return json([{ snapshotId: "snapshot-fixture-001", createdAt: "2026-08-10T12:00:00Z", sources: ["fixture"], survivorshipBiased: false, tableCount: 4, validation: {} }]);
    if (path === "/api/runs" && method === "POST") return json({ runId: "run-new-001", state: "QUEUED" }, 202);
    if (path === "/api/runs") return json(pageOf([completedRun, failedRun], 1, 2));

    if (path === "/api/runs/run-new-001") {
      runPolls += 1;
      if (!options.pollRun || runPolls >= 3) return json({ ...completedRun, runId: "run-new-001" });
      if (runPolls === 1) return json({ ...completedRun, runId: "run-new-001", state: "QUEUED", stage: null, completedStages: 0, completedAt: null });
      return json({ ...completedRun, runId: "run-new-001", state: "RUNNING", stage: "MODEL_TRAINING", completedStages: 2, completedAt: null });
    }
    if (path === `/api/runs/${completedRun.runId}`) return json(completedRun);
    if (path === `/api/runs/${failedRun.runId}`) return json(failedRun);

    const runMatch = path.match(/^\/api\/runs\/([^/]+)\/(.+)$/);
    if (runMatch) {
      const runId = runMatch[1]!;
      const resource = runMatch[2]!;
      if (resource === "summary") return json({ run: { ...completedRun, runId }, resultCounts: { prediction: 23294, fold_metric: 30, fold: 5, strategy_result: 18, trade: 42 }, cacheHit: true, timings: { total_seconds: 8.12 } });
      if (resource === "model-metrics") return json(pageOf([
        { model: "factor_composite", fold: 0, metric: "mean_ic", value: 0.052 },
        { model: "no_skill", fold: 0, metric: "mean_ic", value: 0.001 },
      ]));
      if (resource === "strategy-metrics") return json(pageOf([
        { model: "factor_composite", strategy: "equal_weight", costScenario: "base", metric: "sharpe_ratio", value: 1.21 },
      ]));
      if (resource === "predictions") {
        const security = url.searchParams.get("security");
        const page = Number(url.searchParams.get("page") ?? "1");
        const securityId = security || (page === 2 ? "SEC-PAGE-TWO" : "SEC-PAGE-ONE");
        return json(pageOf([{ model: "factor_composite", fold: 0, asOf: "2026-01-02", securityId, score: 0.25, rank: 0.9 }], page, 2));
      }
      if (resource === "reports") return json([{ id: "report-safe-id", name: "research_report.html", type: "html", size: 4096, url: `/api/runs/${runId}/reports/report-safe-id` }]);
      if (resource === "reports/report-safe-id") return route.fulfill({ status: 200, contentType: "text/html", body: "<title>Governed Research Report</title><h1>Governed Research Report</h1>" });
      if (resource === "holdout-evaluations" && method === "POST") {
        holdoutPosts += 1;
        if (options.holdoutRefusal) return json({ error: { code: "HOLDOUT_VIOLATION", message: "The locked holdout was already consumed.", details: {} } }, 409);
        return json({ runId, model: "factor_composite", metrics: { mean_ic: 0.04 }, consumed: true });
      }
      if (resource === "trades") return json(pageOf([], 1, 0));
      if (resource === "equity") return json(pageOf([], 1, 0));
      if (resource === "weights") return json(pageOf([], 1, 0));
      if (resource === "optimizer-diagnostics") return json(pageOf([], 1, 0));
      if (resource === "overfitting") return json([]);
      if (resource === "folds") return json(pageOf([], 1, 0));
      if (resource === "artifacts") return json([]);
    }

    if (path === "/api/paper") return json({ ...paper, pendingApprovals: proposals.filter((item) => item.status === "PROPOSED").length });
    if (path === "/api/paper/positions") return json([]);
    if (path === "/api/paper/history") return json({ rebalances: [], proposals: [], decisions: [], fills: [], reconciliations: [], productionModels: [] });
    if (path === "/api/paper/proposals" && method === "GET") return json(proposals);
    const decisionMatch = path.match(/^\/api\/paper\/proposals\/([^/]+)\/decision$/);
    if (decisionMatch && method === "POST") {
      const body = request.postDataJSON() as { decision: "APPROVE" | "REJECT"; actor: string; reason?: string };
      proposals = proposals.map((item) => item.orderId === decisionMatch[1]
        ? { ...item, status: body.decision === "APPROVE" ? "APPROVED" : "REJECTED", approved: body.decision === "APPROVE", approvedBy: body.decision === "APPROVE" ? body.actor : null, rejectedBy: body.decision === "REJECT" ? body.actor : null, rejectionReason: body.reason ?? null }
        : item);
      return json(proposals.find((item) => item.orderId === decisionMatch[1]));
    }
    if (/^\/api\/paper\/rebalances\/[^/]+\/execute$/.test(path) && method === "POST") {
      if (proposals.some((item) => item.status === "PROPOSED")) return json({ error: { code: "PENDING_DECISIONS", message: "Every proposal requires a decision.", details: {} } }, 409);
      proposals = proposals.map((item) => item.status === "APPROVED" ? { ...item, status: "EXECUTED" } : item);
      return json({ rebalanceId: "rebalance-001", signalDate: "2026-08-07", orderDate: "2026-08-10", state: "EXECUTED", proposed: 2, approved: 1, rejected: 1, filled: 1, realizedCost: 2.5, warnings: [] });
    }

    return json({ error: { code: "NOT_FOUND", message: `No fixture for ${method} ${path}`, details: {} } }, 404);
  });

  return {
    getRunPolls: () => runPolls,
    getHoldoutPosts: () => holdoutPosts,
  };
}

function pageOf<T>(items: T[], page = 1, total = items.length) {
  return { items, page, limit: 50, total, totalPages: total ? Math.max(1, Math.ceil(total / 1)) : 0 };
}

type ProposalFixture = {
  orderId: string;
  rebalanceId: string;
  securityId: string;
  side: string;
  quantity: number;
  signalDate: string;
  orderDate: string;
  targetWeight: number;
  currentWeight: number;
  referencePrice: number;
  expectedCost: number;
  expectedCostComponents: Record<string, number>;
  status: string;
  approved: boolean;
  approvedAt: string | null;
  approvedBy: string | null;
  rejectionReason: string | null;
  rejectedAt: string | null;
  rejectedBy: string | null;
};

function proposal(orderId: string, securityId: string): ProposalFixture {
  return {
    orderId,
    rebalanceId: "rebalance-001",
    securityId,
    side: "BUY",
    quantity: 10,
    signalDate: "2026-08-07",
    orderDate: "2026-08-10",
    targetWeight: 0.05,
    currentWeight: 0,
    referencePrice: 100,
    expectedCost: 2.5,
    expectedCostComponents: { commission: 1 },
    status: "PROPOSED",
    approved: false,
    approvedAt: null,
    approvedBy: null,
    rejectionReason: null,
    rejectedAt: null,
    rejectedBy: null,
  };
}

test("application starts, health succeeds, and backend metadata renders", async ({ page, context }) => {
  await installApi(context);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Research control room" })).toBeVisible();
  await expect(page.getByText("ONLINE", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Start research run" }).click();
  await expect(page.getByRole("heading", { name: "Configure research run" })).toBeVisible();
  await expect(page.getByText("Factor Composite")).toBeVisible();
  await expect(page.getByText("Equal Weight")).toBeVisible();
});

test("research history opens completed and failed run details", async ({ page, context }) => {
  await installApi(context);
  await page.goto("/research/runs");
  await expect(page.getByRole("heading", { name: "Run history" })).toBeVisible();
  await expect(page.getByText("run-comp")).toBeVisible();
  await page.getByRole("link", { name: "run-comp" }).click();
  await expect(page.getByRole("heading", { name: "run-complete-001" })).toBeVisible();
  await page.goto(`/research/runs/${failedRun.runId}`);
  await expect(page.getByRole("alert")).toContainText("MODEL_EXECUTION_FAILED · MODEL_TRAINING");
  await expect(page.getByRole("alert")).toContainText("Fixture model refused invalid inputs.");
});

test("new research submission returns immediately, polls real stages, then stops", async ({ page, context }) => {
  const apiState = await installApi(context, { pollRun: true });
  await page.goto("/research/new");
  const started = Date.now();
  await page.getByRole("button", { name: "Queue research run" }).click();
  await expect(page).toHaveURL(/\/research\/runs\/run-new-001/, { timeout: 1_000 });
  expect(Date.now() - started).toBeLessThan(1_000);
  await expect(page.getByText("This job is durable")).toBeVisible();
  await expect(page.getByText("Model training")).toBeVisible({ timeout: 4_000 });
  await expect(page.getByText("23,294")).toBeVisible({ timeout: 6_000 });
  const terminalPolls = apiState.getRunPolls();
  await page.waitForTimeout(1_800);
  expect(apiState.getRunPolls()).toBe(terminalPolls);
});

test("persisted result tabs render metrics, server pagination, and filters", async ({ page, context }) => {
  await installApi(context);
  await page.goto(`/research/runs/${completedRun.runId}`);
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await expect(page.getByRole("cell", { name: "0.052" })).toBeVisible();
  await page.getByRole("button", { name: "Strategies", exact: true }).click();
  await expect(page.getByText("sharpe_ratio")).toBeVisible();
  await page.getByRole("button", { name: "Predictions", exact: true }).click();
  await expect(page.getByText("SEC-PAGE-ONE")).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByText("SEC-PAGE-TWO")).toBeVisible();
  await page.getByLabel("Security").fill("SEC-FILTERED");
  await expect(page.getByText("SEC-FILTERED")).toBeVisible();
});

test("locked holdout requires confirmation and renders structured refusal", async ({ page, context }) => {
  const apiState = await installApi(context, { holdoutRefusal: true });
  await page.goto(`/research/runs/${completedRun.runId}`);
  await page.getByRole("button", { name: "Evaluate holdout" }).click();
  expect(apiState.getHoldoutPosts()).toBe(0);
  const confirm = page.getByRole("button", { name: "I understand — evaluate once" });
  await expect(confirm).toBeDisabled();
  await page.getByLabel("Acknowledged by").fill("risk-owner");
  await confirm.click();
  await expect(page.getByRole("alert")).toContainText("HOLDOUT_VIOLATION");
  await expect(page.getByRole("alert")).toContainText("already consumed");
});

test("paper decisions persist and execution stays gated until every decision", async ({ page, context }) => {
  await installApi(context);
  await page.goto("/paper/proposals");
  const execute = page.getByRole("button", { name: "Execute approved set" });
  await expect(page.getByText("SEC-ALPHA")).toBeVisible();
  await expect(execute).toBeDisabled();
  await page.getByRole("button", { name: "Approve SEC-ALPHA" }).click();
  await expect(page.getByText("APPROVED", { exact: true })).toBeVisible();
  await expect(execute).toBeDisabled();
  await page.getByRole("button", { name: "Reject SEC-BETA" }).click();
  await page.getByLabel("Reason").fill("Exposure limit");
  await page.getByRole("button", { name: "Record rejection" }).click();
  await expect(page.getByText("REJECTED", { exact: true })).toBeVisible();
  await expect(execute).toBeEnabled();
  await execute.click();
  await expect(page.getByText(/Executed 1 approved fill/)).toBeVisible();
});

test("report list exposes only opaque server-owned URLs and valid reports open", async ({ page, context }) => {
  await installApi(context);
  await page.goto(`/research/runs/${completedRun.runId}`);
  await page.getByRole("button", { name: "Reports", exact: true }).click();
  const report = page.getByRole("link", { name: /research_report\.html/ });
  await expect(report).toHaveAttribute("href", `/api/runs/${completedRun.runId}/reports/report-safe-id`);
  await expect(page.locator('a[href*=".."]')).toHaveCount(0);
  const popupPromise = page.waitForEvent("popup");
  await report.click();
  const popup = await popupPromise;
  await expect(popup.getByRole("heading", { name: "Governed Research Report" })).toBeVisible();
});
