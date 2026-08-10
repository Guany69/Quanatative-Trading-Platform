import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { AlertTriangle, Download, FileText, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import type {
  ModelMetric,
  OptimizerDiagnostic,
  Prediction,
  Run,
  StrategyMetric,
  Trade,
  Weight,
} from "../../api/types";
import { DataTable } from "../../components/ui/DataTable";
import { MetricCard } from "../../components/ui/MetricCard";
import { Pagination } from "../../components/ui/Pagination";
import { EmptyState, ErrorState, LoadingState } from "../../components/ui/States";
import { formatCurrency, formatNumber, formatPercent } from "../../lib/format";

function ResultPanel({ children, title, kicker }: { children: React.ReactNode; title: string; kicker: string }) {
  return <section className="panel result-panel"><div className="panel-header"><div><span className="panel-kicker">{kicker}</span><h2>{title}</h2></div></div>{children}</section>;
}

export function OverviewTab({ run }: { run: Run }) {
  const summary = useQuery({ queryKey: queryKeys.summary(run.runId), queryFn: () => api.runSummary(run.runId) });
  if (summary.isLoading) return <LoadingState />;
  if (summary.error) return <ErrorState error={summary.error} />;
  const counts = summary.data!.resultCounts;
  return (
    <div className="tab-content">
      <section className="metric-grid compact-grid">
        <MetricCard label="Predictions" value={(counts.prediction ?? 0).toLocaleString()} detail="Server-paginated rows" />
        <MetricCard label="Model metrics" value={(counts.fold_metric ?? 0).toLocaleString()} detail={`${counts.fold ?? 0} folds`} />
        <MetricCard label="Strategy metrics" value={(counts.strategy_result ?? 0).toLocaleString()} detail={`${counts.trade ?? 0} trades`} />
        <MetricCard label="PIT cache" value={summary.data!.cacheHit == null ? "UNKNOWN" : summary.data!.cacheHit ? "HIT" : "MISS"} detail={summary.data!.timings?.total_seconds ? `${formatNumber(summary.data!.timings.total_seconds, 2)} seconds` : "No benchmark record"} />
      </section>
      <ResultPanel kicker="PROVENANCE" title="Run identity">
        <dl className="definition-grid">
          <div><dt>Run ID</dt><dd><code>{run.runId}</code></dd></div>
          <div><dt>Status</dt><dd>{run.state}</dd></div>
          <div><dt>Snapshot</dt><dd><code>{run.snapshotId ?? "—"}</code></dd></div>
          <div><dt>Charter hash</dt><dd><code>{run.charterHash ?? "—"}</code></dd></div>
          <div><dt>Code version</dt><dd><code>{run.codeVersion ?? "—"}</code>{run.codeDirty ? " (dirty)" : ""}</dd></div>
          <div><dt>Seed</dt><dd>{run.seed ?? "—"}</dd></div>
          <div><dt>Models</dt><dd>{run.models?.join(", ") || "—"}</dd></div>
          <div><dt>Strategies</dt><dd>{run.strategies?.join(", ") || "—"}</dd></div>
          <div><dt>Cost scenarios</dt><dd>{run.costScenarios?.join(", ") || "—"}</dd></div>
          <div><dt>Fold schedule</dt><dd><code>{run.foldScheduleHash ?? "—"}</code></dd></div>
        </dl>
      </ResultPanel>
    </div>
  );
}

export function ModelsTab({ runId }: { runId: string }) {
  const metrics = useQuery({ queryKey: queryKeys.modelMetrics(runId), queryFn: () => api.modelMetrics(runId) });
  const [metric, setMetric] = useState("mean_ic");
  const columns = useMemo<ColumnDef<ModelMetric>[]>(() => [
    { header: "Model", accessorKey: "model" },
    { header: "Fold", accessorKey: "fold" },
    { header: "Metric", accessorKey: "metric" },
    { header: "Value", accessorKey: "value", cell: ({ getValue }) => formatNumber(getValue<number>(), 5) },
  ], []);
  if (metrics.isLoading) return <LoadingState />;
  if (metrics.error) return <ErrorState error={metrics.error} />;
  const items = metrics.data?.items ?? [];
  const metricNames = [...new Set(items.map((item) => item.metric))];
  const chart = items.filter((item) => item.metric === metric).map((item) => ({ name: `${item.model} / F${item.fold}`, value: item.value }));
  return (
    <div className="tab-content two-panel-grid">
      <ResultPanel kicker="CROSS-FOLD COMPARISON" title="Model metric profile">
        <label className="field compact-field"><span>Metric</span><select value={metric} onChange={(event) => setMetric(event.target.value)}>{metricNames.map((name) => <option key={name}>{name}</option>)}</select></label>
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height={330}>
            <BarChart data={chart}><CartesianGrid strokeDasharray="3 3" stroke="#263241" /><XAxis dataKey="name" hide /><YAxis stroke="#7f8da0" /><Tooltip /><Bar dataKey="value" fill="#4cb7a5" /></BarChart>
          </ResponsiveContainer>
        </div>
      </ResultPanel>
      <ResultPanel kicker="PERSISTED RESULTS" title="Fold metrics">
        <DataTable data={items} columns={columns} compact />
      </ResultPanel>
    </div>
  );
}

export function StrategiesTab({ runId }: { runId: string }) {
  const metrics = useQuery({ queryKey: queryKeys.strategyMetrics(runId), queryFn: () => api.strategyMetrics(runId) });
  const columns = useMemo<ColumnDef<StrategyMetric>[]>(() => [
    { header: "Strategy", accessorKey: "strategy" },
    { header: "Model", accessorKey: "model" },
    { header: "Cost", accessorKey: "costScenario" },
    { header: "Metric", accessorKey: "metric" },
    { header: "Value", accessorKey: "value", cell: ({ getValue }) => formatNumber(getValue<number>(), 5) },
  ], []);
  if (metrics.isLoading) return <LoadingState />;
  if (metrics.error) return <ErrorState error={metrics.error} />;
  const items = metrics.data?.items ?? [];
  const chart = items.filter((item) => item.metric === "sharpe_ratio").map((item) => ({ name: `${item.strategy}/${item.costScenario}`, sharpe: item.value }));
  return (
    <div className="tab-content two-panel-grid">
      <ResultPanel kicker="NET COMPARISON" title="Sharpe by strategy / cost">
        {chart.length ? <div className="chart-frame"><ResponsiveContainer width="100%" height={330}><BarChart data={chart}><CartesianGrid strokeDasharray="3 3" stroke="#263241" /><XAxis dataKey="name" stroke="#7f8da0" /><YAxis stroke="#7f8da0" /><Tooltip /><Bar dataKey="sharpe" fill="#76a7e8" /></BarChart></ResponsiveContainer></div> : <EmptyState message="No persisted Sharpe metric is available." />}
      </ResultPanel>
      <ResultPanel kicker="PERSISTED RESULTS" title="Strategy metrics"><DataTable data={items} columns={columns} compact /></ResultPanel>
    </div>
  );
}

function RunFilters({ run, model, strategy, cost, setModel, setStrategy, setCost }: { run: Run; model: string; strategy: string; cost: string; setModel: (value: string) => void; setStrategy: (value: string) => void; setCost: (value: string) => void }) {
  return <div className="filter-row">
    <label><span>Model</span><select value={model} onChange={(event) => setModel(event.target.value)}><option value="">All</option>{run.models?.map((item) => <option key={item}>{item}</option>)}</select></label>
    <label><span>Strategy</span><select value={strategy} onChange={(event) => setStrategy(event.target.value)}><option value="">All</option>{run.strategies?.map((item) => <option key={item}>{item}</option>)}</select></label>
    <label><span>Cost</span><select value={cost} onChange={(event) => setCost(event.target.value)}><option value="">All</option>{run.costScenarios?.map((item) => <option key={item}>{item}</option>)}</select></label>
  </div>;
}

export function EquityTab({ runId, run }: { runId: string; run: Run }) {
  const [model, setModel] = useState(run.models?.[0] ?? "");
  const [strategy, setStrategy] = useState(run.strategies?.[0] ?? "");
  const [cost, setCost] = useState(run.costScenarios?.[0] ?? "");
  const [page, setPage] = useState(1);
  const filters = { model: model || undefined, strategy: strategy || undefined, costScenario: cost || undefined };
  const equity = useQuery({ queryKey: queryKeys.equity(runId, filters, page), queryFn: () => api.equity(runId, filters, page) });
  return <div className="tab-content"><ResultPanel kicker="BOUNDED EQUITY SERIES" title="Net, gross, and benchmark path">
    <RunFilters run={run} model={model} strategy={strategy} cost={cost} setModel={(v) => { setModel(v); setPage(1); }} setStrategy={(v) => { setStrategy(v); setPage(1); }} setCost={(v) => { setCost(v); setPage(1); }} />
    {equity.isLoading ? <LoadingState /> : equity.error ? <ErrorState error={equity.error} /> : !(equity.data?.items.length) ? <EmptyState /> : <>
      <div className="chart-frame wide"><ResponsiveContainer width="100%" height={410}><LineChart data={equity.data.items}><CartesianGrid strokeDasharray="3 3" stroke="#263241" /><XAxis dataKey="asOf" stroke="#7f8da0" minTickGap={35} /><YAxis stroke="#7f8da0" /><Tooltip formatter={(value) => formatNumber(Number(value), 4)} /><Legend /><Line type="monotone" dataKey="totalValue" stroke="#4cb7a5" dot={false} name="Total value" /><Line type="monotone" dataKey="netReturn" stroke="#76a7e8" dot={false} name="Net return" /><Line type="monotone" dataKey="benchmarkReturn" stroke="#d6a95d" dot={false} name="Benchmark" /></LineChart></ResponsiveContainer></div>
      <Pagination page={equity.data.page} totalPages={equity.data.totalPages} total={equity.data.total} onPage={setPage} />
    </>}
  </ResultPanel></div>;
}

export function TradesTab({ runId, run }: { runId: string; run: Run }) {
  const [page, setPage] = useState(1); const [security, setSecurity] = useState("");
  const [model, setModel] = useState(""); const [strategy, setStrategy] = useState(""); const [cost, setCost] = useState("");
  const filters = { model: model || undefined, strategy: strategy || undefined, costScenario: cost || undefined, security: security || undefined };
  const trades = useQuery({ queryKey: queryKeys.trades(runId, filters, page), queryFn: () => api.trades(runId, filters, page) });
  const columns = useMemo<ColumnDef<Trade>[]>(() => [
    { header: "Fill date", accessorKey: "fillDate" }, { header: "Security", accessorKey: "securityId" }, { header: "Side", accessorKey: "side" },
    { header: "Shares", accessorKey: "shares", cell: ({ getValue }) => formatNumber(getValue<number>(), 1) }, { header: "Fill", accessorKey: "fillPrice", cell: ({ getValue }) => formatNumber(getValue<number>(), 2) },
    { header: "Notional", accessorKey: "notional", cell: ({ getValue }) => formatCurrency(getValue<number>()) }, { header: "Costs", cell: ({ row }) => formatNumber(row.original.commission + row.original.spreadCost + row.original.slippageCost + row.original.impactCost, 2) },
    { header: "Strategy", accessorKey: "strategy" }, { header: "Scenario", accessorKey: "costScenario" },
  ], []);
  return <div className="tab-content"><ResultPanel kicker="SERVER-PAGINATED" title="Trade ledger"><RunFilters run={run} model={model} strategy={strategy} cost={cost} setModel={(v) => { setModel(v); setPage(1); }} setStrategy={(v) => { setStrategy(v); setPage(1); }} setCost={(v) => { setCost(v); setPage(1); }} /><label className="field compact-field search-field"><span>Security</span><input value={security} onChange={(event) => { setSecurity(event.target.value); setPage(1); }} placeholder="Permanent security ID" /></label>{trades.isLoading ? <LoadingState /> : trades.error ? <ErrorState error={trades.error} /> : <><DataTable data={trades.data?.items ?? []} columns={columns} /><Pagination page={trades.data?.page ?? page} totalPages={trades.data?.totalPages ?? 0} total={trades.data?.total ?? 0} onPage={setPage} /></>}</ResultPanel></div>;
}

export function PredictionsTab({ runId, run }: { runId: string; run: Run }) {
  const [page, setPage] = useState(1); const [model, setModel] = useState(""); const [security, setSecurity] = useState(""); const [fold, setFold] = useState<string>("");
  const filters = { model: model || undefined, security: security || undefined, fold: fold ? Number(fold) : undefined };
  const predictions = useQuery({ queryKey: queryKeys.predictions(runId, filters, page), queryFn: () => api.predictions(runId, filters, page) });
  const columns = useMemo<ColumnDef<Prediction>[]>(() => [
    { header: "As of", accessorKey: "asOf" }, { header: "Security", accessorKey: "securityId" }, { header: "Model", accessorKey: "model" }, { header: "Fold", accessorKey: "fold" },
    { header: "Score", accessorKey: "score", cell: ({ getValue }) => formatNumber(getValue<number>(), 6) }, { header: "Rank", accessorKey: "rank", cell: ({ getValue }) => formatPercent(getValue<number>(), 2) },
  ], []);
  return <div className="tab-content"><ResultPanel kicker="SERVER-PAGINATED" title="Prediction records"><div className="filter-row"><label><span>Model</span><select value={model} onChange={(event) => { setModel(event.target.value); setPage(1); }}><option value="">All</option>{run.models?.map((item) => <option key={item}>{item}</option>)}</select></label><label><span>Fold</span><input type="number" min="0" value={fold} onChange={(event) => { setFold(event.target.value); setPage(1); }} /></label><label><span>Security</span><input value={security} onChange={(event) => { setSecurity(event.target.value); setPage(1); }} /></label></div>{predictions.isLoading ? <LoadingState /> : predictions.error ? <ErrorState error={predictions.error} /> : <><DataTable data={predictions.data?.items ?? []} columns={columns} /><Pagination page={predictions.data?.page ?? page} totalPages={predictions.data?.totalPages ?? 0} total={predictions.data?.total ?? 0} onPage={setPage} /></>}</ResultPanel></div>;
}

export function WeightsTab({ runId, run }: { runId: string; run: Run }) {
  const [page, setPage] = useState(1); const [model, setModel] = useState(""); const [strategy, setStrategy] = useState(""); const [security, setSecurity] = useState(""); const [startDate, setStartDate] = useState(""); const [endDate, setEndDate] = useState("");
  const filters = { model: model || undefined, strategy: strategy || undefined, security: security || undefined, startDate: startDate || undefined, endDate: endDate || undefined };
  const weights = useQuery({ queryKey: queryKeys.weights(runId, filters, page), queryFn: () => api.weights(runId, filters, page) });
  const columns = useMemo<ColumnDef<Weight>[]>(() => [{ header: "As of", accessorKey: "asOf" }, { header: "Security", accessorKey: "securityId" }, { header: "Model", accessorKey: "model" }, { header: "Strategy", accessorKey: "strategy" }, { header: "Weight", accessorKey: "weight", cell: ({ getValue }) => formatPercent(getValue<number>(), 3) }], []);
  return <div className="tab-content"><ResultPanel kicker="SERVER-PAGINATED" title="Target weight history"><div className="filter-row"><label><span>Model</span><select value={model} onChange={(event) => { setModel(event.target.value); setPage(1); }}><option value="">All</option>{run.models?.map((item) => <option key={item}>{item}</option>)}</select></label><label><span>Strategy</span><select value={strategy} onChange={(event) => { setStrategy(event.target.value); setPage(1); }}><option value="">All</option>{run.strategies?.map((item) => <option key={item}>{item}</option>)}</select></label><label><span>Security</span><input value={security} onChange={(event) => { setSecurity(event.target.value); setPage(1); }} /></label><label><span>Start date</span><input type="date" value={startDate} onChange={(event) => { setStartDate(event.target.value); setPage(1); }} /></label><label><span>End date</span><input type="date" value={endDate} onChange={(event) => { setEndDate(event.target.value); setPage(1); }} /></label></div>{weights.isLoading ? <LoadingState /> : weights.error ? <ErrorState error={weights.error} /> : <><DataTable data={weights.data?.items ?? []} columns={columns} /><Pagination page={weights.data?.page ?? page} totalPages={weights.data?.totalPages ?? 0} total={weights.data?.total ?? 0} onPage={setPage} /></>}</ResultPanel></div>;
}

export function DiagnosticsTab({ runId }: { runId: string }) {
  const diagnostics = useQuery({ queryKey: queryKeys.diagnostics(runId), queryFn: () => api.diagnostics(runId) });
  const overfitting = useQuery({ queryKey: queryKeys.overfitting(runId), queryFn: () => api.overfitting(runId) });
  const columns = useMemo<ColumnDef<OptimizerDiagnostic>[]>(() => [{ header: "As of", accessorKey: "asOf" }, { header: "Status", accessorKey: "solverStatus" }, { header: "Solver", accessorKey: "solverName" }, { header: "Turnover", accessorKey: "expectedTurnover", cell: ({ getValue }) => formatPercent(getValue<number>(), 2) }, { header: "Cost", accessorKey: "expectedCost", cell: ({ getValue }) => formatNumber(getValue<number>(), 4) }, { header: "Relaxations", accessorKey: "relaxations", cell: ({ row }) => row.original.relaxations.length ? <span className="relaxation-alert"><AlertTriangle size={14} /> {row.original.relaxations.join(", ")}</span> : "None" }], []);
  if (diagnostics.isLoading || overfitting.isLoading) return <LoadingState />;
  if (diagnostics.error) return <ErrorState error={diagnostics.error} />;
  if (overfitting.error) return <ErrorState error={overfitting.error} />;
  return <div className="tab-content two-panel-grid"><ResultPanel kicker="OPTIMIZER" title="Constraint / relaxation audit"><DataTable data={diagnostics.data?.items ?? []} columns={columns} compact /></ResultPanel><ResultPanel kicker="SELECTION RISK" title="Overfitting & stress"><div className="stress-list">{(overfitting.data ?? []).map((item) => <div className={item.optimistic ? "stress-row warning" : "stress-row"} key={item.metric}><div><strong>{item.metric}</strong><span>{item.optimistic ? "Optimistic / incomplete trial linkage" : "Persisted statistic"}</span></div><code>{formatNumber(item.value, 5)}</code></div>)}</div></ResultPanel></div>;
}

export function ReproducibilityTab({ runId, run }: { runId: string; run: Run }) {
  const folds = useQuery({ queryKey: ["run", runId, "folds"], queryFn: () => api.folds(runId) });
  const artifacts = useQuery({ queryKey: queryKeys.artifacts(runId), queryFn: () => api.artifacts(runId) });
  if (folds.isLoading || artifacts.isLoading) return <LoadingState />;
  if (folds.error) return <ErrorState error={folds.error} />;
  if (artifacts.error) return <ErrorState error={artifacts.error} />;
  return <div className="tab-content two-panel-grid"><ResultPanel kicker="IDENTITY" title="Reproduction inputs"><dl className="definition-grid single"><div><dt>Snapshot identity</dt><dd><code>{run.snapshotId}</code></dd></div><div><dt>Configuration identity</dt><dd><code>{run.charterHash}</code></dd></div><div><dt>Code version</dt><dd><code>{run.codeVersion}</code>{run.codeDirty && " · dirty"}</dd></div><div><dt>Random seed</dt><dd>{run.seed}</dd></div><div><dt>Fold schedule hash</dt><dd><code>{run.foldScheduleHash}</code></dd></div></dl></ResultPanel><ResultPanel kicker="FOLDS & ARTIFACTS" title="Persisted boundaries"><div className="mini-records">{folds.data?.items.map((fold) => <div key={fold.fold}><strong>Fold {fold.fold}</strong><span>Train {fold.trainStart} → {fold.trainEnd}</span><span>Test {fold.testStart} → {fold.testEnd}</span></div>)}</div><div className="artifact-count"><ShieldCheck size={18} /> {artifacts.data?.length ?? 0} model/fold artifact references verified by the backend</div></ResultPanel></div>;
}

export function ReportsTab({ runId }: { runId: string }) {
  const reports = useQuery({ queryKey: queryKeys.reports(runId), queryFn: () => api.reports(runId) });
  if (reports.isLoading) return <LoadingState />;
  if (reports.error) return <ErrorState error={reports.error} />;
  if (!reports.data?.length) return <EmptyState title="No reports" message="No run-owned report files were discovered." />;
  return <div className="tab-content"><ResultPanel kicker="SAFE SERVER-OWNED URLS" title="Reports and exports"><div className="report-grid">{reports.data.map((report) => <a key={report.id} className="report-card" href={report.url} target="_blank" rel="noreferrer"><FileText size={21} /><div><strong>{report.name}</strong><span>{report.type.toUpperCase()} · {formatNumber(report.size / 1024, 1)} KB</span></div><Download size={16} /></a>)}</div></ResultPanel></div>;
}
