import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { ArrowRight, Landmark, PlayCircle, Shield, WalletCards } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api, ApiClientError } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import type { Position } from "../../api/types";
import { DataTable } from "../../components/ui/DataTable";
import { MetricCard } from "../../components/ui/MetricCard";
import { PageHeader } from "../../components/ui/PageHeader";
import { ErrorState, LoadingState } from "../../components/ui/States";
import { formatCurrency, formatNumber, formatPercent, shortId } from "../../lib/format";

export function PaperDashboardPage() {
  const client = useQueryClient();
  const [capital, setCapital] = useState(1_000_000);
  const [accountId, setAccountId] = useState("paper-001");
  const [runId, setRunId] = useState("");
  const [model, setModel] = useState("factor_composite");
  const [actor, setActor] = useState("");
  const paper = useQuery({ queryKey: queryKeys.paper, queryFn: api.paper, retry: false });
  const positions = useQuery({ queryKey: queryKeys.positions, queryFn: api.positions, enabled: Boolean(paper.data) });
  const initialize = useMutation({ mutationFn: () => api.initializePaper({ initialCapital: capital, accountId }), onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.paper }) });
  const designate = useMutation({ mutationFn: () => api.designateModel({ runId, model, designatedBy: actor }), onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.paper }) });
  const rebalance = useMutation({ mutationFn: () => api.createRebalance("equal_weight"), onSuccess: () => { client.invalidateQueries({ queryKey: queryKeys.paper }); client.invalidateQueries({ queryKey: queryKeys.proposals }); } });
  const columns = useMemo<ColumnDef<Position>[]>(() => [
    { header: "Security", accessorKey: "securityId" }, { header: "Quantity", accessorKey: "quantity", cell: ({ getValue }) => formatNumber(getValue<number>(), 2) },
    { header: "Average cost", accessorKey: "averageCost", cell: ({ getValue }) => formatCurrency(getValue<number>()) }, { header: "Last price", accessorKey: "lastPrice", cell: ({ getValue }) => formatNumber(getValue<number>(), 2) },
    { header: "Market value", accessorKey: "marketValue", cell: ({ getValue }) => formatCurrency(getValue<number>()) }, { header: "Unrealized P&L", accessorKey: "unrealizedPnl", cell: ({ getValue }) => formatCurrency(getValue<number>()) },
  ], []);
  const missing = paper.error instanceof ApiClientError && paper.error.code === "PAPER_STATE_NOT_INITIALIZED";
  if (paper.isLoading) return <LoadingState label="Loading authoritative paper state…" />;
  if (paper.error && !missing) return <ErrorState error={paper.error} />;

  if (missing) {
    return <><PageHeader eyebrow="Forward-only state" title="Initialize paper account" description="This creates the separate authoritative JSON state. Research results remain replayable and untouched." /><section className="panel initialize-card"><Landmark size={30} /><h2>New simulated account</h2><div className="form-grid"><label className="field"><span>Account ID</span><input value={accountId} onChange={(event) => setAccountId(event.target.value)} /></label><label className="field"><span>Initial capital</span><input type="number" value={capital} onChange={(event) => setCapital(Number(event.target.value))} /></label></div>{initialize.error && <ErrorState error={initialize.error} />}<button className="button primary" disabled={initialize.isPending} onClick={() => initialize.mutate()}>{initialize.isPending ? "Initializing…" : "Initialize paper state"}</button></section></>;
  }

  const state = paper.data!;
  return <>
    <PageHeader eyebrow="Paper trading" title={state.accountId} description="Forward-only simulated account. Every mutation is serialized and atomically persisted." actions={<Link className="button secondary" to="/paper/proposals">Approval inbox <ArrowRight size={15} /></Link>} />
    <section className="metric-grid"><MetricCard label="Paper equity" value={formatCurrency(state.totalValue)} detail={formatPercent(state.totalReturn)} icon={<WalletCards size={18} />} /><MetricCard label="Cash" value={formatCurrency(state.cash)} detail={`${formatPercent(state.cash / state.totalValue)} of equity`} /><MetricCard label="Positions" value={state.nPositions} detail={formatCurrency(state.positionsValue)} /><MetricCard label="Pending approvals" value={state.pendingApprovals} detail={`${state.nRebalances} recorded rebalances`} tone={state.pendingApprovals ? "warning" : "default"} /></section>
    <div className="paper-grid">
      <section className="panel span-2"><div className="panel-header"><div><span className="panel-kicker">AUTHORITATIVE BOOK</span><h2>Positions</h2></div></div>{positions.isLoading ? <LoadingState /> : positions.error ? <ErrorState error={positions.error} /> : <DataTable data={positions.data ?? []} columns={columns} emptyMessage="No positions have filled yet." />}</section>
      <section className="panel"><div className="panel-header"><div><span className="panel-kicker">EXPLICIT DESIGNATION</span><h2>Production model</h2></div><Shield size={18} /></div>{state.productionModel ? <div className="production-card"><strong>{state.productionModel.model}</strong><code>{shortId(state.productionModel.runId)}</code><span>Designated by {state.productionModel.designatedBy}</span></div> : <p className="muted">No model may generate proposals until a completed research model is explicitly designated.</p>}<div className="stack-form"><label className="field"><span>Completed run ID</span><input value={runId} onChange={(event) => setRunId(event.target.value)} /></label><label className="field"><span>Model</span><input value={model} onChange={(event) => setModel(event.target.value)} /></label><label className="field"><span>Designated by</span><input value={actor} onChange={(event) => setActor(event.target.value)} /></label><button className="button secondary full" disabled={!runId || !actor || designate.isPending} onClick={() => designate.mutate()}>Set production model</button>{designate.error && <ErrorState error={designate.error} title="Designation refused" />}</div></section>
      <section className="panel span-3 action-panel"><div><span className="panel-kicker">NEXT REBALANCE</span><h2>Generate reviewable proposals</h2><p>Prediction, portfolio construction, and expected cost stay in Python. This action stops before approval.</p></div><button className="button primary" disabled={!state.productionModel || rebalance.isPending || state.pendingApprovals > 0} onClick={() => rebalance.mutate()}><PlayCircle size={16} /> {rebalance.isPending ? "Generating…" : "Generate proposals"}</button>{rebalance.data && <Link to="/paper/proposals" className="success-link">{rebalance.data.proposed} proposals ready for human review →</Link>}{rebalance.error && <ErrorState error={rebalance.error} title="Proposal generation blocked" />}</section>
    </div>
  </>;
}
