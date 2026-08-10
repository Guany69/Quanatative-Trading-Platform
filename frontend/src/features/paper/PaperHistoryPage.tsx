import { useQuery } from "@tanstack/react-query";

import { api } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import { PageHeader } from "../../components/ui/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "../../components/ui/States";
import { formatCurrency, formatDate } from "../../lib/format";

export function PaperHistoryPage() {
  const history = useQuery({ queryKey: queryKeys.paperHistory, queryFn: api.paperHistory, retry: false });
  if (history.isLoading) return <LoadingState />;
  if (history.error) return <ErrorState error={history.error} />;
  const data = history.data!;
  return <><PageHeader eyebrow="Forward audit log" title="Paper history" description="Decisions, fills, reconciliation, and production designations are recorded when made—not reconstructed." /><div className="history-grid"><HistorySection title="Rebalances" items={data.rebalances} render={(item) => <><strong>{String(item.rebalance_id ?? "Rebalance")}</strong><span>{String(item.signal_date ?? "")} → {String(item.order_date ?? "")}</span><code>{String(item.n_filled ?? 0)} fills · {formatCurrency(Number(item.realized_cost ?? 0))}</code></>} /><HistorySection title="Decisions" items={data.decisions} render={(item) => <><strong>{String(item.order_id ?? "Order")}</strong><span>{String(item.security_id ?? "")} · {String(item.status ?? "").toUpperCase()}</span><code>{String(item.approved_by ?? item.rejected_by ?? "unknown actor")}</code></>} /><HistorySection title="Fills" items={data.fills} render={(item) => <><strong>{String(item.security_id ?? "Security")}</strong><span>{String(item.side ?? "")} {String(item.quantity ?? "")} @ {String(item.fill_price ?? "")}</span><code>{String(item.fill_date ?? "")}</code></>} /><HistorySection title="Production designations" items={data.productionModels} render={(item) => <><strong>{String(item.model ?? "Model")}</strong><span>{String(item.run_id ?? "")}</span><code>{formatDate(String(item.designated_at ?? ""))}</code></>} /></div></>;
}

function HistorySection({ title, items, render }: { title: string; items: Record<string, unknown>[]; render: (item: Record<string, unknown>) => React.ReactNode }) {
  return <section className="panel"><div className="panel-header"><div><span className="panel-kicker">AUDIT RECORDS</span><h2>{title}</h2></div><span>{items.length}</span></div>{!items.length ? <EmptyState message={`No ${title.toLowerCase()} recorded.`} /> : <div className="history-list">{items.slice().reverse().map((item, index) => <div className="history-row" key={`${title}-${index}`}>{render(item)}</div>)}</div>}</section>;
}
