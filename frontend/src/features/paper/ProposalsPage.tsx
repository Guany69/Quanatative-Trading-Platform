import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { Check, Play, X } from "lucide-react";
import { useMemo, useState } from "react";

import { api } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import type { Proposal } from "../../api/types";
import { DataTable } from "../../components/ui/DataTable";
import { Modal } from "../../components/ui/Modal";
import { PageHeader } from "../../components/ui/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "../../components/ui/States";
import { StatusBadge } from "../../components/ui/StatusBadge";
import { formatCurrency, formatNumber, formatPercent } from "../../lib/format";

export function ProposalsPage() {
  const client = useQueryClient();
  const [actor, setActor] = useState("local-reviewer");
  const [rejecting, setRejecting] = useState<Proposal | null>(null);
  const [reason, setReason] = useState("");
  const proposals = useQuery({ queryKey: queryKeys.proposals, queryFn: () => api.proposals() });
  const decision = useMutation({
    mutationFn: ({ orderId, action, reason }: { orderId: string; action: "APPROVE" | "REJECT"; reason?: string }) => api.decideProposal(orderId, { decision: action, actor, reason }),
    onSuccess: () => { setRejecting(null); setReason(""); client.invalidateQueries({ queryKey: queryKeys.proposals }); client.invalidateQueries({ queryKey: queryKeys.paper }); },
  });
  const execute = useMutation({ mutationFn: (rebalanceId: string) => api.executeRebalance(rebalanceId), onSuccess: () => { client.invalidateQueries({ queryKey: queryKeys.proposals }); client.invalidateQueries({ queryKey: queryKeys.paper }); client.invalidateQueries({ queryKey: queryKeys.paperHistory }); } });
  const columns = useMemo<ColumnDef<Proposal>[]>(() => [
    { header: "Security", accessorKey: "securityId" }, { header: "Side", accessorKey: "side" }, { header: "Quantity", accessorKey: "quantity", cell: ({ getValue }) => formatNumber(getValue<number>(), 1) },
    { header: "Current", accessorKey: "currentWeight", cell: ({ getValue }) => formatPercent(getValue<number>(), 2) }, { header: "Target", accessorKey: "targetWeight", cell: ({ getValue }) => formatPercent(getValue<number>(), 2) },
    { header: "Reference", accessorKey: "referencePrice", cell: ({ getValue }) => formatCurrency(getValue<number>()) }, { header: "Expected cost", accessorKey: "expectedCost", cell: ({ getValue }) => formatCurrency(getValue<number>()) },
    { header: "Order date", accessorKey: "orderDate" }, { header: "Status", accessorKey: "status", cell: ({ getValue }) => <StatusBadge value={getValue<string>()} /> },
    { header: "Decision", cell: ({ row }) => row.original.status === "PROPOSED" ? <div className="row-actions"><button aria-label={`Approve ${row.original.securityId}`} className="approve-button" disabled={!actor || decision.isPending} onClick={() => decision.mutate({ orderId: row.original.orderId, action: "APPROVE" })}><Check size={15} /></button><button aria-label={`Reject ${row.original.securityId}`} className="reject-button" disabled={!actor || decision.isPending} onClick={() => setRejecting(row.original)}><X size={15} /></button></div> : <span className="muted">{row.original.approvedBy ?? row.original.rejectedBy ?? "Recorded"}</span> },
  ], [actor, decision]);
  if (proposals.isLoading) return <LoadingState />;
  if (proposals.error) return <ErrorState error={proposals.error} />;
  const items = proposals.data ?? [];
  const pending = items.filter((item) => item.status === "PROPOSED").length;
  const approved = items.filter((item) => item.status === "APPROVED").length;
  const rebalanceId = items[0]?.rebalanceId;
  return <>
    <PageHeader eyebrow="Human approval gate" title="Proposal inbox" description="Approval changes state; it does not execute. Every proposal needs an explicit approve or reject decision before the approved subset can run." actions={<label className="actor-field"><span>Decision actor</span><input value={actor} onChange={(event) => setActor(event.target.value)} /></label>} />
    {!items.length ? <EmptyState title="No proposals awaiting workflow" message="Generate a rebalance from the paper account." /> : <section className="panel"><div className="approval-summary"><span><strong>{items.length}</strong> proposed</span><span><strong>{approved}</strong> approved</span><span><strong>{pending}</strong> undecided</span><button className="button primary" disabled={pending > 0 || !approved || !rebalanceId || execute.isPending} onClick={() => rebalanceId && execute.mutate(rebalanceId)}><Play size={15} /> Execute approved set</button></div><DataTable data={items} columns={columns} />{decision.error && <ErrorState error={decision.error} title="Decision was not persisted" />}{execute.error && <ErrorState error={execute.error} title="Execution refused" />}{execute.data && <div className="success-banner">Executed {execute.data.filled} approved fill{execute.data.filled === 1 ? "" : "s"}; {execute.data.rejected} rejected proposal{execute.data.rejected === 1 ? "" : "s"} remained outside execution.</div>}</section>}
    <Modal title={`Reject ${rejecting?.securityId ?? "proposal"}`} open={Boolean(rejecting)} onClose={() => setRejecting(null)} footer={<><button className="button secondary" onClick={() => setRejecting(null)}>Cancel</button><button className="button danger" disabled={!reason || decision.isPending} onClick={() => rejecting && decision.mutate({ orderId: rejecting.orderId, action: "REJECT", reason })}>Record rejection</button></>}><label className="field"><span>Reason</span><textarea value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Required audit reason" /></label></Modal>
  </>;
}
