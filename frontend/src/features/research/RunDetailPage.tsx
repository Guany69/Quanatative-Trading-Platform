import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertOctagon, LockKeyhole, RefreshCw } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import { Modal } from "../../components/ui/Modal";
import { PageHeader } from "../../components/ui/PageHeader";
import { ErrorState, LoadingState } from "../../components/ui/States";
import { StatusBadge } from "../../components/ui/StatusBadge";
import { formatDate } from "../../lib/format";
import { RunProgress } from "./RunProgress";
import {
  DiagnosticsTab,
  EquityTab,
  ModelsTab,
  OverviewTab,
  PredictionsTab,
  ReportsTab,
  ReproducibilityTab,
  StrategiesTab,
  TradesTab,
  WeightsTab,
} from "./RunResultTabs";

const tabs = [
  "Overview",
  "Models",
  "Strategies",
  "Equity",
  "Trades",
  "Predictions",
  "Portfolio / Weights",
  "Diagnostics / Stress",
  "Reproducibility",
  "Reports",
] as const;

type Tab = (typeof tabs)[number];

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<Tab>("Overview");
  const [holdoutOpen, setHoldoutOpen] = useState(false);
  const [actor, setActor] = useState("");
  const [holdoutModel, setHoldoutModel] = useState("factor_composite");
  const run = useQuery({
    queryKey: queryKeys.run(runId),
    queryFn: () => api.run(runId),
    enabled: Boolean(runId),
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state === "QUEUED" || state === "RUNNING" ? 1500 : false;
    },
  });
  const holdout = useMutation({
    mutationFn: () =>
      api.evaluateHoldout(runId, {
        model: holdoutModel,
        acknowledgedBy: actor,
        acknowledgeOneShotEvaluation: true,
      }),
    onSuccess: () => {
      setHoldoutOpen(false);
      queryClient.invalidateQueries({ queryKey: queryKeys.reports(runId) });
    },
  });

  if (run.isLoading) return <LoadingState label="Loading run identity…" />;
  if (run.error) return <ErrorState error={run.error} title="Run unavailable" />;
  const detail = run.data!;
  const terminal = detail.state === "COMPLETED" || detail.state === "FAILED";

  return (
    <>
      <PageHeader
        eyebrow="Research run"
        title={detail.runId}
        description={`Created ${formatDate(detail.createdAt)} · snapshot ${detail.snapshotId ?? "pending"}`}
        actions={
          <>
            <StatusBadge value={detail.state} />
            <button className="button secondary" onClick={() => run.refetch()}><RefreshCw size={15} /> Refresh</button>
            {detail.state === "COMPLETED" && (
              <button className="button danger-outline" onClick={() => setHoldoutOpen(true)}><LockKeyhole size={15} /> Evaluate holdout</button>
            )}
          </>
        }
      />

      {(detail.state === "QUEUED" || detail.state === "RUNNING") && <RunProgress run={detail} />}
      {detail.state === "FAILED" && (
        <section className="failure-banner" role="alert">
          <AlertOctagon size={22} />
          <div>
            <strong>{detail.failure?.code ?? "RUN_FAILED"} · {detail.failure?.stage ?? detail.stage}</strong>
            <p>{detail.failure?.message ?? "The backend retained no additional failure message."}</p>
          </div>
        </section>
      )}

      {terminal && (
        <>
          <div className="tab-strip">
            {tabs.map((item) => <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>{item}</button>)}
          </div>
          {tab === "Overview" && <OverviewTab run={detail} />}
          {tab === "Models" && <ModelsTab runId={runId} />}
          {tab === "Strategies" && <StrategiesTab runId={runId} />}
          {tab === "Equity" && <EquityTab runId={runId} run={detail} />}
          {tab === "Trades" && <TradesTab runId={runId} run={detail} />}
          {tab === "Predictions" && <PredictionsTab runId={runId} run={detail} />}
          {tab === "Portfolio / Weights" && <WeightsTab runId={runId} run={detail} />}
          {tab === "Diagnostics / Stress" && <DiagnosticsTab runId={runId} />}
          {tab === "Reproducibility" && <ReproducibilityTab runId={runId} run={detail} />}
          {tab === "Reports" && <ReportsTab runId={runId} />}
        </>
      )}
      {!terminal && detail.state === "QUEUED" && (
        <div className="queue-note">This job is durable and waiting for the sole research writer. <Link to="/research/runs">View queue</Link></div>
      )}

      <Modal
        title="Consume locked holdout"
        open={holdoutOpen}
        onClose={() => setHoldoutOpen(false)}
        footer={
          <>
            <button className="button secondary" onClick={() => setHoldoutOpen(false)}>Cancel</button>
            <button className="button danger" disabled={!actor || holdout.isPending} onClick={() => holdout.mutate()}>
              {holdout.isPending ? "Evaluating…" : "I understand — evaluate once"}
            </button>
          </>
        }
      >
        <div className="warning-copy">
          <LockKeyhole size={26} />
          <p>This is the locked, one-shot evaluation path. The backend will permanently record consumption and will refuse a second evaluation.</p>
        </div>
        <label className="field"><span>Model</span><select value={holdoutModel} onChange={(event) => setHoldoutModel(event.target.value)}>{(detail.models ?? []).map((model) => <option key={model}>{model}</option>)}</select></label>
        <label className="field"><span>Acknowledged by</span><input value={actor} onChange={(event) => setActor(event.target.value)} placeholder="Your name" /></label>
        {holdout.error && <ErrorState error={holdout.error} title="Holdout refused" />}
      </Modal>
    </>
  );
}
