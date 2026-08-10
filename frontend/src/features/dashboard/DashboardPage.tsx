import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, Clock3, Database, FlaskConical, Wallet } from "lucide-react";
import { Link } from "react-router-dom";

import { api, ApiClientError } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import { MetricCard } from "../../components/ui/MetricCard";
import { PageHeader } from "../../components/ui/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "../../components/ui/States";
import { StatusBadge } from "../../components/ui/StatusBadge";
import { formatCurrency, formatDate, shortId } from "../../lib/format";

export function DashboardPage() {
  const health = useQuery({ queryKey: queryKeys.health, queryFn: api.health });
  const runs = useQuery({ queryKey: queryKeys.runs(1), queryFn: () => api.runs(1) });
  const paper = useQuery({ queryKey: queryKeys.paper, queryFn: api.paper, retry: false });

  if (health.isLoading || runs.isLoading) return <LoadingState label="Opening workstation…" />;
  if (health.error) return <ErrorState error={health.error} title="Backend unavailable" />;
  if (runs.error) return <ErrorState error={runs.error} />;

  const runItems = runs.data?.items ?? [];
  const active = runItems.find((run) => run.state === "RUNNING" || run.state === "QUEUED");
  const completed = runItems.filter((run) => run.state === "COMPLETED");
  const failed = runItems.filter((run) => run.state === "FAILED");
  const paperMissing = paper.error instanceof ApiClientError && paper.error.code === "PAPER_STATE_NOT_INITIALIZED";

  return (
    <>
      <PageHeader
        eyebrow="Operations overview"
        title="Research control room"
        description="Authoritative run state, paper risk, and recent workflow events from the local Python application."
        actions={<Link className="button primary" to="/research/new">Start research run</Link>}
      />

      <section className="metric-grid">
        <MetricCard
          label="Backend"
          value="ONLINE"
          detail="FastAPI / loopback"
          icon={<CheckCircle2 size={18} />}
          tone="positive"
        />
        <MetricCard
          label="Active queue"
          value={active ? active.state : "IDLE"}
          detail={active ? shortId(active.runId) : "No writer active"}
          icon={<Clock3 size={18} />}
          tone={active ? "warning" : "default"}
        />
        <MetricCard
          label="Completed runs"
          value={completed.length}
          detail={`${failed.length} recent failure${failed.length === 1 ? "" : "s"}`}
          icon={<FlaskConical size={18} />}
        />
        <MetricCard
          label="Paper equity"
          value={paper.data ? formatCurrency(paper.data.totalValue) : "NOT INITIALIZED"}
          detail={paper.data ? `${paper.data.pendingApprovals} pending approvals` : "Create an account to begin"}
          icon={<Wallet size={18} />}
          tone={paper.data?.pendingApprovals ? "warning" : "default"}
        />
      </section>

      <div className="dashboard-grid">
        <section className="panel span-2">
          <div className="panel-header">
            <div>
              <span className="panel-kicker">RECENT ACTIVITY</span>
              <h2>Research runs</h2>
            </div>
            <Link to="/research/runs">View all</Link>
          </div>
          {!runItems.length ? (
            <EmptyState
              title="No research runs yet"
              message="The results database is ready. Submit the first governed experiment."
              action={<Link className="button secondary" to="/research/new">Configure run</Link>}
            />
          ) : (
            <div className="activity-list">
              {runItems.slice(0, 7).map((run) => (
                <Link to={`/research/runs/${run.runId}`} className="activity-row" key={run.runId}>
                  <div className="run-glyph"><Database size={16} /></div>
                  <div className="activity-main">
                    <strong>{run.runId}</strong>
                    <span>{(run.models ?? []).join(", ") || "Metadata pending"}</span>
                  </div>
                  <div className="activity-time">{formatDate(run.createdAt)}</div>
                  <StatusBadge value={run.state} />
                </Link>
              ))}
            </div>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <span className="panel-kicker">PAPER ACCOUNT</span>
              <h2>Forward state</h2>
            </div>
          </div>
          {paper.isLoading ? (
            <LoadingState />
          ) : paper.data ? (
            <div className="paper-snapshot">
              <div className="big-number">{formatCurrency(paper.data.totalValue)}</div>
              <div className="split-stat">
                <span>Cash <strong>{formatCurrency(paper.data.cash)}</strong></span>
                <span>Positions <strong>{paper.data.nPositions}</strong></span>
              </div>
              <div className="production-ref">
                <span>Production model</span>
                <code>
                  {paper.data.productionModel
                    ? `${shortId(paper.data.productionModel.runId)}:${paper.data.productionModel.model}`
                    : "NOT SET"}
                </code>
              </div>
              <Link className="button secondary full" to="/paper">Open paper account</Link>
            </div>
          ) : paperMissing ? (
            <EmptyState
              title="Paper state not initialized"
              message="Initialization is explicit and creates the authoritative forward account."
              action={<Link className="button secondary" to="/paper">Initialize</Link>}
            />
          ) : (
            <ErrorState error={paper.error} />
          )}
        </section>

        {failed.length > 0 && (
          <section className="panel span-3 alert-panel">
            <AlertTriangle size={19} />
            <div>
              <strong>{failed.length} recent failed run{failed.length === 1 ? "" : "s"}</strong>
              <span>{failed[0]?.failure?.message ?? "Inspect run history for retained failure causes."}</span>
            </div>
            <Link to="/research/runs?state=FAILED">Inspect failures</Link>
          </section>
        )}
      </div>
    </>
  );
}
