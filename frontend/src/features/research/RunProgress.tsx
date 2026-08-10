import { Check, Circle, LoaderCircle, X } from "lucide-react";

import type { Run } from "../../api/types";

const stages = [
  ["INITIALIZATION", "Initialization"],
  ["PIT_CACHE", "Point-in-time cache"],
  ["MODEL_TRAINING", "Model training"],
  ["PORTFOLIO_SIMULATION", "Portfolio simulation"],
  ["OVERFITTING_STRESS", "Overfitting & stress"],
  ["REPORTING", "Reporting"],
] as const;

export function RunProgress({ run }: { run: Run }) {
  const current = run.stage ? stages.findIndex(([key]) => key === run.stage) : -1;
  return (
    <section className="panel progress-panel">
      <div className="panel-header">
        <div><span className="panel-kicker">EXECUTION PIPELINE</span><h2>Real orchestrator stages</h2></div>
        <span className="stage-count">{run.completedStages} / {run.totalStages} stages</span>
      </div>
      <div className="stage-list">
        {stages.map(([key, label], index) => {
          const complete = run.state === "COMPLETED" || index < current;
          const active = index === current && run.state === "RUNNING";
          const failed = index === current && run.state === "FAILED";
          return (
            <div className={`stage ${complete ? "complete" : ""} ${active ? "current" : ""} ${failed ? "failed" : ""}`} key={key}>
              <div className="stage-icon">
                {complete ? <Check size={15} /> : active ? <LoaderCircle className="spin" size={15} /> : failed ? <X size={15} /> : <Circle size={13} />}
              </div>
              <div><strong>{label}</strong><span>{active ? "In progress" : complete ? "Complete" : failed ? "Failed here" : "Waiting"}</span></div>
            </div>
          );
        })}
      </div>
      <p className="progress-note">No elapsed-time percentage is estimated. This view advances only when Python persists a stage transition.</p>
    </section>
  );
}
