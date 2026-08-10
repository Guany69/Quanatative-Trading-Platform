import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Cpu, Database, FlaskConical, Layers3 } from "lucide-react";
import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { z } from "zod";

import { api } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import { PageHeader } from "../../components/ui/PageHeader";
import { ErrorState, LoadingState } from "../../components/ui/States";

const runSchema = z.object({
  snapshotId: z.string().min(1),
  models: z.array(z.string()).min(1, "Select at least one model."),
  strategies: z.array(z.string()).min(1, "Select at least one strategy."),
  costScenarios: z.array(z.string()).min(1, "Select at least one cost scenario."),
  portfolioModel: z.string().min(1),
  maxWorkers: z.number().int().min(1).max(64),
  fixtureSecurities: z.number().int().min(20).max(2000),
});

type RunForm = z.infer<typeof runSchema>;

export function NewRunPage() {
  const navigate = useNavigate();
  const metadata = useQuery({ queryKey: queryKeys.metadata, queryFn: api.metadata });
  const snapshots = useQuery({ queryKey: queryKeys.snapshots, queryFn: api.snapshots });
  const form = useForm<RunForm>({
    resolver: zodResolver(runSchema),
    defaultValues: {
      snapshotId: "auto",
      models: ["no_skill", "momentum_baseline", "factor_composite"],
      strategies: ["equal_weight"],
      costScenarios: ["base"],
      portfolioModel: "factor_composite",
      maxWorkers: 4,
      fixtureSecurities: 120,
    },
  });
  const selectedModels = form.watch("models");

  useEffect(() => {
    if (selectedModels.length && !selectedModels.includes(form.getValues("portfolioModel"))) {
      form.setValue("portfolioModel", selectedModels[0] ?? "");
    }
  }, [form, selectedModels]);

  const create = useMutation({
    mutationFn: api.createRun,
    onSuccess: (result) => navigate(`/research/runs/${result.runId}`),
  });

  if (metadata.isLoading || snapshots.isLoading) return <LoadingState label="Loading backend inventories…" />;
  if (metadata.error) return <ErrorState error={metadata.error} />;
  if (snapshots.error) return <ErrorState error={snapshots.error} />;
  const meta = metadata.data!;

  return (
    <>
      <PageHeader
        eyebrow="Governed experiment"
        title="Configure research run"
        description="Selections come from backend registries. The POST queues work and returns immediately; Python remains authoritative for validation."
      />
      <form className="run-form" onSubmit={form.handleSubmit((values) => create.mutate(values))}>
        <section className="panel form-section">
          <div className="section-heading">
            <Database size={19} />
            <div><h2>Data snapshot</h2><p>Immutable point-in-time input identity.</p></div>
          </div>
          <label className="field">
            <span>Snapshot</span>
            <select {...form.register("snapshotId")}>
              <option value="auto">Auto / fixture snapshot</option>
              {(snapshots.data ?? []).map((snapshot) => (
                <option key={snapshot.snapshotId} value={snapshot.snapshotId}>
                  {snapshot.snapshotId} · {snapshot.sources.join(", ")}
                </option>
              ))}
            </select>
            <small>Filesystem paths are never accepted by this form or API.</small>
          </label>
        </section>

        <section className="panel form-section">
          <div className="section-heading">
            <FlaskConical size={19} />
            <div><h2>Forecasting models</h2><p>Controls and challengers share the same governed fold path.</p></div>
          </div>
          <div className="choice-grid">
            {meta.models.map((model) => (
              <label className="choice-card" key={model.id}>
                <input type="checkbox" value={model.id} {...form.register("models")} />
                <span>
                  <strong>{model.name}</strong>
                  <small>{model.productionEligible ? "Production eligible" : "Diagnostic / control"}</small>
                </span>
              </label>
            ))}
          </div>
          {form.formState.errors.models && <p className="field-error">{form.formState.errors.models.message}</p>}
          <label className="field inline-field">
            <span>Portfolio signal model</span>
            <select {...form.register("portfolioModel")}>
              {selectedModels.map((model) => <option key={model} value={model}>{model}</option>)}
            </select>
          </label>
        </section>

        <section className="panel form-section">
          <div className="section-heading">
            <Layers3 size={19} />
            <div><h2>Portfolio comparison</h2><p>All selected variants consume the same persisted prediction set.</p></div>
          </div>
          <div className="choice-grid two-col">
            {meta.strategies.map((strategy) => (
              <label className="choice-card" key={strategy.id}>
                <input type="checkbox" value={strategy.id} {...form.register("strategies")} />
                <span><strong>{strategy.name}</strong><small>{strategy.description}</small></span>
              </label>
            ))}
          </div>
          <div className="subsection-label">COST SCENARIOS</div>
          <div className="choice-row">
            {meta.costScenarios.map((scenario) => (
              <label className="chip-choice" key={scenario.id}>
                <input type="checkbox" value={scenario.id} {...form.register("costScenarios")} />
                <span>{scenario.name}</span>
              </label>
            ))}
          </div>
        </section>

        <section className="panel form-section">
          <div className="section-heading">
            <Cpu size={19} />
            <div><h2>Execution resources</h2><p>Model workers compute in parallel; one parent remains the sole DuckDB writer.</p></div>
          </div>
          <div className="form-grid">
            <label className="field"><span>Maximum workers</span><input type="number" {...form.register("maxWorkers", { valueAsNumber: true })} /></label>
            <label className="field"><span>Fixture securities</span><input type="number" {...form.register("fixtureSecurities", { valueAsNumber: true })} /></label>
          </div>
        </section>

        {create.error && <ErrorState error={create.error} title="Run was not queued" />}
        <div className="form-actions">
          <span>Submission creates a durable FIFO job.</span>
          <button className="button primary" type="submit" disabled={create.isPending}>
            {create.isPending ? "Queuing…" : "Queue research run"}
          </button>
        </div>
      </form>
    </>
  );
}
