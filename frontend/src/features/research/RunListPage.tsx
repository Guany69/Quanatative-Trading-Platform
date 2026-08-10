import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { Plus } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { api } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import type { Run } from "../../api/types";
import { DataTable } from "../../components/ui/DataTable";
import { PageHeader } from "../../components/ui/PageHeader";
import { Pagination } from "../../components/ui/Pagination";
import { ErrorState, LoadingState } from "../../components/ui/States";
import { StatusBadge } from "../../components/ui/StatusBadge";
import { formatDate, formatNumber, shortId } from "../../lib/format";

export function RunListPage() {
  const [params, setParams] = useSearchParams();
  const [page, setPage] = useState(1);
  const state = params.get("state") ?? "";
  const runs = useQuery({
    queryKey: queryKeys.runs(page, state),
    queryFn: () => api.runs(page, state),
    refetchInterval: (query) =>
      query.state.data?.items.some((run) => run.state === "RUNNING" || run.state === "QUEUED")
        ? 1500
        : false,
  });
  const columns = useMemo<ColumnDef<Run>[]>(
    () => [
      {
        header: "Run ID",
        accessorKey: "runId",
        cell: ({ row }) => <Link className="mono-link" to={`/research/runs/${row.original.runId}`}>{shortId(row.original.runId)}</Link>,
      },
      { header: "Created", accessorKey: "createdAt", cell: ({ getValue }) => formatDate(getValue<string>()) },
      { header: "Status", accessorKey: "state", cell: ({ getValue }) => <StatusBadge value={getValue<string>()} /> },
      { header: "Snapshot", accessorKey: "snapshotId", cell: ({ getValue }) => <code>{shortId(getValue<string>() ?? "—")}</code> },
      { header: "Models", accessorKey: "models", cell: ({ row }) => <span title={(row.original.models ?? []).join(", ")}>{row.original.models?.length ?? 0}</span> },
      { header: "Strategies", accessorKey: "strategies", cell: ({ row }) => row.original.strategies?.length ?? 0 },
      { header: "Duration", accessorKey: "durationSeconds", cell: ({ getValue }) => `${formatNumber(getValue<number>(), 1)}s` },
      { header: "Failure stage", accessorKey: "failure", cell: ({ row }) => row.original.failure?.stage ?? "—" },
    ],
    [],
  );

  return (
    <>
      <PageHeader
        eyebrow="Research archive"
        title="Run history"
        description="Paged run metadata only; large result tables remain behind purpose-specific APIs."
        actions={<Link className="button primary" to="/research/new"><Plus size={16} /> New run</Link>}
      />
      <section className="panel">
        <div className="toolbar">
          <div className="segmented" aria-label="Run status filter">
            {["", "RUNNING", "QUEUED", "COMPLETED", "FAILED"].map((value) => (
              <button
                key={value || "ALL"}
                className={state === value ? "active" : ""}
                onClick={() => {
                  setParams(value ? { state: value } : {});
                  setPage(1);
                }}
              >
                {value || "ALL"}
              </button>
            ))}
          </div>
        </div>
        {runs.isLoading ? (
          <LoadingState />
        ) : runs.error ? (
          <ErrorState error={runs.error} />
        ) : (
          <>
            <DataTable data={runs.data?.items ?? []} columns={columns} emptyMessage="No runs match this state." />
            <Pagination
              page={runs.data?.page ?? page}
              totalPages={runs.data?.totalPages ?? 0}
              total={runs.data?.total ?? 0}
              onPage={setPage}
            />
          </>
        )}
      </section>
    </>
  );
}
