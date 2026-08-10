import { AlertTriangle, Database, LoaderCircle } from "lucide-react";
import type { ReactNode } from "react";

import { ApiClientError } from "../../api/client";

export function LoadingState({ label = "Loading authoritative data…" }: { label?: string }) {
  return (
    <div className="state-panel">
      <LoaderCircle className="spin" size={20} />
      <span>{label}</span>
    </div>
  );
}
export function EmptyState({
  title = "No records",
  message = "The backend returned an empty result for this view.",
  action,
}: {
  title?: string;
  message?: string;
  action?: ReactNode;
}) {
  return (
    <div className="state-panel state-empty">
      <Database size={22} />
      <strong>{title}</strong>
      <span>{message}</span>
      {action}
    </div>
  );
}

export function ErrorState({ error, title = "Request failed" }: { error: unknown; title?: string }) {
  const known = error instanceof ApiClientError;
  const message = error instanceof Error ? error.message : "An unexpected client error occurred.";
  return (
    <div className="state-panel state-error" role="alert">
      <AlertTriangle size={22} />
      <div>
        <strong>{title}</strong>
        {known && <code>{error.code}</code>}
        <p>{message}</p>
      </div>
    </div>
  );
}
