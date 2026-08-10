const tones: Record<string, string> = {
  COMPLETED: "success",
  APPROVED: "success",
  EXECUTED: "success",
  RUNNING: "info",
  QUEUED: "neutral",
  PROPOSED: "warning",
  AWAITING_DECISIONS: "warning",
  FAILED: "danger",
  REJECTED: "danger",
};

export function StatusBadge({ value }: { value: string }) {
  return <span className={`status status-${tones[value] ?? "neutral"}`}>{value}</span>;
}
