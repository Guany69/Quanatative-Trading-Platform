export function Pagination({
  page,
  totalPages,
  total,
  onPage,
}: {
  page: number;
  totalPages: number;
  total: number;
  onPage: (page: number) => void;
}) {
  if (!total) return null;
  return (
    <div className="pagination">
      <span>{total.toLocaleString()} records</span>
      <div>
        <button className="button secondary small" disabled={page <= 1} onClick={() => onPage(page - 1)}>
          Previous
        </button>
        <span>
          Page {page} / {Math.max(totalPages, 1)}
        </span>
        <button
          className="button secondary small"
          disabled={page >= totalPages}
          onClick={() => onPage(page + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
}
