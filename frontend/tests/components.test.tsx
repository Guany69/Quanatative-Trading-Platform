import { fireEvent, render, screen } from "@testing-library/react";
import type { ColumnDef } from "@tanstack/react-table";
import { describe, expect, it } from "vitest";

import { ApiClientError } from "../src/api/client";
import { DataTable } from "../src/components/ui/DataTable";
import { ErrorState } from "../src/components/ui/States";
import { StatusBadge } from "../src/components/ui/StatusBadge";

describe("shared workstation UI", () => {
  it("renders workflow status with a semantic tone", () => {
    const { container } = render(<StatusBadge value="FAILED" />);
    expect(screen.getByText("FAILED")).toBeVisible();
    expect(container.querySelector(".status-danger")).toBeInTheDocument();
  });

  it("surfaces the backend error code and actionable message", () => {
    render(<ErrorState error={new ApiClientError("HOLDOUT_VIOLATION", "Holdout was already consumed.", 409)} />);
    expect(screen.getByRole("alert")).toHaveTextContent("HOLDOUT_VIOLATION");
    expect(screen.getByRole("alert")).toHaveTextContent("Holdout was already consumed.");
  });

  it("sorts bounded page data without fetching unrelated result rows", () => {
    const columns: ColumnDef<{ name: string }>[] = [{ header: "Name", accessorKey: "name" }];
    render(<DataTable data={[{ name: "Zulu" }, { name: "Alpha" }]} columns={columns} />);
    fireEvent.click(screen.getByRole("button", { name: /Name/ }));
    const cells = screen.getAllByRole("cell");
    expect(cells.map((cell) => cell.textContent)).toEqual(["Alpha", "Zulu"]);
  });
});
