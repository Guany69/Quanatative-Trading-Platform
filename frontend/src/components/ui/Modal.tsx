import { X } from "lucide-react";
import type { PropsWithChildren, ReactNode } from "react";

export function Modal({
  title,
  open,
  onClose,
  children,
  footer,
}: PropsWithChildren<{
  title: string;
  open: boolean;
  onClose: () => void;
  footer?: ReactNode;
}>) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="modal" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <h2>{title}</h2>
          <button className="icon-button" aria-label="Close" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-footer">{footer}</div>}
      </div>
    </div>
  );
}
