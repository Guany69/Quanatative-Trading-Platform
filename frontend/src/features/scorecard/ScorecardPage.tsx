import { ExternalLink } from "lucide-react";

import { PageHeader } from "../../components/ui/PageHeader";

export function ScorecardPage() {
  return <><PageHeader eyebrow="Auxiliary analysis" title="Factor scorecard" description="The legacy live-ticker scorecard remains a separate descriptive tool outside the governed research and paper correctness boundary." /><section className="panel legacy-card"><div><span className="panel-kicker">INTENTIONAL DEFERMENT</span><h2>Core workflows migrated first</h2><p>The existing Python scorecard server still owns its analysis logic. It has not been duplicated in TypeScript; a later migration can expose its data as JSON and reuse the same backend computation.</p></div><a className="button secondary" href="http://127.0.0.1:8001" target="_blank" rel="noreferrer">Open legacy scorecard <ExternalLink size={15} /></a></section></>;
}
