"""Local web UI for the factor scorecard.

Serves a small browser interface on **127.0.0.1 only**. The bind address is deliberate and
not configurable to a wider interface: this tool downloads and renders market data on your
behalf and has no authentication, so there is no good reason for it to be reachable from
anything but the machine it runs on.

Built on the standard library's ``http.server`` rather than FastAPI or Flask. That keeps the
core dependency-free (ADR 0011) and avoids pulling a web framework into a research platform
for what is a single form and a results page. It is single-threaded and intended for one
local user; it is not a production server and does not pretend to be.

Every page carries the same caveats as the CLI. A prettier presentation must not become a
more confident one -- the scorecard is descriptive, and the HTML says so as plainly as the
terminal does.
"""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from quant_platform.analysis import (
    COMPOSITE_EXPLANATION,
    DEFAULT_PEER_UNIVERSE,
    FAMILY_EXPLANATIONS,
    Scorecard,
    analyze_tickers,
    explain_factor,
    interpret_score,
)
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("web")

# Bound to loopback on purpose. See the module docstring.
HOST = "127.0.0.1"
DEFAULT_PORT = 8000

_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1000px; margin: 0 auto; padding: 2rem 1.25rem 4rem;
  line-height: 1.5; color: #1a1a1a; background: #fbfbfc;
}
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
.sub { color: #666; font-size: .9rem; margin-bottom: 1.5rem; }
form { background: #fff; border: 1px solid #e2e2e6; border-radius: 10px; padding: 1.25rem;
       margin-bottom: 1.5rem; }
label { display: block; font-weight: 600; font-size: .85rem; margin-bottom: .35rem; }
input[type=text] { width: 100%; padding: .6rem .7rem; font-size: 1rem; border-radius: 7px;
                   border: 1px solid #ccc; font-family: ui-monospace, SFMono-Regular, monospace; }
.hint { color: #777; font-size: .78rem; margin: .3rem 0 .9rem; }
button { background: #1a56db; color: #fff; border: 0; border-radius: 7px;
         padding: .6rem 1.4rem; font-size: .95rem; font-weight: 600; cursor: pointer; }
button:hover { background: #1444b0; }
.card { background: #fff; border: 1px solid #e2e2e6; border-radius: 10px;
        padding: 1.25rem 1.4rem; margin-bottom: 1.25rem; }
.card h2 { margin: 0; font-size: 1.25rem; }
.meta { color: #666; font-size: .85rem; margin: .2rem 0 1rem; }
.composite { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 8px;
             padding: .8rem 1rem; margin-bottom: 1.2rem; }
.composite .big { font-size: 1.5rem; font-weight: 700; }
.family { margin: 1.1rem 0 .3rem; font-size: .78rem; font-weight: 700; letter-spacing: .07em;
          text-transform: uppercase; color: #444; border-bottom: 1px solid #eee;
          padding-bottom: .3rem; }
.row { display: grid; grid-template-columns: 1fr 130px 62px; gap: .6rem; align-items: center;
       padding: .3rem 0; font-size: .87rem; }
.desc { color: #333; }
.raw { color: #999; font-size: .74rem; font-family: ui-monospace, monospace; }
.bar { background: #ececf1; border-radius: 4px; height: 9px; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 4px; }
.pct { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600;
       font-size: .82rem; }
.warn { background: #fff8e1; border: 1px solid #ffe28a; border-radius: 8px;
        padding: .9rem 1.1rem; font-size: .84rem; margin-bottom: 1.25rem; }
.warn strong { display: block; margin-bottom: .35rem; }
.warn ul { margin: .3rem 0 0; padding-left: 1.15rem; }
.na { color: #888; font-size: .84rem; }
.family-note { font-size: .8rem; color: #666; margin: 0 0 .6rem; line-height: 1.45; }
.plain { font-size: .8rem; color: #1a56db; font-weight: 600; margin-top: .1rem; }
.explain { margin: 0 0 .5rem; font-size: .8rem; }
.explain summary { cursor: pointer; color: #777; font-size: .75rem; padding: .1rem 0; }
.explain summary:hover { color: #1a56db; }
.explain p { margin: .35rem 0; color: #555; line-height: 1.5;
             border-left: 2px solid #e0e0e6; padding-left: .7rem; }
table { width: 100%; border-collapse: collapse; font-size: .87rem; }
th, td { padding: .45rem .5rem; text-align: right; border-bottom: 1px solid #eee; }
th:first-child, td:first-child { text-align: left; font-weight: 600; }
.err { background: #fee; border: 1px solid #fbb; border-radius: 8px; padding: 1rem; }
footer { margin-top: 2.5rem; color: #777; font-size: .78rem; border-top: 1px solid #e5e5e5;
         padding-top: 1rem; }
@media (prefers-color-scheme: dark) {
  body { background: #16171a; color: #e6e6e8; }
  form, .card { background: #1e1f24; border-color: #33343a; }
  input[type=text] { background: #16171a; color: #e6e6e8; border-color: #44454c; }
  .composite { background: #1e2749; border-color: #33407a; }
  .bar { background: #33343a; }
  .warn { background: #2e2712; border-color: #5c4d1a; color: #f0e2b8; }
  .family { color: #b8b8c0; border-color: #33343a; }
  .desc { color: #d5d5da; }
  th, td { border-color: #2c2d33; }
  .err { background: #3a1d1d; border-color: #6b2c2c; }
  .family-note { color: #a0a0a8; }
  .plain { color: #7aa2f7; }
  .explain p { color: #b0b0b8; border-color: #3a3b42; }
  .explain summary { color: #9a9aa2; }
}
"""


def _bar_color(percentile: float) -> str:
    """Colour a bar by strength.

    Deliberately a neutral blue-to-grey scale rather than red/green: green would read as
    "buy", and this scorecard makes no such claim.
    """
    if percentile >= 80:
        return "#1a56db"
    if percentile >= 60:
        return "#4a7fe0"
    if percentile >= 40:
        return "#93a4c4"
    if percentile >= 20:
        return "#b8bcc6"
    return "#d0d2d8"


def _page(body: str, title: str = "Factor Scorecard") -> bytes:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{_STYLE}</style></head>
<body>{body}
<footer>
  quant-platform &middot; running locally on {HOST} &middot; descriptive factor standings only.<br>
  <strong>Not investment advice, not a forecast, and not a recommendation to buy or sell
  anything.</strong>
</footer>
</body></html>""".encode()


def _form(tickers: str = "", peers: str = "") -> str:
    return f"""
<h1>Factor Scorecard</h1>
<div class="sub">Where a stock sits relative to its peers on momentum, reversal,
defensive, and liquidity factors. Real market data via Yahoo Finance.</div>
<form method="get" action="/analyze">
  <label for="tickers">Tickers</label>
  <input type="text" id="tickers" name="tickers" value="{html.escape(tickers)}"
         placeholder="AAPL,NVDA,KO" autofocus>
  <div class="hint">Comma-separated. Try <code>AAPL,MSFT,GOOGL,AMZN,META</code></div>
  <label for="peers">Peer universe (optional)</label>
  <input type="text" id="peers" name="peers" value="{html.escape(peers)}"
         placeholder="leave blank for the default {len(DEFAULT_PEER_UNIVERSE)} large caps">
  <div class="hint">Ranking is only as meaningful as the comparison group.
      Leave blank to use the built-in large-cap list.</div>
  <button type="submit">Analyze</button>
</form>
"""


def _caveats(card: Scorecard) -> str:
    unavailable = "".join(
        f"<li><strong>{html.escape(p)}</strong>: {html.escape(r)}</li>"
        for p, r in card.unavailable_pillars.items()
    )
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in card.warnings)
    return f"""
<div class="warn">
  <strong>What this does not tell you</strong>
  <ul>{unavailable}{warnings}</ul>
</div>"""


def _live_meta(card: Scorecard) -> str:
    """Card subtitle: live quote when available, last close otherwise."""
    if card.live_price is not None and card.quote_time is not None:
        change = ""
        if card.live_change_pct is not None and card.live_change_pct == card.live_change_pct:
            arrow = "&#9650;" if card.live_change_pct >= 0 else "&#9660;"
            colour = "#1a7f37" if card.live_change_pct >= 0 else "#b42318"
            change = (
                f' <span style="color:{colour};font-weight:600">{arrow} '
                f"{card.live_change_pct:+.2%} today</span>"
            )
        session = (
            "factors include today&rsquo;s partial session"
            if card.includes_live_session
            else "market closed &middot; latest completed session"
        )
        return (
            f'<div class="meta"><strong>LIVE ${card.live_price:,.2f}</strong>{change} '
            f"&middot; quote {card.quote_time:%H:%M} (may be ~15 min delayed) "
            f"&middot; {session} &middot; ranked against {card.peer_count} peers</div>"
        )
    return (
        f'<div class="meta">${card.price:,.2f} &middot; as of {card.as_of} &middot; '
        f"ranked against {card.peer_count} peers</div>"
    )


def _render_card(card: Scorecard) -> str:
    parts = [
        '<div class="card">',
        f"<h2>{html.escape(card.ticker)}</h2>",
        _live_meta(card),
    ]

    if card.composite_percentile == card.composite_percentile:  # not NaN
        parts.append(
            f'<div class="composite"><span class="big">'
            f"{card.composite_percentile:.0f}<span style='font-size:.9rem'>th percentile</span>"
            f"</span> &middot; #{card.composite_rank} of {card.peer_count + 1}"
            f'<div style="font-size:.8rem;color:#666;margin-top:.3rem">'
            f"{html.escape(COMPOSITE_EXPLANATION)}</div></div>"
        )

    for family, scores in card.factors.items():
        family_pct = card.family_percentiles.get(family, float("nan"))
        parts.append(
            f'<div class="family">{html.escape(family)} &mdash; {family_pct:.0f}th percentile</div>'
        )
        # What this family of factors actually captures, in plain language.
        blurb = FAMILY_EXPLANATIONS.get(family)
        if blurb:
            parts.append(f'<div class="family-note">{html.escape(blurb)}</div>')
        for s in scores:
            explanation = explain_factor(s.name)
            plain = ""
            if explanation and explanation.read_value:
                try:
                    plain = explanation.read_value(s.raw_value)
                except Exception:  # a display helper must never break the page
                    plain = ""
            parts.append(
                f'<div class="row">'
                f'<div><div class="desc">{html.escape(s.description)}</div>'
                + (f'<div class="plain">{html.escape(plain)}</div>' if plain else "")
                + f'<div class="raw">raw {s.raw_value:+.4f} &middot; '
                f"peer median {s.peer_median:+.4f}</div></div>"
                f'<div class="bar"><span style="width:{max(s.percentile, 1):.0f}%;'
                f'background:{_bar_color(s.percentile)}"></span></div>'
                f'<div class="pct">{s.percentile:.0f}%</div>'
                f"</div>"
            )
            if explanation:
                reading = interpret_score(s)
                parts.append(
                    '<details class="explain"><summary>what does this mean?</summary>'
                    f"<p><strong>Measures:</strong> {html.escape(explanation.measures)}</p>"
                    f"<p><strong>Reading:</strong> {html.escape(reading)} "
                    f"It sits at the {s.percentile:.0f}th percentile, meaning it scores higher "
                    f"than {s.percentile:.0f}% of the peer group on this measure.</p>"
                    "</details>"
                )

    parts.append(_caveats(card))
    parts.append("</div>")
    return "".join(parts)


def _render_comparison(cards: list[Scorecard]) -> str:
    if len(cards) < 2:
        return ""
    families = sorted({f for c in cards for f in c.family_percentiles})
    header = "".join(f"<th>{html.escape(f)}</th>" for f in families)
    rows = []
    for c in sorted(cards, key=lambda x: -x.composite_percentile):
        cells = "".join(
            f"<td>{c.family_percentiles.get(f, float('nan')):.0f}</td>" for f in families
        )
        rows.append(
            f"<tr><td>{html.escape(c.ticker)}</td><td>${c.price:,.2f}</td>"
            f"<td>{c.composite_percentile:.0f}</td><td>#{c.composite_rank}</td>{cells}</tr>"
        )
    return f"""
<div class="card"><h2>Comparison</h2>
<div class="meta">All values are percentiles against the peer group.</div>
<table><thead><tr><th>Ticker</th><th>Price</th><th>Composite</th><th>Rank</th>
{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>"""


class ScorecardHandler(BaseHTTPRequestHandler):
    """Request handler for the local scorecard UI."""

    server_version = "quant-platform"

    def log_message(self, fmt: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, payload: bytes, status: int = 200, content_type: str = "text/html") -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self._send(_page(_form()))
            return
        if parsed.path == "/health":
            self._send(json.dumps({"status": "ok"}).encode(), content_type="application/json")
            return
        if parsed.path != "/analyze":
            self._send(_page('<div class="err">Not found.</div>'), status=404)
            return

        raw_tickers = (params.get("tickers") or [""])[0]
        raw_peers = (params.get("peers") or [""])[0]
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

        if not tickers:
            self._send(
                _page(_form() + '<div class="err">Enter at least one ticker.</div>'),
                status=400,
            )
            return
        if len(tickers) > 10:
            self._send(
                _page(
                    _form(raw_tickers, raw_peers)
                    + '<div class="err">Please request 10 tickers or fewer at a time.</div>'
                ),
                status=400,
            )
            return

        peers = [p.strip().upper() for p in raw_peers.split(",") if p.strip()] or None

        try:
            cards, failures = analyze_tickers(tickers, peers)
        except Exception as exc:
            logger.exception("analysis failed")
            detail = html.escape(f"{type(exc).__name__}: {exc}")
            self._send(
                _page(
                    _form(raw_tickers, raw_peers)
                    + f'<div class="err"><strong>Analysis failed.</strong><br>{detail}'
                    f"<br><br>This needs network access to download prices.</div>"
                ),
                status=500,
            )
            return

        body = [_form(raw_tickers, raw_peers)]
        for failure in failures:
            body.append(f'<div class="err">{html.escape(failure)}</div>')
        body.append(_render_comparison(cards))
        body.extend(_render_card(c) for c in cards)

        if not cards:
            body.append('<div class="err">No tickers could be scored.</div>')

        title = f"{', '.join(tickers)} - Factor Scorecard"
        self._send(_page("".join(body), title))


def serve(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Start the local server. Blocks until interrupted."""
    try:
        httpd = HTTPServer((HOST, port), ScorecardHandler)
    except OSError as exc:
        raise RuntimeError(
            f"could not bind {HOST}:{port} ({exc}). Another process may be using that port; "
            f"try --port {port + 1}."
        ) from exc

    url = f"http://{HOST}:{port}"
    logger.info("serving on %s (loopback only)", url)

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        httpd.server_close()
