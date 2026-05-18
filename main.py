"""Autonomous orchestrator for the semiconductor research agent.

Pipeline:
    1. News scraper          (src/news_scraper.py)
    2. Financial scraper     (src/financial_scraper.py)
    3. Supply-chain mapper   (src/supply_chain_mapper.py)
    4. Analysis (autonomous decisions): financial Δ, capacity constraints,
       graph-aware news keyword scan
    5. Daily report          (output/daily_report_YYYY-MM-DD.txt)

The orchestrator is *resumable*: each scrape step records completion in
output/.state.json. Re-running after a crash skips already-completed
steps and retries the one that failed. Analysis + report regenerate on
every run from whatever cached data is available.

Designed to run as:
    python main.py                  # full run; resumable
    python main.py --force          # ignore state and rescrape everything
    python main.py --no-scrape      # rebuild report from cached data only

The --no-scrape mode is what a scheduler (Task Scheduler / cron) would
call between scrape windows. Plain `python main.py` works for ad-hoc use.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"
STATE_PATH = OUTPUT_DIR / ".state.json"

sys.path.insert(0, str(SRC_DIR))  # make src/ modules importable

# --------------------------------------------------------------------------- #
# Decision thresholds — codified as constants so every autonomous decision is #
# inspectable and tunable in one place.                                       #
# --------------------------------------------------------------------------- #

REVENUE_PCT_ALERT = 10.0       # |QoQ % change| ≥ this → alert
REVENUE_PCT_HIGH_SEV = 20.0    # |QoQ % change| ≥ this → escalate to HIGH

# Keyword patterns: (regex, category, base_severity)
FAB_PATTERNS = [
    (r"\bnew fab\b",                                 "fab_announcement", "high"),
    (r"\bnew plant\b",                               "fab_announcement", "high"),
    (r"\bnew foundry\b",                             "fab_announcement", "high"),
    (r"\b(?:groundbreaking|breaks? ground)\b",       "fab_announcement", "high"),
    (r"\bcapacity expansion\b",                      "fab_announcement", "high"),
    (r"\b(?:building|builds) (?:a )?new (?:fab|plant|foundry)\b",
                                                     "fab_announcement", "high"),
    (r"\bcapacity ramp\b",                           "fab_announcement", "medium"),
    (r"\bgigafab\b",                                 "fab_announcement", "medium"),
]

DISRUPTION_PATTERNS = [
    (r"\bshortages?\b",                              "supply_disruption", "high"),
    (r"\bdisruptions?\b",                            "supply_disruption", "high"),
    (r"\bbottlenecks?\b",                            "supply_disruption", "high"),
    (r"\b(?:halts?|halted) (?:production|shipments)\b",
                                                     "supply_disruption", "high"),
    (r"\bsupply chain (?:crisis|risk|shock)\b",      "supply_disruption", "high"),
    (r"\bexport controls?\b",                        "geopolitical",      "high"),
    (r"\bsanctions?\b",                              "geopolitical",      "high"),
    (r"\btariffs?\b",                                "geopolitical",      "medium"),
    (r"\b(?:earthquake|flood|fire|typhoon)\b",       "supply_disruption", "medium"),
    (r"\b(?:delay|delayed|delays)\b",                "supply_disruption", "medium"),
]

GENERAL_PATTERNS = [
    (r"\bacqui(?:re|sition|res|red|ring)\b",         "corporate",  "medium"),
    (r"\bmerger\b",                                  "corporate",  "medium"),
    (r"\bguidance (?:cut|raise|raised|lowered)\b",   "corporate",  "medium"),
    (r"\bbankruptcy\b",                              "corporate",  "high"),
    (r"\blayoffs?\b",                                "corporate",  "medium"),
    (r"\bbreakthrough\b",                            "innovation", "low"),
]


# --------------------------------------------------------------------------- #
# Data classes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Alert:
    severity: str    # "HIGH" | "MEDIUM" | "LOW"
    category: str
    title: str
    evidence: str
    reasoning: str   # the explicit decision rationale shown in the report
    source: str = ""


@dataclass
class State:
    """Per-day completion tracking. Enables resumability across crashes."""
    today: str
    steps_completed: dict[str, list[str]] = field(default_factory=dict)
    last_persisted_at: str = ""

    @classmethod
    def load(cls) -> "State":
        today = date.today().isoformat()
        if not STATE_PATH.exists():
            return cls(today=today)
        try:
            raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logging.warning("State file unreadable (%s); starting fresh", exc)
            return cls(today=today)
        return cls(
            today=today,
            steps_completed=raw.get("steps_completed", {}),
            last_persisted_at=raw.get("last_persisted_at", ""),
        )

    def is_done_today(self, step: str) -> bool:
        return step in self.steps_completed.get(self.today, [])

    def mark_done(self, step: str) -> None:
        self.steps_completed.setdefault(self.today, [])
        if step not in self.steps_completed[self.today]:
            self.steps_completed[self.today].append(step)

    def persist(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "today": self.today,
            "steps_completed": self.steps_completed,
            "last_persisted_at": datetime.now().isoformat(),
        }
        STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"orchestrator_{date.today():%Y-%m-%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# --------------------------------------------------------------------------- #
# Step runner with resumability                                               #
# --------------------------------------------------------------------------- #

def run_step(name: str, fn: Callable[[], int], state: State, *, force: bool) -> bool:
    """Run a pipeline step idempotently.

    Autonomous decision: if a step already completed today (per state file),
    skip it on rerun. This makes the orchestrator crash-safe — re-invoking
    after a failure picks up exactly where it left off. Only --force or a
    new calendar day reruns completed steps.
    """
    if state.is_done_today(name) and not force:
        logging.info("Step '%s' already completed today — skipping", name)
        return True

    logging.info("Step '%s' starting", name)
    try:
        code = fn()
    except Exception:
        logging.exception("Step '%s' raised", name)
        return False
    if code != 0:
        logging.error("Step '%s' returned non-zero exit code %d", name, code)
        return False

    state.mark_done(name)
    state.persist()  # persist between every step → resumable mid-pipeline
    logging.info("Step '%s' completed", name)
    return True


def run_news() -> int:
    import news_scraper
    return news_scraper.main()


def run_financial() -> int:
    import financial_scraper
    return financial_scraper.main()


def run_supply_chain() -> int:
    import supply_chain_mapper
    return supply_chain_mapper.main()


def run_analyzer() -> int:
    """Higher-order autonomous decisions (priority ranking, tiered alerts,
    cross-company trends). Reads data/, writes output/analysis_YYYY-MM-DD.txt.
    """
    import analyzer
    return analyzer.main()


# --------------------------------------------------------------------------- #
# Autonomous analyzers — turn raw scraped data into ranked alerts             #
# --------------------------------------------------------------------------- #

def analyze_financials() -> list[Alert]:
    """Decision logic: for each company in data/financials.csv, compare the
    most recent two quarterly_revenue points. If |Δ%| meets thresholds,
    raise an alert. Both directions matter — a surge may signal capacity
    breakthrough; a drop may signal demand weakness.
    """
    alerts: list[Alert] = []
    csv_path = DATA_DIR / "financials.csv"
    if not csv_path.exists():
        logging.info("analyze_financials: no financials.csv yet — skipping")
        return alerts

    series: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["metric"] != "quarterly_revenue":
                continue
            try:
                v = float(row["value"])
            except ValueError:
                continue
            series[row["company"]].append((row["date"], v, row.get("unit", "")))

    for company, points in series.items():
        points.sort()  # chronological
        if len(points) < 2:
            continue
        (prev_date, prev_v, unit), (cur_date, cur_v, _) = points[-2], points[-1]
        if prev_v == 0:
            continue
        pct = (cur_v - prev_v) / prev_v * 100.0
        if abs(pct) < REVENUE_PCT_ALERT:
            continue
        direction = "rose" if pct > 0 else "fell"
        sev = "HIGH" if abs(pct) >= REVENUE_PCT_HIGH_SEV else "MEDIUM"
        alerts.append(Alert(
            severity=sev,
            category="revenue_change",
            title=f"{company} quarterly revenue {direction} {pct:+.1f}% QoQ",
            evidence=f"{prev_date}: {prev_v:.2f} {unit} → {cur_date}: {cur_v:.2f} {unit}",
            reasoning=(
                f"Decision: |Δ| = {abs(pct):.1f}% ≥ {REVENUE_PCT_ALERT}% alert threshold; "
                f"severity {sev} because |Δ| "
                f"{'≥' if abs(pct) >= REVENUE_PCT_HIGH_SEV else '<'} "
                f"{REVENUE_PCT_HIGH_SEV}% high-severity threshold."
            ),
            source=f"data/financials.csv ({company})",
        ))
    return alerts


def analyze_supply_chain() -> list[Alert]:
    """Decision logic: surface every entry in capacity_constraints as an
    alert (these are curated known-bad conditions). Also lift the top-3
    structural bottlenecks as LOW-severity reminders so daily readers stay
    aware of pinch points even when no acute event occurs.
    """
    alerts: list[Alert] = []
    sc_path = DATA_DIR / "supply_chain.json"
    if not sc_path.exists():
        logging.info("analyze_supply_chain: no supply_chain.json yet — skipping")
        return alerts
    sc = json.loads(sc_path.read_text(encoding="utf-8"))

    for c in sc.get("capacity_constraints", []):
        status = c.get("status", "").lower()
        # Severity map: phrase-based. "severely / sold out / halt" is acute;
        # "constrained / tight / yield" is chronic; otherwise informational.
        if any(t in status for t in ("severe", "sold out", "halt")):
            sev = "HIGH"
        elif any(t in status for t in ("constrained", "tight", "yield", "ramping")):
            sev = "MEDIUM"
        else:
            sev = "LOW"
        impact = c.get("downstream_impact", [])
        alerts.append(Alert(
            severity=sev,
            category="capacity_constraint",
            title=f"{c['company']}: {c['resource']} — {c['status']}",
            evidence=c.get("notes", ""),
            reasoning=(
                f"Decision: status keyword in '{c['status']}' maps to {sev}; "
                f"propagates to {len(impact)} downstream(s): "
                f"{', '.join(impact) if impact else 'unspecified'}."
            ),
            source=f"data/supply_chain.json (as_of {c.get('as_of', 'n/a')})",
        ))

    for b in sc.get("analysis", {}).get("bottleneck_ranking", [])[:3]:
        alerts.append(Alert(
            severity="LOW",
            category="structural_bottleneck",
            title=f"Structural bottleneck: {b['company']} (score {b['weighted_bottleneck_score']})",
            evidence=(
                f"{b['downstream_count']} downstream(s); "
                f"sole-source for {b['sole_source_relationships']}"
            ),
            reasoning=(
                "Decision: persistent graph-derived risk surfaced as LOW so it "
                "stays visible even on quiet news days."
            ),
            source="data/supply_chain.json (analysis.bottleneck_ranking)",
        ))
    return alerts


def analyze_news(known_companies: set[str]) -> tuple[list[Alert], dict]:
    """Decision logic: diff today's news snapshot against the most recent
    prior snapshot, then scan *only the new items* against three pattern
    families (fab announcements, supply disruptions, general market).

    Graph-aware escalation: if a matched headline mentions a company that
    appears in the supply-chain graph, escalate severity by one step
    (capped at HIGH). This is the autonomous link from unstructured news
    to structured graph entities.
    """
    stats = {"new_items_vs_prior": 0, "total_items_today": 0, "matched_items": 0}
    alerts: list[Alert] = []

    today_path = DATA_DIR / f"news_{date.today():%Y-%m-%d}.json"
    if not today_path.exists():
        logging.info("analyze_news: no news_%s.json yet — skipping", date.today())
        return alerts, stats

    today_payload = json.loads(today_path.read_text(encoding="utf-8"))
    today_items = today_payload.get("items", [])
    stats["total_items_today"] = len(today_items)

    # Diff against most recent *prior* snapshot. On the first run there's no
    # prior file → treat all of today as new.
    prior_snapshots = sorted(p for p in DATA_DIR.glob("news_*.json") if p != today_path)
    if prior_snapshots:
        prior_payload = json.loads(prior_snapshots[-1].read_text(encoding="utf-8"))
        prior_urls = {i["url"] for i in prior_payload.get("items", [])}
        new_items = [i for i in today_items if i["url"] not in prior_urls]
        logging.info("analyze_news: diffing against %s (%d items)",
                     prior_snapshots[-1].name, len(prior_urls))
    else:
        new_items = today_items
        logging.info("analyze_news: no prior snapshot — treating all %d items as new",
                     len(today_items))
    stats["new_items_vs_prior"] = len(new_items)

    all_patterns = (
        [(p, c, s, "fab")        for (p, c, s) in FAB_PATTERNS] +
        [(p, c, s, "disruption") for (p, c, s) in DISRUPTION_PATTERNS] +
        [(p, c, s, "general")    for (p, c, s) in GENERAL_PATTERNS]
    )

    severity_rank = {"low": 0, "medium": 1, "high": 2}
    severity_label = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    company_aliases = {c.lower(): c for c in known_companies}

    seen_titles: set[str] = set()
    for item in new_items:
        text = f"{item.get('headline', '')} {item.get('summary', '')}".lower()

        # Pick the highest-severity pattern that matched.
        best: tuple[int, str, str, str] | None = None
        for pat, cat, sev_hint, family in all_patterns:
            if re.search(pat, text):
                sev_i = severity_rank[sev_hint]
                if best is None or sev_i > best[0]:
                    best = (sev_i, cat, pat, family)
        if best is None:
            continue

        base_sev_i, cat, pat, family = best

        # Graph-aware escalation: cross-reference against supply-chain entities.
        graph_hits = [orig for low, orig in company_aliases.items() if low in text]
        final_sev_i = base_sev_i
        escalation = ""
        if graph_hits and base_sev_i < 2:
            final_sev_i = base_sev_i + 1
            escalation = (
                f" Escalated +1 (base={severity_label[base_sev_i]}) because the "
                f"headline references graph entity/entities: "
                f"{', '.join(sorted(set(graph_hits))[:3])}."
            )

        title = item.get("headline", "(no title)")
        if title in seen_titles:
            continue
        seen_titles.add(title)

        alerts.append(Alert(
            severity=severity_label[final_sev_i],
            category=cat,
            title=title,
            evidence=f"source={item.get('source', '?')} | date={item.get('date', '?')}",
            reasoning=f"Decision: matched /{pat}/ (family={family}).{escalation}",
            source=item.get("url", "")[:140],
        ))
        stats["matched_items"] += 1
    return alerts, stats


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #

def generate_report(alerts: list[Alert], news_stats: dict, state: State,
                    ran_scrapers: bool) -> str:
    sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    alerts_sorted = sorted(alerts, key=lambda a: (sev_order[a.severity], a.category, a.title))
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for a in alerts:
        counts[a.severity] += 1

    lines: list[str] = []
    bar = "=" * 80
    sub = "-" * 80
    lines.append(bar)
    lines.append("  SEMICONDUCTOR SUPPLY CHAIN RESEARCH AGENT — DAILY REPORT")
    lines.append(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(bar)
    lines.append("")
    lines.append("## RUN SUMMARY")
    lines.append(f"  Mode                     : {'full pipeline' if ran_scrapers else 'analysis-only (--no-scrape)'}")
    lines.append(f"  Steps completed today    : {', '.join(state.steps_completed.get(state.today, [])) or '(none)'}")
    lines.append(f"  News articles today      : {news_stats.get('total_items_today', 0)}")
    lines.append(f"  New since last snapshot  : {news_stats.get('new_items_vs_prior', 0)}")
    lines.append(f"  News matches             : {news_stats.get('matched_items', 0)}")
    lines.append(f"  Alerts total             : {len(alerts)}  "
                 f"(HIGH={counts['HIGH']}  MEDIUM={counts['MEDIUM']}  LOW={counts['LOW']})")
    lines.append("")

    for sev in ("HIGH", "MEDIUM", "LOW"):
        bucket = [a for a in alerts_sorted if a.severity == sev]
        if not bucket:
            continue
        lines.append(f"## {sev} SEVERITY ({len(bucket)})")
        lines.append(sub)
        for i, a in enumerate(bucket, 1):
            lines.append(f"[{sev[0]}{i}] {a.category.upper()} — {a.title}")
            if a.evidence:
                lines.append(f"     Evidence : {a.evidence}")
            lines.append(f"     Reasoning: {a.reasoning}")
            if a.source:
                lines.append(f"     Source   : {a.source}")
            lines.append("")
        lines.append("")

    if not alerts:
        lines.append("## NO ALERTS")
        lines.append("  No data triggered any decision rule on this run.")
        lines.append("")

    lines.append(bar)
    lines.append(f"  END OF REPORT — written to "
                 f"output/daily_report_{date.today():%Y-%m-%d}.txt")
    lines.append(bar)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> int:
    configure_logging()
    logging.info("Orchestrator starting (force=%s, no_scrape=%s)",
                 args.force, args.no_scrape)
    state = State.load()

    scrape_ok = True
    if not args.no_scrape:
        steps: list[tuple[str, Callable[[], int]]] = [
            ("news",          run_news),
            ("financial",     run_financial),
            ("supply_chain",  run_supply_chain),
        ]
        for name, fn in steps:
            if not run_step(name, fn, state, force=args.force):
                scrape_ok = False
                # Decision: keep going — analysis can still emit useful alerts
                # from whatever data we have. Better to ship a partial report
                # than nothing on a flaky network day.
                logging.warning("Continuing pipeline despite '%s' failure", name)
    else:
        logging.info("--no-scrape set; skipping scrape steps")

    # ---- Analysis phase: always runs, even after partial scrape failure ---- #
    logging.info("Analysis phase starting")

    known_companies: set[str] = set()
    sc_path = DATA_DIR / "supply_chain.json"
    if sc_path.exists():
        sc = json.loads(sc_path.read_text(encoding="utf-8"))
        known_companies = set(sc.get("companies", {}).keys())

    fin_alerts = analyze_financials()
    sc_alerts = analyze_supply_chain()
    news_alerts, news_stats = analyze_news(known_companies)
    alerts = fin_alerts + sc_alerts + news_alerts

    logging.info(
        "Analysis: %d alerts total (financial=%d, supply_chain=%d, news=%d)",
        len(alerts), len(fin_alerts), len(sc_alerts), len(news_alerts),
    )

    # ---- Write report ----------------------------------------------------- #
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(alerts, news_stats, state, ran_scrapers=not args.no_scrape)
    report_path = OUTPUT_DIR / f"daily_report_{date.today():%Y-%m-%d}.txt"
    report_path.write_text(report, encoding="utf-8")
    logging.info("Wrote report to %s", report_path)

    state.mark_done("report")
    state.persist()

    # ---- Higher-order autonomous analysis (analyzer.py) ------------------- #
    # Runs as a gated step so it's resumable and skippable on rerun, but soft-
    # fails: an analyzer crash should not poison the overall exit code, because
    # the daily report above is the must-have artifact and the analysis_*.txt
    # is the bonus higher-order layer.
    if not state.is_done_today("analyzer") or args.force:
        logging.info("Step 'analyzer' starting")
        try:
            code = run_analyzer()
            if code == 0:
                state.mark_done("analyzer")
                state.persist()
                logging.info("Step 'analyzer' completed")
            else:
                logging.warning("Analyzer returned exit %d; continuing anyway", code)
        except Exception:
            logging.exception("Analyzer crashed; continuing anyway")
    else:
        logging.info("Step 'analyzer' already completed today — skipping")

    # Echo report header + HIGH section so the operator sees the highlights
    # inline without having to open the file.
    head_lines = report.splitlines()
    high_start = next((i for i, ln in enumerate(head_lines) if ln.startswith("## HIGH ")), None)
    if high_start is not None:
        # show the summary block + the entire HIGH section
        end = next((i for i in range(high_start + 1, len(head_lines))
                    if head_lines[i].startswith("## ")), len(head_lines))
        print("\n".join(head_lines[: end]))
    else:
        print("\n".join(head_lines[:20]))
    print(f"\n(Full report: {report_path})")

    return 0 if scrape_ok else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autonomous semiconductor research agent orchestrator.",
    )
    p.add_argument("--force", action="store_true",
                   help="Rerun every step even if marked complete in today's state.")
    p.add_argument("--no-scrape", action="store_true",
                   help="Skip scrapers; regenerate the report from cached data.")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
