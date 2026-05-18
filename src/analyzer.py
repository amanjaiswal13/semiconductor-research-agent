"""Autonomous analyzer for the semiconductor research agent.

This module sits on top of the scraped data layers (news, financials,
supply chain) and produces higher-order analytical *decisions*:

  1. DEEP-DIVE PRIORITY RANKING — composite score per company,
     deciding which name deserves the analyst's first hour today.
  2. TIERED ALERTS — Critical / Bullish / Watchlist, each gated by
     a different rule designed to catch a different kind of signal.
  3. EMERGING TREND DETECTION — pattern recognition across multiple
     companies in the same news category.

Every threshold and weight is declared as a named constant at the top
of this file so that the autonomous decision logic is *transparent
and tunable*. Inline comments explain WHY each value was chosen.

Reads:    data/news_YYYY-MM-DD.json, data/financials.csv, data/supply_chain.json
Writes:   output/analysis_YYYY-MM-DD.txt, logs/analyzer_YYYY-MM-DD.log
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

# =========================================================================== #
# DECISION THRESHOLDS — the autonomous brain is parameterized here.           #
# =========================================================================== #

# --- Composite scoring weights for the deep-dive ranking ------------------- #
#
# Each signal answers a different question:
#   news_volume       — what's *happening* RIGHT NOW (real-time pulse)
#   financial_change  — what's the *fundamental* trajectory (slow signal)
#   bottleneck        — what is *structurally* important (lasting risk)
#   constraint        — what is under *acute stress* (near-term hot zone)
#
# Weights must sum to 1.0. We weight `news_volume` and `bottleneck` highest
# (0.30 each) because together they capture the actionable cross-product:
# "something newsworthy is happening at a company that genuinely matters."
# Financial change at 0.25 because earnings cycles are slower-moving so the
# signal is noisier on any given day. Constraint at 0.15 because the supply-
# chain JSON only flags a handful of companies — it's high-precision but
# low-recall by design.
SCORING_WEIGHTS = {
    "news_volume":      0.30,
    "financial_change": 0.25,
    "bottleneck":       0.30,
    "constraint":       0.15,
}

# --- CRITICAL tier --------------------------------------------------------- #
# A company is CRITICAL if it's a structural bottleneck AND something
# negative is happening to it today. Rationale: a bottleneck in a quiet
# news cycle isn't actionable (we already know it's a pinch point), but
# a bottleneck experiencing live distress is the highest-signal event in
# the entire dataset — both the lever AND the spark are present.
#
# bottleneck_score ≥ 10 captures the 4–6 names that genuinely matter
# (ASML, TSMC, AppliedMaterials, Shin-Etsu...) while excluding long-tail
# suppliers. One negative-news article is enough to trigger — we want
# to surface, not bury.
CRITICAL_BOTTLENECK_MIN = 10.0
CRITICAL_NEG_NEWS_MIN = 1

# --- BULLISH tier ---------------------------------------------------------- #
# BULLISH = +15% QoQ revenue growth AND operating-margin expansion.
# Rationale: at multi-billion scale a single-quarter +15% is uncommon
# and almost always signals secular demand inflection. Pairing with
# margin expansion is the key filter — it excludes revenue growth that
# came from price cuts, channel-stuffing, or one-time effects. We only
# want *profitable* growth surfaced.
BULLISH_REV_QOQ_MIN = 15.0
BULLISH_REQUIRE_MARGIN_EXPANSION = True

# --- WATCHLIST tier -------------------------------------------------------- #
# WATCHLIST = at least one capex / fab-expansion mention. These don't
# move tomorrow's price, but capacity decisions resolve 2-4 quarters
# out and reshape the supply curve — so they belong on a longer-horizon
# watch list, not the urgent queue.
WATCHLIST_CAPEX_MIN = 1

# --- TREND detection ------------------------------------------------------- #
# A news category becomes a TREND when:
#   (a) ≥ 3 articles surface it in today's snapshot, AND
#   (b) ≥ 2 distinct supply-chain companies are named across those articles.
# The two-company rule prevents a single name's PR cycle from being mis-
# classified as "industry trend." Real trends touch the fabric — multiple
# unrelated players are caught up in the same wave.
TREND_ARTICLE_MIN = 3
TREND_DISTINCT_COMPANIES_MIN = 2


# =========================================================================== #
# Pattern families                                                            #
# =========================================================================== #
# Kept here (not imported from main.py) so analyzer.py is self-contained and
# can be reasoned about independently. Slight duplication is the cost of
# decoupling the autonomous-decision layer from the alerting layer.

NEGATIVE_NEWS_PATTERNS = [
    r"\bshortages?\b", r"\bdisruptions?\b", r"\bbottlenecks?\b",
    r"\b(?:halt|halted|halts)\b", r"\bdelay(?:s|ed)?\b",
    r"\bexport controls?\b", r"\bsanctions?\b", r"\btariffs?\b",
    r"\bban(?:ned|ning)?\b", r"\blayoffs?\b", r"\bbankruptcy\b",
    r"\bguidance (?:cut|lowered)\b", r"\b(?:recall|investigation)\b",
    r"\b(?:downgrade|downgraded)\b",
]

CAPEX_PATTERNS = [
    r"\bnew fab\b", r"\bnew plant\b", r"\bnew foundry\b",
    r"\b(?:groundbreaking|breaks? ground)\b",
    r"\bcapacity expansion\b",
    r"\b(?:building|builds) (?:a )?new (?:fab|plant|foundry)\b",
    r"\b(?:invest|allocates?|commits?)\s+(?:up to\s+)?(?:\$|US\$|USD|€)?\s?\d+\s*billion\b",
    r"\b\$\d+(?:\.\d+)?\s*(?:b|bn|billion)\b",
]

# News categories used for cross-company trend detection.
TREND_CATEGORIES: dict[str, list[str]] = {
    "supply_disruption": [
        r"\bshortages?\b", r"\bdisruptions?\b", r"\bbottlenecks?\b",
        r"\b(?:halt|halted|halts)\b",
    ],
    "geopolitical": [
        r"\bexport controls?\b", r"\bsanctions?\b", r"\btariffs?\b",
        r"\bban(?:ned|ning)? (?:on|of)\b",
    ],
    "capacity_expansion": [
        r"\bnew fab\b", r"\bnew plant\b", r"\bcapacity expansion\b",
        r"\bgigafab\b", r"\bcapacity ramp\b",
    ],
    "memory_supply": [
        r"\bHBM[1-4]?e?\b", r"\bmemory shortage\b",
    ],
    "geographic_risk": [
        r"\b(?:Taiwan|China|Korea)\b.*\b(?:tensions|risk|escalation)\b",
        r"\bStrait of\b",
    ],
    "ai_demand": [
        r"\bAI (?:demand|buildout|infrastructure)\b",
        r"\b(?:AI|hyperscaler) capex\b",
    ],
}


# =========================================================================== #
# Data loaders                                                                #
# =========================================================================== #

def load_supply_chain() -> dict:
    path = DATA_DIR / "supply_chain.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_news() -> list[dict]:
    path = DATA_DIR / f"news_{date.today():%Y-%m-%d}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("items", [])


def load_financial_series() -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Return {(company, metric): [(date, value), ...]} sorted ascending."""
    series: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    path = DATA_DIR / "financials.csv"
    if not path.exists():
        return series
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["value"])
            except ValueError:
                continue
            series[(row["company"], row["metric"])].append((row["date"], v))
    for key in series:
        series[key].sort()
    return series


# =========================================================================== #
# Per-company signal helpers                                                  #
# =========================================================================== #

def _compile_company_pattern(name: str) -> re.Pattern[str]:
    """Word-boundary, case-insensitive regex for a company name.

    Word boundaries on both ends prevent false positives like 'apples'
    matching 'Apple' or 'intelligence' matching 'Intel' (the trailing
    'l' is a word char so \\bIntel\\b will not match 'intelligence').
    """
    return re.compile(rf"\b{re.escape(name)}\b", re.I)


def count_company_mentions(news_items: list[dict], company: str) -> tuple[int, list[dict]]:
    pat = _compile_company_pattern(company)
    matched = []
    for item in news_items:
        text = (item.get("headline") or "") + " " + (item.get("summary") or "")
        if pat.search(text):
            matched.append(item)
    return len(matched), matched


def is_negative_news(item: dict) -> bool:
    text = ((item.get("headline") or "") + " " + (item.get("summary") or "")).lower()
    return any(re.search(p, text) for p in NEGATIVE_NEWS_PATTERNS)


def is_capex_news(item: dict) -> bool:
    text = ((item.get("headline") or "") + " " + (item.get("summary") or "")).lower()
    return any(re.search(p, text) for p in CAPEX_PATTERNS)


def latest_qoq_revenue(series, company: str) -> tuple[float, float, str, str] | None:
    """Return (pct_change, current_value, prev_date, cur_date) or None."""
    pts = series.get((company, "quarterly_revenue"), [])
    if len(pts) < 2:
        return None
    (prev_date, prev_v), (cur_date, cur_v) = pts[-2], pts[-1]
    if prev_v == 0:
        return None
    return (cur_v - prev_v) / prev_v * 100.0, cur_v, prev_date, cur_date


def margin_expanded_qoq(series, company: str) -> bool | None:
    """True iff operating margin moved up vs the prior quarter."""
    pts = series.get((company, "quarterly_operating_margin"), [])
    if len(pts) < 2:
        return None
    return pts[-1][1] > pts[-2][1]


# =========================================================================== #
# Component 1 — Deep-dive priority ranking                                    #
# =========================================================================== #

@dataclass
class CompanySignals:
    company: str
    news_count: int
    fin_change_pct: float        # |QoQ %|; 0 when no data
    bottleneck_score: float
    constraint_flag: int         # 0 or 1
    composite_score: float = 0.0
    matched_news: list[dict] = field(default_factory=list)
    negative_news: list[dict] = field(default_factory=list)


def _normalize(values: list[float]) -> list[float]:
    """Min-max scale to [0, 1]. If everything is equal, return zeros.

    Decision: normalization is *within the cohort*, not against absolute
    scale. This means a company's rank reflects its position *today*
    relative to its peers, not against an arbitrary baseline. Tomorrow's
    ranking will reorder organically as the inputs shift.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def prioritize_companies(supply_chain: dict, news: list[dict], series) -> list[CompanySignals]:
    """Score and rank every company in the graph on four normalized signals."""
    companies = list(supply_chain.get("companies", {}).keys())
    bottleneck_lookup = {
        b["company"]: b["weighted_bottleneck_score"]
        for b in supply_chain.get("analysis", {}).get("bottleneck_ranking", [])
    }
    constrained = {c["company"] for c in supply_chain.get("capacity_constraints", [])}

    signals: list[CompanySignals] = []
    for c in companies:
        count, items = count_company_mentions(news, c)
        neg = [i for i in items if is_negative_news(i)]
        fin = latest_qoq_revenue(series, c)
        signals.append(CompanySignals(
            company=c,
            news_count=count,
            fin_change_pct=abs(fin[0]) if fin else 0.0,
            bottleneck_score=bottleneck_lookup.get(c, 0.0),
            constraint_flag=1 if c in constrained else 0,
            matched_news=items,
            negative_news=neg,
        ))

    nv = _normalize([s.news_count       for s in signals])
    fv = _normalize([s.fin_change_pct   for s in signals])
    bv = _normalize([s.bottleneck_score for s in signals])
    cv = [float(s.constraint_flag)      for s in signals]  # already 0/1

    w = SCORING_WEIGHTS
    for i, s in enumerate(signals):
        s.composite_score = 100.0 * (
            nv[i] * w["news_volume"] +
            fv[i] * w["financial_change"] +
            bv[i] * w["bottleneck"] +
            cv[i] * w["constraint"]
        )

    signals.sort(key=lambda s: s.composite_score, reverse=True)
    return signals


# =========================================================================== #
# Component 2 — Tiered alerts                                                 #
# =========================================================================== #

@dataclass
class TieredAlert:
    tier: str                # "CRITICAL" | "BULLISH" | "WATCHLIST"
    company: str
    title: str
    evidence: list[str]
    reasoning: str


def detect_critical(signals: list[CompanySignals]) -> list[TieredAlert]:
    """Surface the (rare) intersection of structural bottleneck + acute distress.

    This is the only tier that gates on TWO independent conditions.
    The reason: structural risk alone is not actionable, and acute news
    alone may not propagate widely. Their conjunction is what creates
    asymmetric investment-relevant signal.
    """
    alerts: list[TieredAlert] = []
    for s in signals:
        if s.bottleneck_score < CRITICAL_BOTTLENECK_MIN:
            continue
        if len(s.negative_news) < CRITICAL_NEG_NEWS_MIN:
            continue
        evidence = [n["headline"] for n in s.negative_news[:3]]
        alerts.append(TieredAlert(
            tier="CRITICAL",
            company=s.company,
            title=f"{s.company}: structural bottleneck under live stress",
            evidence=evidence,
            reasoning=(
                f"Decision: bottleneck_score={s.bottleneck_score:.1f} ≥ "
                f"{CRITICAL_BOTTLENECK_MIN} (structural lever) AND "
                f"{len(s.negative_news)} negative-news article(s) today "
                f"(acute spark, threshold ≥ {CRITICAL_NEG_NEWS_MIN}). "
                f"Recommended action: deep-dive within 24h."
            ),
        ))
    return alerts


def detect_bullish(series, supply_chain: dict) -> list[TieredAlert]:
    """Surface profitable growth — rev surge confirmed by margin expansion."""
    alerts: list[TieredAlert] = []
    for c in supply_chain.get("companies", {}).keys():
        rev = latest_qoq_revenue(series, c)
        if rev is None:
            continue
        pct, cur_v, prev_date, cur_date = rev
        if pct < BULLISH_REV_QOQ_MIN:
            continue

        margin_ok = margin_expanded_qoq(series, c)
        if BULLISH_REQUIRE_MARGIN_EXPANSION and margin_ok is not True:
            # margin_ok being None (no data) blocks the BULLISH classification —
            # we don't want to "trust" a rev surge without margin confirmation.
            continue

        op_pts = series.get((c, "quarterly_operating_margin"), [])
        margin_evidence = ""
        if len(op_pts) >= 2:
            delta = op_pts[-1][1] - op_pts[-2][1]
            margin_evidence = (
                f"op margin {op_pts[-2][1]:.1f}% → {op_pts[-1][1]:.1f}% "
                f"({delta:+.1f} pp)"
            )

        alerts.append(TieredAlert(
            tier="BULLISH",
            company=c,
            title=f"{c}: profitable growth surge",
            evidence=[
                f"revenue {prev_date} → {cur_date}: {pct:+.1f}% QoQ "
                f"(latest {cur_v:.2f})",
                margin_evidence,
            ],
            reasoning=(
                f"Decision: QoQ revenue {pct:+.1f}% ≥ {BULLISH_REV_QOQ_MIN}% "
                f"AND operating margin expanded. Margin filter excludes "
                f"unprofitable growth (price cuts, one-offs)."
            ),
        ))
    return alerts


def detect_watchlist(signals: list[CompanySignals]) -> list[TieredAlert]:
    """Surface long-horizon capex / capacity expansion signals."""
    alerts: list[TieredAlert] = []
    for s in signals:
        capex_items = [n for n in s.matched_news if is_capex_news(n)]
        if len(capex_items) < WATCHLIST_CAPEX_MIN:
            continue
        evidence = [n["headline"] for n in capex_items[:3]]
        alerts.append(TieredAlert(
            tier="WATCHLIST",
            company=s.company,
            title=f"{s.company}: capex / expansion signal",
            evidence=evidence,
            reasoning=(
                f"Decision: {len(capex_items)} capex-tagged article(s) today "
                f"(threshold ≥ {WATCHLIST_CAPEX_MIN}). "
                f"Capacity decisions resolve 2-4 quarters out — track for "
                f"ramp-vs-delay signals."
            ),
        ))
    return alerts


# =========================================================================== #
# Component 3 — Cross-company trend detection                                 #
# =========================================================================== #

@dataclass
class Trend:
    category: str
    article_count: int
    companies: list[str]
    sample_headlines: list[str]
    reasoning: str


def detect_trends(news: list[dict], supply_chain: dict) -> list[Trend]:
    """Detect categories where multiple unrelated companies converge."""
    companies = list(supply_chain.get("companies", {}).keys())
    co_patterns = [(c, _compile_company_pattern(c)) for c in companies]

    trends: list[Trend] = []
    for category, patterns in TREND_CATEGORIES.items():
        compiled = [re.compile(p, re.I) for p in patterns]
        matched_items: list[dict] = []
        companies_seen: set[str] = set()
        for item in news:
            text = (item.get("headline") or "") + " " + (item.get("summary") or "")
            if any(p.search(text) for p in compiled):
                matched_items.append(item)
                for cname, cpat in co_patterns:
                    if cpat.search(text):
                        companies_seen.add(cname)

        if (len(matched_items) >= TREND_ARTICLE_MIN
                and len(companies_seen) >= TREND_DISTINCT_COMPANIES_MIN):
            trends.append(Trend(
                category=category,
                article_count=len(matched_items),
                companies=sorted(companies_seen),
                sample_headlines=[m["headline"] for m in matched_items[:5]],
                reasoning=(
                    f"Decision: {len(matched_items)} article(s) matched the "
                    f"'{category}' pattern family (threshold ≥ {TREND_ARTICLE_MIN}) "
                    f"AND {len(companies_seen)} distinct graph companies named "
                    f"(threshold ≥ {TREND_DISTINCT_COMPANIES_MIN}). "
                    f"Cross-company spread implies industry pattern, not "
                    f"single-name news cycle."
                ),
            ))

    trends.sort(key=lambda t: t.article_count, reverse=True)
    return trends


# =========================================================================== #
# Report rendering                                                            #
# =========================================================================== #

def render_report(signals: list[CompanySignals],
                  criticals: list[TieredAlert],
                  bullish: list[TieredAlert],
                  watchlist: list[TieredAlert],
                  trends: list[Trend]) -> str:
    bar = "=" * 80
    sub = "-" * 80
    out: list[str] = []
    out.append(bar)
    out.append("  AUTONOMOUS ANALYZER — SEMICONDUCTOR RESEARCH AGENT")
    out.append(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.append(bar)
    out.append("")

    # ----- Priority ranking ----- #
    out.append("## DEEP-DIVE PRIORITY RANKING (top 10)")
    out.append(sub)
    out.append(f"{'#':<3} {'Company':<20} {'Score':>6} {'News':>5} "
               f"{'|ΔRev%|':>8} {'Bottlnk':>8} {'Constr':>7}")
    for i, s in enumerate(signals[:10], 1):
        out.append(
            f"{i:<3} {s.company:<20} {s.composite_score:>6.1f} "
            f"{s.news_count:>5} {s.fin_change_pct:>8.1f} "
            f"{s.bottleneck_score:>8.1f} "
            f"{'YES' if s.constraint_flag else '-':>7}"
        )
    if signals:
        top = signals[0]
        out.append("")
        out.append(f"  Top pick: {top.company} (score {top.composite_score:.1f})")
        rationale = []
        if top.news_count > 0:
            rationale.append(f"{top.news_count} news mention(s) today")
        if top.fin_change_pct > 0:
            rationale.append(f"|ΔRev| {top.fin_change_pct:.1f}% QoQ")
        if top.bottleneck_score > 0:
            rationale.append(f"bottleneck score {top.bottleneck_score:.1f}")
        if top.constraint_flag:
            rationale.append("named in capacity constraints")
        out.append(f"  Drivers: {', '.join(rationale) if rationale else 'none'}")
    out.append("")

    # ----- Tiered alerts ----- #
    out.append("## TIERED ALERTS")
    out.append(sub)
    for tier_name, alerts in (
        ("CRITICAL", criticals),
        ("BULLISH",  bullish),
        ("WATCHLIST", watchlist),
    ):
        out.append(f"### {tier_name} ({len(alerts)})")
        if not alerts:
            out.append(f"  No companies met {tier_name} criteria today.")
        for a in alerts:
            out.append(f"  • {a.title}")
            for e in a.evidence:
                if e:
                    out.append(f"      - {e}")
            out.append(f"      Reasoning: {a.reasoning}")
        out.append("")

    # ----- Trends ----- #
    out.append("## EMERGING TRENDS")
    out.append(sub)
    if not trends:
        out.append("  No category cleared both trend thresholds today.")
    else:
        for i, t in enumerate(trends, 1):
            out.append(f"  [T{i}] {t.category.upper().replace('_', ' ')}")
            out.append(f"        Articles : {t.article_count}")
            out.append(f"        Companies: {', '.join(t.companies)}")
            for h in t.sample_headlines[:3]:
                out.append(f"        • {h}")
            out.append(f"        Reasoning: {t.reasoning}")
            out.append("")

    # ----- Decision logic transparency ----- #
    out.append("## DECISION LOGIC (active this run)")
    out.append(sub)
    out.append(f"  Scoring weights      : {SCORING_WEIGHTS}")
    out.append(f"  CRITICAL threshold   : bottleneck ≥ {CRITICAL_BOTTLENECK_MIN} "
               f"AND ≥ {CRITICAL_NEG_NEWS_MIN} negative-news article(s)")
    out.append(f"  BULLISH threshold    : QoQ revenue ≥ {BULLISH_REV_QOQ_MIN}% "
               f"AND operating-margin expansion")
    out.append(f"  WATCHLIST threshold  : ≥ {WATCHLIST_CAPEX_MIN} capex / expansion article(s)")
    out.append(f"  TREND threshold      : ≥ {TREND_ARTICLE_MIN} articles AND "
               f"≥ {TREND_DISTINCT_COMPANIES_MIN} distinct companies named")
    out.append("")
    out.append(bar)
    out.append("  END OF ANALYSIS")
    out.append(bar)
    return "\n".join(out) + "\n"


# =========================================================================== #
# Entrypoint                                                                  #
# =========================================================================== #

def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"analyzer_{date.today():%Y-%m-%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    configure_logging()
    logging.info("Autonomous analyzer starting")

    sc = load_supply_chain()
    if not sc:
        logging.error("Missing data/supply_chain.json — run main.py (or "
                      "src/supply_chain_mapper.py) first.")
        return 2
    news = load_news()
    series = load_financial_series()
    logging.info("Inputs loaded: %d news items, %d financial series, %d companies",
                 len(news), len(series), len(sc.get("companies", {})))

    signals    = prioritize_companies(sc, news, series)
    criticals  = detect_critical(signals)
    bullish    = detect_bullish(series, sc)
    watchlist  = detect_watchlist(signals)
    trends     = detect_trends(news, sc)
    logging.info("Decisions: %d CRITICAL, %d BULLISH, %d WATCHLIST, %d TRENDS",
                 len(criticals), len(bullish), len(watchlist), len(trends))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"analysis_{date.today():%Y-%m-%d}.txt"
    report = render_report(signals, criticals, bullish, watchlist, trends)
    report_path.write_text(report, encoding="utf-8")
    logging.info("Wrote %s", report_path)

    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
