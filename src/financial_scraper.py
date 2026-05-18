"""Phase 2 financial scraper for the semiconductor research agent.

Targets:
  * TSMC monthly revenue                (investor.tsmc.com/english/monthly-revenue)
  * TSMC quarterly revenue + margins    (investor.tsmc.com/english/quarterly-results)
  * ASML quarterly financials + system
    shipment volume                     (asml.com press releases)

Output: data/financials.csv with columns
    date, company, metric, value, unit, source_url

Rows are deduplicated on (date, company, metric) across runs so the
file grows into a historical series. Each run pulls several quarters /
months back so we get historical comparison for free.

NOTE on EUV vs DUV: ASML's HTML press releases now report
"new lithography systems sold" as a single aggregate. The explicit
EUV / DUV unit split is only published in their PDF factsheet. The
HTML aggregate is captured here under `new_lithography_systems_sold`;
the EUV / DUV split would require a PDF parser (next phase).
"""

from __future__ import annotations

import csv
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
CSV_PATH = DATA_DIR / "financials.csv"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_BASE = 1.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

CSV_COLUMNS = ["date", "company", "metric", "value", "unit", "source_url"]

# How far back to walk for historical comparison
TSMC_QUARTERS_HISTORY = 4
TSMC_MONTHLY_YEARS_HISTORY = 2   # current year + 1 prior
ASML_PRESS_RELEASES = 3


@dataclass(frozen=True)
class FinancialRow:
    date: str          # YYYY-MM-DD (period end)
    company: str
    metric: str
    value: str
    unit: str = ""
    source_url: str = ""


# --------------------------------------------------------------------------- #
# Infrastructure                                                              #
# --------------------------------------------------------------------------- #

def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"financial_scraper_{datetime.now():%Y-%m-%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    )
    return s


def fetch(url: str, session: requests.Session) -> requests.Response | None:
    """GET with retry + exponential backoff. Returns None on permanent failure."""
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last = exc
            logging.warning("attempt %d/%d GET %s -> %s", attempt, MAX_RETRIES, url, exc)
        else:
            if r.status_code < 400:
                return r
            if r.status_code == 429 or 500 <= r.status_code < 600:
                logging.warning(
                    "attempt %d/%d GET %s -> HTTP %d (retryable)",
                    attempt, MAX_RETRIES, url, r.status_code,
                )
            else:
                logging.error("GET %s -> HTTP %d (non-retryable)", url, r.status_code)
                return None
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
    logging.error("GET %s: exhausted retries (last error: %s)", url, last)
    return None


def _to_number(s: str) -> str | None:
    """Strip commas/units and return a clean numeric string, or None."""
    if not s:
        return None
    cleaned = s.replace(",", "").strip()
    m = re.match(r"^-?\d+(?:\.\d+)?", cleaned)
    return m.group(0) if m else None


# --------------------------------------------------------------------------- #
# TSMC monthly revenue                                                        #
# --------------------------------------------------------------------------- #

TSMC_BASE = "https://investor.tsmc.com"

MONTH_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _tsmc_monthly_url(year: int, current_year: int) -> str:
    if year == current_year:
        return f"{TSMC_BASE}/english/monthly-revenue"
    return f"{TSMC_BASE}/english/monthly-revenue/{year}"


def scrape_tsmc_monthly(session: requests.Session, years: list[int]) -> list[FinancialRow]:
    rows: list[FinancialRow] = []
    current_year = date.today().year
    for year in years:
        url = _tsmc_monthly_url(year, current_year)
        resp = fetch(url, session)
        if resp is None:
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        tbl = soup.find("table")
        if tbl is None:
            logging.warning("TSMC monthly %d: no table found", year)
            continue
        added = 0
        for tr in tbl.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            month_token = cells[0].lower().rstrip(".")
            if month_token not in MONTH_TO_NUM:
                continue
            month = MONTH_TO_NUM[month_token]
            revenue = _to_number(cells[1])
            yoy = _to_number(cells[2]) if len(cells) > 2 else None
            if not revenue:
                continue  # un-released month
            period = f"{year}-{month:02d}-01"
            rows.append(FinancialRow(
                period, "TSMC", "monthly_revenue", revenue,
                "NT$ thousands", url,
            ))
            if yoy is not None:
                rows.append(FinancialRow(
                    period, "TSMC", "monthly_revenue_yoy", yoy, "%", url,
                ))
            added += 1
        logging.info("TSMC monthly %d: %d months extracted", year, added)
    return rows


# --------------------------------------------------------------------------- #
# TSMC quarterly results                                                      #
# --------------------------------------------------------------------------- #

QUARTER_END = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def _previous_quarters(n: int, today: date) -> list[tuple[int, int]]:
    """Return up to N most recently *reported* TSMC quarters (newest first).

    TSMC publishes Q-results ~2-3 weeks after quarter end, so we only include
    quarters whose end date is at least 21 days in the past.
    """
    candidates: list[tuple[int, int]] = []
    for y in range(today.year - 3, today.year + 1):
        for q in (1, 2, 3, 4):
            mo, day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[q]
            if (today - date(y, mo, day)).days >= 21:
                candidates.append((y, q))
    return list(reversed(candidates))[:n]


def scrape_tsmc_quarterly(session: requests.Session, n: int) -> list[FinancialRow]:
    rows: list[FinancialRow] = []
    today = date.today()
    for (year, q) in _previous_quarters(n, today):
        url = f"{TSMC_BASE}/english/quarterly-results/{year}/q{q}"
        resp = fetch(url, session)
        if resp is None:
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        tbl = soup.find("table")
        if tbl is None:
            logging.warning("TSMC quarterly %d Q%d: no table found", year, q)
            continue
        period = f"{year}-{QUARTER_END[q]}"
        extracted = 0
        for tr in tbl.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            label = cells[0].lower()
            actual_idx = 1  # the "Actual" column comes right after the label
            # Some rows have a sub-header layout where cells[1] is "Actual" text;
            # we skip those because they have no number in column 1.
            value = _to_number(cells[actual_idx])
            if value is None:
                continue
            if "net revenue" in label and "us$" in label:
                rows.append(FinancialRow(period, "TSMC", "quarterly_revenue",
                                         value, "US$ billion", url))
                extracted += 1
            elif "gross margin" in label:
                rows.append(FinancialRow(period, "TSMC", "quarterly_gross_margin",
                                         value, "%", url))
                extracted += 1
            elif "operating margin" in label:
                rows.append(FinancialRow(period, "TSMC", "quarterly_operating_margin",
                                         value, "%", url))
                extracted += 1
            elif "exchange rate" in label:
                rows.append(FinancialRow(period, "TSMC", "exchange_rate_usd_ntd",
                                         value, "NTD/USD", url))
                extracted += 1
        logging.info("TSMC quarterly %d Q%d: %d metrics", year, q, extracted)
    return rows


# --------------------------------------------------------------------------- #
# ASML quarterly press releases                                               #
# --------------------------------------------------------------------------- #

ASML_BASE = "https://www.asml.com"
ASML_FIN_RESULTS_INDEX = f"{ASML_BASE}/en/investors/financial-results"

# Map ASML press-release labels -> our metric names
ASML_METRIC_MAP = {
    "total net sales": ("quarterly_net_sales", "EUR millions"),
    "new lithography systems sold (units)": ("new_lithography_systems_sold", "units"),
    "used lithography systems sold (units)": ("used_lithography_systems_sold", "units"),
    "gross profit": ("quarterly_gross_profit", "EUR millions"),
    "gross margin (%)": ("quarterly_gross_margin", "%"),
    "net income": ("quarterly_net_income", "EUR millions"),
    "eps (basic; in euros)": ("quarterly_eps_basic", "EUR"),
}


def _find_asml_quarter_pages(session: requests.Session, n: int) -> list[tuple[int, int, str]]:
    """Return [(year, quarter, press_release_url), ...] for the latest N quarters."""
    resp = fetch(ASML_FIN_RESULTS_INDEX, session)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "lxml")
    quarter_pages: list[tuple[int, int, str]] = []  # (year, quarter, results-page-url)
    seen: set[tuple[int, int]] = set()
    pattern = re.compile(r"/en/investors/financial-results/q([1-4])-(20\d{2})$", re.I)
    for a in soup.find_all("a", href=True):
        m = pattern.search(a["href"])
        if not m:
            continue
        q, y = int(m.group(1)), int(m.group(2))
        if (y, q) in seen:
            continue
        seen.add((y, q))
        quarter_pages.append((y, q, ASML_BASE + a["href"] if a["href"].startswith("/") else a["href"]))

    # Sort newest first
    quarter_pages.sort(key=lambda t: (t[0], t[1]), reverse=True)
    quarter_pages = quarter_pages[:n]

    # Resolve each to its actual press-release page (contains the financial table)
    resolved: list[tuple[int, int, str]] = []
    for (y, q, page_url) in quarter_pages:
        pr_resp = fetch(page_url, session)
        if pr_resp is None:
            continue
        psoup = BeautifulSoup(pr_resp.text, "lxml")
        press_url: str | None = None
        for a in psoup.find_all("a", href=True):
            h = a["href"]
            if "press-releases" in h and "financial-results" in h:
                press_url = h
                break
        if not press_url:
            logging.warning("ASML Q%d %d: no press-release link found on %s", q, y, page_url)
            continue
        if press_url.startswith("/"):
            press_url = ASML_BASE + press_url
        resolved.append((y, q, press_url))
    return resolved


def scrape_asml_quarterly(session: requests.Session, n: int) -> list[FinancialRow]:
    rows: list[FinancialRow] = []
    quarters = _find_asml_quarter_pages(session, n)
    if not quarters:
        logging.warning("ASML: no quarter pages resolved")
        return rows

    for (year, q, pr_url) in quarters:
        resp = fetch(pr_url, session)
        if resp is None:
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        tbl = soup.find("table")
        if tbl is None:
            logging.warning("ASML Q%d %d: no table at %s", q, year, pr_url)
            continue

        # The header row encodes which columns correspond to which quarters.
        # e.g.: ['(Figures in millions of euros unless otherwise indicated)', 'Q4 2025', 'Q1 2026']
        header_cells: list[str] = []
        body_rows: list[list[str]] = []
        for tr in tbl.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if not cells:
                continue
            if not header_cells and any(re.match(r"q[1-4]\s+20\d{2}", c, re.I) for c in cells):
                header_cells = cells
                continue
            body_rows.append(cells)

        if not header_cells:
            logging.warning("ASML Q%d %d: no header row recognized", q, year)
            continue

        # Map column index -> period date (YYYY-MM-DD)
        col_periods: dict[int, tuple[str, str]] = {}  # idx -> (period, q_label)
        for idx, h in enumerate(header_cells):
            m = re.match(r"Q([1-4])\s+(20\d{2})", h, re.I)
            if m:
                cq, cy = int(m.group(1)), int(m.group(2))
                col_periods[idx] = (f"{cy}-{QUARTER_END[cq]}", f"Q{cq} {cy}")

        extracted = 0
        for cells in body_rows:
            label = cells[0].lower().strip()
            if label not in ASML_METRIC_MAP:
                continue
            metric, unit = ASML_METRIC_MAP[label]
            for col_idx, (period, _q_label) in col_periods.items():
                if col_idx >= len(cells):
                    continue
                val = _to_number(cells[col_idx])
                if val is None:
                    continue
                rows.append(FinancialRow(period, "ASML", metric, val, unit, pr_url))
                extracted += 1
        logging.info("ASML Q%d %d: %d metrics from %s", q, year, extracted, pr_url)

    return rows


# --------------------------------------------------------------------------- #
# CSV persistence (append + dedupe)                                           #
# --------------------------------------------------------------------------- #

def write_csv(new_rows: list[FinancialRow]) -> tuple[int, int]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if CSV_PATH.exists():
        with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    seen = {(r["date"], r["company"], r["metric"]) for r in existing}
    added: list[dict] = []
    for r in new_rows:
        key = (r.date, r.company, r.metric)
        if key in seen:
            continue
        seen.add(key)
        added.append(asdict(r))
    combined = existing + added
    # Keep file sorted by (company, date) for readability
    combined.sort(key=lambda d: (d["company"], d["date"], d["metric"]))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in combined:
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
    return len(added), len(combined)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def main() -> int:
    configure_logging()
    logging.info("Financial scraper starting")
    session = make_session()

    cur_year = date.today().year
    monthly_years = list(range(cur_year - TSMC_MONTHLY_YEARS_HISTORY + 1, cur_year + 1))

    sources: list[tuple[str, Callable[[], list[FinancialRow]]]] = [
        ("TSMC monthly revenue",
         lambda: scrape_tsmc_monthly(session, monthly_years)),
        ("TSMC quarterly results",
         lambda: scrape_tsmc_quarterly(session, TSMC_QUARTERS_HISTORY)),
        ("ASML quarterly results",
         lambda: scrape_asml_quarterly(session, ASML_PRESS_RELEASES)),
    ]

    all_rows: list[FinancialRow] = []
    for label, fn in sources:
        logging.info("Source: %s", label)
        try:
            rows = fn()
        except Exception:
            logging.exception("Source crashed: %s", label)
            rows = []
        logging.info("  -> %d rows", len(rows))
        all_rows.extend(rows)

    if not all_rows:
        logging.warning("No financial rows collected; CSV not updated")
        return 2

    added, total = write_csv(all_rows)
    logging.info("CSV: +%d new rows, %d total -> %s", added, total, CSV_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
