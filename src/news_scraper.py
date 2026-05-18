"""Phase 1 news scraper for the semiconductor supply chain research agent.

Pulls headlines from Google News RSS for a set of semiconductor-related
queries (industry-wide + key tickers) and writes a deduplicated JSON
snapshot to data/news_YYYY-MM-DD.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

QUERIES = [
    "semiconductor supply chain",
    "TSMC",
    "ASML",
    "NVIDIA",
    "AMD",
    "Intel semiconductor",
]

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
REQUEST_TIMEOUT = 20
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "SemiconductorResearchAgent/0.1 (+contact: stanley-laman-group)"
)


@dataclass(frozen=True)
class NewsItem:
    headline: str
    date: str
    url: str
    summary: str
    source: str
    query: str


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"scraper_{datetime.now():%Y-%m-%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def fetch_feed(query: str, session: requests.Session) -> str | None:
    url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.error("Failed to fetch feed for %r: %s", query, exc)
        return None
    return response.text


def clean_summary(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = BeautifulSoup(raw_html, "lxml").get_text(" ", strip=True)
    return " ".join(text.split())


def parse_feed(xml_text: str, query: str) -> list[NewsItem]:
    soup = BeautifulSoup(xml_text, "lxml-xml")
    items: list[NewsItem] = []
    for entry in soup.find_all("item"):
        title = (entry.title.text if entry.title else "").strip()
        link = (entry.link.text if entry.link else "").strip()
        pub_date = (entry.pubDate.text if entry.pubDate else "").strip()
        description = clean_summary(entry.description.text if entry.description else "")
        source_tag = entry.find("source")
        source = source_tag.text.strip() if source_tag and source_tag.text else ""

        if not title or not link:
            continue

        items.append(
            NewsItem(
                headline=title,
                date=pub_date,
                url=link,
                summary=description,
                source=source,
                query=query,
            )
        )
    return items


def deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        if item.url in seen:
            continue
        seen.add(item.url)
        unique.append(item)
    return unique


def scrape() -> list[NewsItem]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    collected: list[NewsItem] = []
    for query in QUERIES:
        logging.info("Fetching feed: %s", query)
        xml_text = fetch_feed(query, session)
        if xml_text is None:
            continue
        try:
            parsed = parse_feed(xml_text, query)
        except Exception as exc:  # parser-level failures shouldn't kill the run
            logging.exception("Parse failure for %r: %s", query, exc)
            continue
        logging.info("  -> %d items", len(parsed))
        collected.extend(parsed)
        time.sleep(1)  # be polite

    unique = deduplicate(collected)
    logging.info("Collected %d total / %d unique items", len(collected), len(unique))
    return unique


def save(items: list[NewsItem]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"news_{datetime.now():%Y-%m-%d}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queries": QUERIES,
        "count": len(items),
        "items": [asdict(item) for item in items],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote %d items to %s", len(items), out_path)
    return out_path


def main() -> int:
    configure_logging()
    logging.info("Semiconductor news scraper starting")
    try:
        items = scrape()
    except Exception as exc:
        logging.exception("Unhandled scrape failure: %s", exc)
        return 1
    if not items:
        logging.warning("No items collected; nothing written")
        return 2
    save(items)
    return 0


if __name__ == "__main__":
    sys.exit(main())
