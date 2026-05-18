"""Phase 3: semiconductor supply chain mapper.

Encodes a curated dependency graph for the semi supply chain, computes
bottleneck rankings and single points of failure, and persists the
combined dataset to data/supply_chain.json.

Edge semantics: each dependency edge is (downstream -> upstream),
meaning ``downstream`` requires something supplied by ``upstream``.
So ``("TSMC", "ASML", ...)`` reads as "TSMC depends on ASML".
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_PATH = DATA_DIR / "supply_chain.json"

# Weight applied to each upstream company when computing a bottleneck score.
# A "critical" dependency contributes more downstream blast radius than a
# "medium" one.
CRITICALITY_WEIGHTS = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.5}


# --------------------------------------------------------------------------- #
# Curated dataset                                                             #
# --------------------------------------------------------------------------- #

COMPANIES: dict[str, dict] = {
    # Foundries / IDMs
    "TSMC": {
        "full_name": "Taiwan Semiconductor Manufacturing Company",
        "ticker": "TSM", "country": "Taiwan", "role": "pure-play foundry",
        "products": ["leading-edge logic (N3/N5/N7)", "CoWoS advanced packaging"],
    },
    "Samsung Foundry": {
        "full_name": "Samsung Electronics — Foundry Business",
        "ticker": "005930.KS", "country": "South Korea", "role": "foundry",
        "products": ["GAA 3nm logic", "memory + foundry IDM"],
    },
    "Intel": {
        "full_name": "Intel Corporation",
        "ticker": "INTC", "country": "USA", "role": "IDM + foundry (IFS)",
        "products": ["x86 CPUs", "Intel 18A node", "Foundry services"],
    },
    "GlobalFoundries": {
        "full_name": "GlobalFoundries Inc.",
        "ticker": "GFS", "country": "USA", "role": "specialty foundry",
        "products": ["mature-node logic", "RF / automotive silicon"],
    },

    # Equipment
    "ASML": {
        "full_name": "ASML Holding N.V.",
        "ticker": "ASML", "country": "Netherlands", "role": "lithography equipment",
        "products": ["EUV scanners", "DUV scanners", "High-NA EUV (EXE:5000)"],
    },
    "Applied Materials": {
        "full_name": "Applied Materials, Inc.",
        "ticker": "AMAT", "country": "USA", "role": "wafer fab equipment",
        "products": ["deposition", "etch", "implant", "CMP"],
    },
    "Lam Research": {
        "full_name": "Lam Research Corporation",
        "ticker": "LRCX", "country": "USA", "role": "wafer fab equipment",
        "products": ["etch", "deposition"],
    },
    "KLA": {
        "full_name": "KLA Corporation",
        "ticker": "KLAC", "country": "USA", "role": "process control",
        "products": ["metrology", "inspection"],
    },
    "Tokyo Electron": {
        "full_name": "Tokyo Electron Limited",
        "ticker": "8035.T", "country": "Japan", "role": "wafer fab equipment",
        "products": ["track / coater-developer", "etch", "deposition"],
    },

    # ASML sub-tier
    "Zeiss": {
        "full_name": "Carl Zeiss SMT GmbH",
        "ticker": None, "country": "Germany", "role": "EUV/DUV optics",
        "products": ["EUV mirror systems", "projection optics"],
        "notes": "Sole optics supplier to ASML; 24.9% owned by ASML.",
    },
    "Cymer": {
        "full_name": "Cymer LLC",
        "ticker": None, "country": "USA", "role": "EUV light source",
        "products": ["LPP EUV source"],
        "notes": "Wholly owned ASML subsidiary since 2013; modeled separately to capture component lineage.",
    },
    "Trumpf": {
        "full_name": "TRUMPF GmbH + Co. KG",
        "ticker": None, "country": "Germany", "role": "high-power lasers",
        "products": ["CO2 drive lasers for EUV light source"],
    },

    # Materials
    "Shin-Etsu": {
        "full_name": "Shin-Etsu Chemical Co., Ltd.",
        "ticker": "4063.T", "country": "Japan", "role": "silicon wafers / chemicals",
        "products": ["300mm silicon wafers", "photoresists", "rare-earth materials"],
    },
    "SUMCO": {
        "full_name": "SUMCO Corporation",
        "ticker": "3436.T", "country": "Japan", "role": "silicon wafers",
        "products": ["300mm silicon wafers"],
    },
    "JSR": {
        "full_name": "JSR Corporation",
        "ticker": "4185.T", "country": "Japan", "role": "photoresists",
        "products": ["EUV / ArF photoresists"],
    },

    # Memory
    "SK Hynix": {
        "full_name": "SK Hynix Inc.",
        "ticker": "000660.KS", "country": "South Korea", "role": "memory",
        "products": ["HBM3e", "HBM4", "DDR5", "NAND"],
    },
    "Samsung Memory": {
        "full_name": "Samsung Electronics — Memory Business",
        "ticker": "005930.KS", "country": "South Korea", "role": "memory",
        "products": ["HBM", "DRAM", "NAND"],
    },
    "Micron": {
        "full_name": "Micron Technology, Inc.",
        "ticker": "MU", "country": "USA", "role": "memory",
        "products": ["HBM3e", "DDR5", "NAND"],
    },

    # EDA / software
    "Synopsys": {
        "full_name": "Synopsys, Inc.",
        "ticker": "SNPS", "country": "USA", "role": "EDA software + IP",
        "products": ["design tools", "verification IP", "interface IP"],
    },
    "Cadence": {
        "full_name": "Cadence Design Systems, Inc.",
        "ticker": "CDNS", "country": "USA", "role": "EDA software + IP",
        "products": ["design tools", "verification", "IP"],
    },

    # Fabless / downstream
    "NVIDIA": {
        "full_name": "NVIDIA Corporation",
        "ticker": "NVDA", "country": "USA", "role": "fabless (AI accelerators / GPUs)",
        "products": ["Blackwell / Rubin GPUs", "data-center accelerators"],
    },
    "AMD": {
        "full_name": "Advanced Micro Devices, Inc.",
        "ticker": "AMD", "country": "USA", "role": "fabless (CPU/GPU)",
        "products": ["EPYC CPUs", "MI300/MI350 accelerators", "Ryzen / Radeon"],
    },
    "Apple": {
        "full_name": "Apple Inc.",
        "ticker": "AAPL", "country": "USA", "role": "fabless (consumer SoCs)",
        "products": ["A-series / M-series SoCs"],
    },
}


# (downstream, upstream, category, description, criticality, sole_source)
DEPENDENCIES: list[tuple[str, str, str, str, str, bool]] = [
    # TSMC
    ("TSMC",  "ASML",              "equipment",     "EUV + DUV lithography scanners",          "critical", True),
    ("TSMC",  "Applied Materials", "equipment",     "Deposition, etch, CMP",                   "high",     False),
    ("TSMC",  "Lam Research",      "equipment",     "Etch + deposition",                       "high",     False),
    ("TSMC",  "KLA",               "equipment",     "Process control + metrology",             "high",     False),
    ("TSMC",  "Tokyo Electron",    "equipment",     "Track, etch, deposition",                 "high",     False),
    ("TSMC",  "Shin-Etsu",         "materials",     "300mm silicon wafers",                    "high",     False),
    ("TSMC",  "SUMCO",             "materials",     "300mm silicon wafers",                    "high",     False),
    ("TSMC",  "JSR",               "materials",     "EUV photoresists",                        "high",     False),
    ("TSMC",  "Synopsys",          "software",      "EDA design / verification tools",         "high",     False),
    ("TSMC",  "Cadence",           "software",      "EDA design / verification tools",         "high",     False),

    # ASML sub-tier
    ("ASML",  "Zeiss",             "optics",        "EUV / DUV projection optics (sole supplier)", "critical", True),
    ("ASML",  "Cymer",             "components",    "EUV LPP light source (ASML subsidiary)",  "critical", True),
    ("ASML",  "Trumpf",            "components",    "CO2 drive laser for EUV source",          "critical", True),

    # Samsung Foundry
    ("Samsung Foundry", "ASML",              "equipment", "EUV scanners",                      "critical", True),
    ("Samsung Foundry", "Applied Materials", "equipment", "Etch + deposition",                 "high",     False),
    ("Samsung Foundry", "Lam Research",      "equipment", "Etch + deposition",                 "high",     False),
    ("Samsung Foundry", "Shin-Etsu",         "materials", "300mm silicon wafers",              "high",     False),

    # Intel
    ("Intel", "ASML",              "equipment",     "High-NA EUV (lead customer) + EUV",       "critical", True),
    ("Intel", "Applied Materials", "equipment",     "Deposition + etch",                       "high",     False),
    ("Intel", "Lam Research",      "equipment",     "Etch",                                    "high",     False),
    ("Intel", "TSMC",              "manufacturing", "External tile / chiplet production",      "medium",   False),
    ("Intel", "Synopsys",          "software",      "EDA",                                     "high",     False),

    # Memory makers
    ("SK Hynix",       "ASML",      "equipment", "DRAM / HBM lithography",         "critical", True),
    ("SK Hynix",       "Shin-Etsu", "materials", "Silicon wafers",                 "high",     False),
    ("Samsung Memory", "ASML",      "equipment", "DRAM / HBM lithography",         "critical", True),
    ("Samsung Memory", "Applied Materials", "equipment", "Deposition + etch",      "high",     False),
    ("Micron",         "ASML",      "equipment", "DRAM lithography",               "critical", True),
    ("Micron",         "Shin-Etsu", "materials", "Silicon wafers",                 "high",     False),

    # Fabless / system vendors
    ("NVIDIA", "TSMC",          "manufacturing", "Advanced-node logic mfg (Blackwell / Rubin)", "critical", True),
    ("NVIDIA", "SK Hynix",      "memory",        "HBM3e / HBM4 for AI accelerators",            "critical", False),
    ("NVIDIA", "Samsung Memory","memory",        "HBM (secondary supplier)",                    "high",     False),
    ("NVIDIA", "Micron",        "memory",        "HBM (qualified, ramping)",                    "medium",   False),
    ("NVIDIA", "Synopsys",      "software",      "EDA",                                         "high",     False),
    ("NVIDIA", "Cadence",       "software",      "EDA",                                         "high",     False),

    ("AMD", "TSMC",          "manufacturing", "CPU + GPU manufacturing",                        "critical", True),
    ("AMD", "SK Hynix",      "memory",        "HBM for MI300 / MI350",                          "critical", False),
    ("AMD", "Samsung Memory","memory",        "HBM (secondary)",                                "high",     False),
    ("AMD", "Synopsys",      "software",      "EDA",                                            "high",     False),

    ("Apple", "TSMC", "manufacturing", "A-series / M-series SoC manufacturing (sole)",          "critical", True),

    ("GlobalFoundries", "ASML",              "equipment", "DUV lithography (mature nodes)",     "high",     False),
    ("GlobalFoundries", "Applied Materials", "equipment", "Deposition + etch",                  "high",     False),
]


CAPACITY_CONSTRAINTS: list[dict] = [
    {
        "company": "TSMC",
        "resource": "CoWoS advanced packaging",
        "status": "severely constrained",
        "as_of": "2026-Q1",
        "notes": "AI accelerator demand outstripping CoWoS-L/S capacity; multi-quarter lead times. Capacity expansion (Zhunan, Chiayi) ramping through 2026.",
        "downstream_impact": ["NVIDIA", "AMD", "Apple"],
    },
    {
        "company": "ASML",
        "resource": "High-NA EUV (EXE:5000)",
        "status": "ramping",
        "as_of": "2026-Q1",
        "notes": "Limited annual output (mid-single-digit units initially); allocation contested among Intel, TSMC, Samsung.",
        "downstream_impact": ["Intel", "TSMC", "Samsung Foundry"],
    },
    {
        "company": "Zeiss",
        "resource": "EUV projection optics",
        "status": "constrained",
        "as_of": "2026-Q1",
        "notes": "Sole optics partner to ASML; multi-year backlog. Capacity is the long-pole on ASML's EUV shipment ramp.",
        "downstream_impact": ["ASML", "TSMC", "Intel", "Samsung Foundry"],
    },
    {
        "company": "SK Hynix",
        "resource": "HBM3e / HBM4",
        "status": "sold out",
        "as_of": "2026-Q1",
        "notes": "Sold out through 2026; HBM4 ramping for late-2026 / 2027 alongside NVIDIA Rubin.",
        "downstream_impact": ["NVIDIA", "AMD"],
    },
    {
        "company": "Samsung Memory",
        "resource": "HBM3e qualification at NVIDIA",
        "status": "yield issues",
        "as_of": "2026-Q1",
        "notes": "Yield improvement in progress; qualification cycle limits revenue ramp vs SK Hynix.",
        "downstream_impact": ["NVIDIA"],
    },
    {
        "company": "Shin-Etsu",
        "resource": "300mm silicon wafers",
        "status": "tight in advanced segments",
        "as_of": "2026-Q1",
        "notes": "Mature-node oversupply but leading-edge wafer demand tight as AI capacity expands.",
        "downstream_impact": ["TSMC", "Samsung Foundry", "SK Hynix"],
    },
]


# --------------------------------------------------------------------------- #
# Analytics                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class GraphAnalysis:
    bottleneck_ranking: list[dict]
    single_points_of_failure: list[dict]
    upstreams_by_company: dict[str, list[dict]]
    downstreams_by_company: dict[str, list[dict]]
    summary: dict


def analyze() -> GraphAnalysis:
    upstreams: dict[str, list[dict]] = defaultdict(list)   # who does X depend on
    downstreams: dict[str, list[dict]] = defaultdict(list) # who depends on X
    weighted_in: dict[str, float] = defaultdict(float)
    sole_source_count: dict[str, int] = defaultdict(int)

    for downstream, upstream, category, description, criticality, sole in DEPENDENCIES:
        edge = {
            "category": category,
            "description": description,
            "criticality": criticality,
            "sole_source": sole,
        }
        upstreams[downstream].append({"upstream": upstream, **edge})
        downstreams[upstream].append({"downstream": downstream, **edge})
        weighted_in[upstream] += CRITICALITY_WEIGHTS.get(criticality, 1.0)
        if sole:
            sole_source_count[upstream] += 1

    bottleneck_ranking = sorted(
        (
            {
                "company": company,
                "downstream_count": len(downstreams[company]),
                "weighted_bottleneck_score": round(weighted_in[company], 1),
                "sole_source_relationships": sole_source_count[company],
                "downstreams": sorted(d["downstream"] for d in downstreams[company]),
            }
            for company in downstreams
        ),
        key=lambda e: (
            -e["weighted_bottleneck_score"],
            -e["sole_source_relationships"],
            -e["downstream_count"],
        ),
    )

    spofs = [
        {
            "downstream": dn,
            "upstream": up,
            "category": cat,
            "description": desc,
        }
        for (dn, up, cat, desc, crit, sole) in DEPENDENCIES
        if sole and crit == "critical"
    ]
    spofs.sort(key=lambda e: (e["upstream"], e["downstream"]))

    summary = {
        "total_companies": len(COMPANIES),
        "total_dependencies": len(DEPENDENCIES),
        "critical_dependencies": sum(1 for d in DEPENDENCIES if d[4] == "critical"),
        "sole_source_dependencies": sum(1 for d in DEPENDENCIES if d[5]),
        "companies_acting_as_suppliers": len(downstreams),
        "capacity_constraints_tracked": len(CAPACITY_CONSTRAINTS),
    }

    return GraphAnalysis(
        bottleneck_ranking=bottleneck_ranking,
        single_points_of_failure=spofs,
        upstreams_by_company=dict(upstreams),
        downstreams_by_company=dict(downstreams),
        summary=summary,
    )


# --------------------------------------------------------------------------- #
# Persistence + entrypoint                                                    #
# --------------------------------------------------------------------------- #

def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"supply_chain_{datetime.now():%Y-%m-%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def build_payload(analysis: GraphAnalysis) -> dict:
    return {
        "generated_at": date.today().isoformat(),
        "schema": {
            "edge_semantics": "Each dependency is (downstream -> upstream): downstream company requires what upstream supplies.",
            "criticality_weights": CRITICALITY_WEIGHTS,
        },
        "companies": COMPANIES,
        "dependencies": [
            {
                "downstream": dn,
                "upstream": up,
                "category": cat,
                "description": desc,
                "criticality": crit,
                "sole_source": sole,
            }
            for (dn, up, cat, desc, crit, sole) in DEPENDENCIES
        ],
        "capacity_constraints": CAPACITY_CONSTRAINTS,
        "analysis": {
            "summary": analysis.summary,
            "bottleneck_ranking": analysis.bottleneck_ranking,
            "single_points_of_failure": analysis.single_points_of_failure,
            "upstreams_by_company": analysis.upstreams_by_company,
            "downstreams_by_company": analysis.downstreams_by_company,
        },
    }


def main() -> int:
    configure_logging()
    logging.info("Supply chain mapper starting")

    # Validate that every edge references a known company
    known = set(COMPANIES.keys())
    issues = []
    for (dn, up, *_rest) in DEPENDENCIES:
        if dn not in known:
            issues.append(f"unknown downstream: {dn}")
        if up not in known:
            issues.append(f"unknown upstream: {up}")
    if issues:
        for msg in issues:
            logging.error("validation: %s", msg)
        return 2

    analysis = analyze()
    payload = build_payload(analysis)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote %s", OUTPUT_PATH)

    # Console summary
    s = analysis.summary
    print()
    print(f"Companies: {s['total_companies']}  |  Dependencies: {s['total_dependencies']}  "
          f"|  Critical: {s['critical_dependencies']}  |  Sole-source: {s['sole_source_dependencies']}")

    print("\n=== Top 5 bottleneck companies (weighted by criticality) ===")
    print(f"{'Company':<22} {'Score':>6} {'Down':>5} {'SoleSrc':>8}  Downstreams")
    for b in analysis.bottleneck_ranking[:5]:
        print(f"{b['company']:<22} "
              f"{b['weighted_bottleneck_score']:>6.1f} "
              f"{b['downstream_count']:>5} "
              f"{b['sole_source_relationships']:>8}  "
              f"{', '.join(b['downstreams'])}")

    print(f"\n=== Single points of failure ({len(analysis.single_points_of_failure)}) ===")
    for spof in analysis.single_points_of_failure:
        print(f"  {spof['downstream']:<18} -> {spof['upstream']:<20} "
              f"[{spof['category']}]  {spof['description']}")

    print(f"\n=== Capacity constraints ({len(CAPACITY_CONSTRAINTS)}) ===")
    for c in CAPACITY_CONSTRAINTS:
        print(f"  [{c['status']:>22}]  {c['company']:<16}  {c['resource']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
