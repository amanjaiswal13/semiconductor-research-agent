# Semiconductor Supply Chain Research Agent

An autonomous AI agent for tracking the semiconductor and AI infrastructure supply chain for investment research.

## Problem Statement

Investment research on semiconductor companies requires continuous monitoring of:
- Company financial performance (revenue, margins, capex)
- Supply chain dependencies and bottlenecks
- Industry news and capacity announcements
- Technology transitions (node migrations)

Manual tracking is time-consuming and misses critical signals. This agent automates that research process.

## What It Does

### Autonomous Monitoring
- Scrapes semiconductor news focusing on TSMC, ASML, NVIDIA
- Tracks financial metrics from company earnings reports
- Maps supply chain dependencies and identifies bottlenecks
- Generates alerts for significant events
- Produces daily analysis reports with investment signals

### Decision-Making Logic

The agent autonomously decides:
- **Which companies to prioritize** based on news volume, financial changes, and supply chain position
- **When to issue alerts**: Critical (bottleneck + negative news), Bullish (revenue growth >15% + margin expansion), Watchlist (high capex)
- **What constitutes a supply chain risk**: Single-source dependencies, capacity constraints

### Key Insights Generated

- **TSMC-ASML Dependency**: 90% of advanced chips require EUV machines only ASML produces (~60 machines/year limit)
- **Installed Base Revenue**: ASML's service revenue grew 26% YoY - shows customer lock-in and recurring revenue model
- **Supply Chain Concentration**: Taiwan produces 90% of advanced chips - major geopolitical risk
- **Capital Intensity**: TSMC spending $30-40B annually on capacity expansion
- **Bottleneck Identification**: ASML identified as critical single point of failure for AI infrastructure

## Architecture

## Built With

- **Claude Code**: Agentic workflow for autonomous code generation
- **Python**: Core logic and data processing
- **BeautifulSoup**: Web scraping
- **JSON/CSV**: Data storage and analysis

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/semiconductor-research-agent

# Navigate to directory
cd semiconductor-research-agent

# Install dependencies
pip install -r requirements.txt
```

## Usage

**Run the autonomous agent:**
```bash
python main.py
```

This will:
1. Scrape latest semiconductor news
2. Update financial metrics
3. Refresh supply chain dependencies
4. Analyze changes and generate alerts
5. Produce daily report in `output/` directory

**Run individual components:**
```bash
python src/news_scraper.py
python src/financial_scraper.py
python src/supply_chain_mapper.py
```

## Investment Research Insights

### Supply Chain Bottlenecks Identified

1. **ASML EUV Machines**: Only source of extreme ultraviolet lithography equipment; production limited to ~60 machines/year; each machine costs $150-200M

2. **TSMC Manufacturing Dominance**: 90% market share in advanced chips (5nm, 3nm); all facilities in Taiwan

3. **Spruce Pine Quartz**: Critical high-purity quartz from single North Carolina town; essential for silicon wafer production crucibles

### Financial Signals Tracked

- **Capex-to-Revenue Ratio**: Indicates aggressive capacity expansion (TSMC ~40%)
- **Gross Margin Trends**: Manufacturing efficiency and pricing power
- **R&D as % of Revenue**: Innovation investment levels
- **Installed Base Growth**: ASML's recurring service revenue opportunity

### Autonomous Decision Framework

The agent uses multi-factor analysis:

## Methodology

This agent demonstrates **autonomous decision-making** rather than fixed rules:
- Adapts analysis based on data availability
- Prioritizes research targets based on market signals
- Learns patterns from historical data to improve alert accuracy
- Documents reasoning in output reports

## Learning Journey

I built this agent to understand the semiconductor supply chain from first principles. Key discoveries:

- The entire AI infrastructure depends on a handful of companies in a concentrated supply chain
- ASML's monopoly on EUV equipment creates a hard constraint on global chip production
- Taiwan's dominance in manufacturing creates both economic opportunity and geopolitical risk
- The shift from selling equipment to recurring service revenue (ASML's installed base) is a powerful business model

## Future Enhancements

- Real-time data integration (Bloomberg/FactSet APIs)
- Sentiment analysis on earnings call transcripts
- Expand coverage: memory manufacturers (SK Hynix, Micron), equipment suppliers (Applied Materials, Tokyo Electron)
- Chinese semiconductor ecosystem (SMIC, Hua Hong)
- Emerging market tracking (India, Middle East sovereign AI investments)
- Predictive modeling for capacity constraints

## Author

**Aman Jaiswal**
MBA Candidate, Simon Business School, University of Rochester

Background: Finance (UBS, Credit Suisse) | Strategy & Corporate Finance

## Acknowledgments

Built using Claude Code for agentic development workflow. Demonstrates autonomous research methodology applicable to frontier AI capability tracking and emerging market analysis.

## License

MIT License