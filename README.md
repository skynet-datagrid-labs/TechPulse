

---

# TechPulse: Developer Ecosystem Analytics Engine

**PostgreSQL • SQL • Technology Momentum • 12 Query Pipeline**

A production-grade analytical workspace for quantifying why developer technologies succeed or fail. Packages twelve reusable, progressively structured SQL queries that synthesize community activity, enterprise adoption patterns, and developer sentiment into actionable technology health metrics.

![PostgreSQL](https://img.shields.io/badge/DATABASE-PostgreSQL-00FF00?style=for-the-badge&logo=postgresql&logoColor=white&labelColor=0D1117&color=00FF00)
![Queries](https://img.shields.io/badge/QUERIES-12_REUSABLE-00FF00?style=for-the-badge&logo=sql&logoColor=white&labelColor=0D1117&color=00FF00)
![Status](https://img.shields.io/badge/STATUS-ANALYTICS_READY-00FF00?style=for-the-badge&logo=github&logoColor=white&labelColor=0D1117&color=00FF00)

---

## Interactive Demonstration

<p align="center">
  <img src="assets/tech_risk_demo.gif" alt="Technology risk assessment query execution" width="90%">
</p>

*Live execution of the technology risk assessment query against the StackExchange-derived database.*

---

## Strategic Objective

Technology adoption decisions require empirical grounding, not intuition. This repository provides a structured analytical framework for evaluating developer ecosystem health. The query pipeline aggregates three discrete data dimensions—community engagement, corporate adoption, and developer sentiment—to produce objective technology momentum scores and comparative rankings.

---

## Repository Architecture

```
developer-ecosystem-analytics/
├── README.md
├── LICENSE
├── .gitignore
├── queries/
│   ├── 01_basic/
│   │   ├── query1_top_technologies.sql
│   │   ├── query2_tech_difficulty.sql
│   │   └── query3_monthly_trends.sql
│   ├── 02_intermediate/
│   │   ├── query4_companies_most_tags.sql
│   │   ├── query5_tech_categories.sql
│   │   ├── query6_hardest_per_company.sql
│   │   └── query7_growth_momentum.sql
│   └── 03_advanced/
│       ├── query8_parent_company_analysis.sql
│       ├── query9_tech_health_score.sql
│       ├── query10_question_quality.sql
│       ├── query11_daily_patterns.sql
│       └── query12_company_diversity.sql
├── results/
│   └── sample_outputs.csv
├── docs/
│   ├── schema_diagram.md
│   └── methodology.md
└── scripts/
    └── diagnostic_queries.sql
```

---

## Data Model Summary

The analytical framework assumes a normalized relational schema comprising five core entities. Complete schema documentation available in `docs/schema_diagram.md`.

| Table | Description |
|-------|-------------|
| `technologies` | Technology dimension with category classification, release year, and lifecycle status flags |
| `stack_overflow_questions` | Community Q&A activity including scores, answer counts, and closure status |
| `stack_overflow_question_tags` | Many-to-many bridge between questions and technology tags |
| `developer_sentiment` | Survey-derived metrics for satisfaction, adoption intent, and perceived learning curve |
| `companies` | Fortune 500–scale company dimension with parent-subsidiary relationships |
| `company_tech_adoption` | Declared technology stacks with adoption depth indicators |
| `company_question_mentions` | Question-to-company attribution mapping |

---

## Query Pipeline Catalog

| Tier | Query ID | Analytical Function |
|------|----------|---------------------|
| **Basic** | Q1 | Top trending technologies by engagement velocity |
| | Q2 | Technology learning difficulty ranking |
| | Q3 | Monthly adoption trend analysis |
| **Intermediate** | Q4 | Company-level tagging volume aggregation |
| | Q5 | Category-level technology rollups |
| | Q6 | Hardest technology per company by adoption friction |
| | Q7 | Growth momentum scoring (velocity + acceleration) |
| **Advanced** | Q8 | Parent-company consolidated technology portfolio analysis |
| | Q9 | Composite technology health scoring (multi-factor weighted) |
| | Q10 | Question quality assessment by response rate and closure ratio |
| | Q11 | Intraday posting pattern analysis |
| | Q12 | Technology stack diversity metrics per company |

Complete scoring methodologies and weighting logic documented inline within each SQL file and expanded in `docs/methodology.md`.

---

## Execution Instructions

| Step | Action |
|------|--------|
| **1** | Load source data into PostgreSQL following column specifications in `docs/schema_diagram.md` |
| **2** | Execute individual queries via `psql` using the connection string pattern below |
| **3** | Export results to `results/` directory for downstream dashboard integration or presentation |

**Execution command template:**
```bash
psql "$DATABASE_URL" -f queries/01_basic/query1_top_technologies.sql
```

**Compatibility:** Standard PostgreSQL syntax, tested against versions 14 and above. No vendor-specific extensions required.

---

## ML Pipeline (Python)

The ML pipeline consumes the CSV outputs in `results/` to build a feature matrix, train classifiers, and generate SHAP explainability artifacts.

```bash
pip install -r requirements.txt
python -m ml_pipeline.train
streamlit run streamlit_app.py
```

Artifacts (models, metrics, predictions, SHAP outputs) are written to `artifacts/`.

---

## Pre-execution Validation

Run `scripts/diagnostic_queries.sql` prior to analytical execution to validate:

- Row count integrity across all tables
- Null density assessment per critical column
- Referential integrity confirmation between foreign key relationships

---

## Contribution Framework

| Action | Requirement |
|--------|-------------|
| **Proposed changes** | Open issue with documented rationale |
| **Code contributions** | Submit PR maintaining PostgreSQL portability |
| **Documentation updates** | Mirror existing schema documentation structure |
| **New data sources** | Include proposed schema extension and methodology note |

---

## Supporting Resources

| Asset | Location |
|-------|----------|
| **Gist summary** | [https://gist.github.com/Tony405-spec/82bbd137d85ada850acdffc90c192486](https://gist.github.com/Tony405-spec/82bbd137d85ada850acdffc90c192486) |
| **Methodology deep-dive** | `docs/methodology.md` |
| **Schema visualization** | `docs/schema_diagram.md` |
| **Sample outputs** | `results/sample_outputs.csv` |

---
