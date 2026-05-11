# Decoding Electoral Behavior in West Bengal
### UGDSAI 29 — Unsupervised Machine Learning | Group 5

**Team:** Abhishek | Mayank | Harsh | Bhawishya  
**Faculty:** Mr. Anant Mittal

---

## Project Overview
Constituency-level clustering and swing analysis across three West Bengal Assembly Elections (2016, 2021, 2026) using unsupervised ML techniques.

## Data Sources
| # | Source | Data |
|---|--------|------|
| 1 | Election Commission of India (`results.eci.gov.in`) | Vote shares, turnout, margins, NOTA |
| 2 | MyNeta / ADR (`myneta.info/WestBengal2021`) | Candidate affidavits: age, education, assets, criminal cases |
| 3 | Census of India 2011 (`censusindia.gov.in`) | Literacy, urban %, sex ratio, population density |

## Techniques Applied
- **K-Means Clustering** (with and without PCA)
- **Hierarchical Clustering** (Ward linkage + Dendrogram)
- **Principal Component Analysis** (Scree, Loadings, Biplot)
- **t-SNE** (Non-linear dimensionality reduction)
- **Association Rule Mining** (Apriori — mlxtend)
- **Swing Analysis** (2016→2021→2026)

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy mlxtend
python3 run.py         # Generates plots 01–12
python3 analysis_p3.py # Generates plots 13–17 (Census + Association)
```

## Output
- `plots/` — 17 visualisation charts
- `clustered_constituencies.csv` — constituency-level cluster assignments

## Key Findings
- TMC dominance clusters (high margin, low BJP) cover ~60% of constituencies
- BJP consolidated in **247 seats** (2016→2026)
- Left Front collapsed (>15pp drop) in **118 seats** between 2016–2021
- High literacy ↔ High urban % association (Lift = 3.19)
- Large TMC margins strongly associated with low BJP vote share (Lift = 2.52)
