# STATS 531 W26 — Final Project (submission folder)

**Topic:** Stochastic leverage POMP for weekly S&P 500 log-returns (`pypomp` / JAX).

## Contents

| File | Role |
|------|------|
| `blinded.qmd` | Quarto source (anonymous for peer review) |
| `blinded.pdf` | Compiled report |
| `references.bib` | Bibliography |
| `spx_weekly.csv` | Cached weekly percentage log-returns (2015–2024 window) |
| `get_data.py` | Regenerates `spx_weekly.csv` from Yahoo Finance `^GSPC` |
| `requirements.txt` | Python dependencies |

## Render

```bash
cd Project
pip install -r requirements.txt
quarto render blinded.qmd
```

Full execution is CPU-intensive (IF2, global search, two profile grids, probes); expect roughly 5–15+ minutes on a laptop.

## Zip for Canvas

Include `blinded.qmd`, `blinded.pdf`, data, `references.bib`, and any generated figures if you distribute a pre-built bundle (figures are recreated by `quarto render`).
