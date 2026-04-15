# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses a local Python virtual environment at `.venv/`. Always run Python via:

```bash
.venv/Scripts/python.exe
```

To run the notebook non-interactively:

```bash
.venv/Scripts/jupyter.exe nbconvert --to notebook --execute COMP3008_Analysis.ipynb --output COMP3008_Analysis.ipynb
```

To install dependencies:

```bash
.venv/Scripts/pip.exe install numpy pandas matplotlib seaborn scikit-learn pmdarima jinja2
```

## Project Purpose

COMP3008 Assessment 2 — data analysis of the UK Annual Population Survey (APS) exploring how COVID-19 impacted employment patterns and regional economic inequality (2019–2024).

## Data

Two raw CSV files in `Report_Data/`:
- `AnnualPopulationSurvey_Jan2019_Dec2021.csv` — 430k rows, 526 columns
- `AnnualPopulationSurvey_Jan2022_Dec2024.csv` — 320k rows, 473 columns

These are loaded, harmonised, and combined into a single 750k-row dataframe at runtime. The data is **not** committed to git (only sample outputs are).

## Notebook Structure (`COMP3008_Analysis.ipynb`)

The notebook is the single source of truth for all analysis. It follows a linear pipeline:

| Section | Content |
|---|---|
| §2 | Load & merge both CSV files; tag each row with `SOURCE_PERIOD` |
| §3 | Pre-processing: column harmonisation, missingness audit, time-series prep |
| §4 | EDA 1 — Descriptive statistics & distributions (age, pay) |
| §5 | EDA 2 — Labour status time-series (2019–2024) |
| §6 | EDA 3 — Regional employment comparison (GOR9D heatmap) |
| §7 | EDA 4 — K-Means clustering of worker profiles (PCA visualisation) |
| §8 | Model 1 — ARIMA forecast of monthly employment rate (requires `pmdarima`) |
| §9 | Model 2 — Ridge regression of gross weekly pay |
| §10 | Model 3 — Random Forest regressor (capped at 200k training rows) |
| §11 | Model comparison table → `analysis_outputs/model_comparison.csv` |

## Column Harmonisation

The two APS periods use different column names for the same concepts (e.g. `GOR9D` vs `GOR9DCENSUS2021`, `HIQUAL15` vs `HIQUAL22`). The `coalesce()` helper resolves these by taking the first non-null value. The canonical column mapping is logged in `analysis_outputs/stage2_variable_selection_log.md`.

Key canonical names used throughout the notebook:

| Canonical | Meaning |
|---|---|
| `LABOUR_STATUS` | ILO labour market status (`ILODEFR`: 1=Employed, 2=Unemployed, 3=Inactive) |
| `OUTCOME_STATUS` | `IOUTCOME` — used as classification target |
| `GROSS_WEEKLY_PAY` | `GRSSWK` — regression target; ~72% missing (employed respondents only) |
| `REGION_CODE` | `GOR9D` (harmonised) — English standard region codes |
| `YEAR` | `FILEYEAR` — numeric calendar year, not a timestamp |

## Outputs

All generated artefacts land in `analysis_outputs/`:
- `figures/` — publication figures saved as PNG (referenced by section prefix: `eda1_`, `model1_`, etc.)
- `model_comparison.csv` — final cross-model metrics table
- `stage*.md` — intermediate pipeline logs (overview, variable selection, EDA summary, modelling summary)
- `stage2_harmonized_core.csv` / `stage2_harmonized_core_sample_5000.csv` — harmonised dataset snapshots

## Key Modelling Decisions

- **Train/test split** is time-aware: train on YEAR ≤ 2023, test on 2024.
- **ARIMA** requires `pmdarima`; the notebook skips it gracefully if the package is absent.
- **Random Forest** training is capped at `MAX_TRAIN = 200_000` rows for runtime speed.
- Wage variables (`GROSS_WEEKLY_PAY`, `HOURLY_RATE`) are high-missingness and apply only to employed respondents — treat any wage findings cautiously.

## Copilot Prompt

`.github/prompts/comp3008-aps-analysis.prompt.md` contains the structured agent prompt used to drive this analysis workflow. It references the assessment brief PDF and data dictionaries as fixed context files.
