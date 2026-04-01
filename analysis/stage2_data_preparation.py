from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Report Data"
OUTPUT_DIR = BASE_DIR / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = [
    ("2019-2021", DATA_DIR / "AnnualPopulationSurvey_Jan2019_Dec2021.csv"),
    ("2022-2024", DATA_DIR / "AnnualPopulationSurvey_Jan2022_Dec2024.csv"),
]

# Canonical variable names and possible raw-column sources across periods.
CANONICAL_MAP = {
    "PERSON_ID": ["IDREF"],
    "YEAR": ["FILEYEAR"],
    "WEIGHT": ["NPWT22C"],
    "COUNTRY_CODE": ["CTRY9D"],
    "REGION_CODE": ["GOR9D", "GOR9DCENSUS2021"],
    "COMBINED_AUTHORITY": ["COMBINEDAUTHORITIES", "COMBINEDAUTHORITIESCENSUS2021"],
    "ITL2_CODE": ["ITL221", "ITL221CENSUS2021", "ITL225CENSUS2021"],
    "ITL3_CODE": ["ITL321", "ITL321CENSUS2021", "ITL325CENSUS2021"],
    "AGE": ["AGE"],
    "SEX": ["SEX"],
    "LABOUR_STATUS": ["ILODEFR"],
    "OUTCOME_STATUS": ["IOUTCOME"],
    "FULLTIME_PARTTIME": ["FTPT"],
    "GROSS_WEEKLY_PAY": ["GRSSWK"],
    "HOURLY_RATE": ["HRRATE"],
    "HOURPAY": ["HOURPAY"],
    "HIGHEST_QUAL": ["HIQUAL15", "HIQUAL22"],
    "HEALTH_LIMITATION": ["HEALYR"],
    "ETHNICITY": ["ETH11EW"],
    "COUNTRY_NAME": ["COUNTRY"],
}

NUMERIC_CANONICAL_COLS = {
    "YEAR",
    "WEIGHT",
    "AGE",
    "GROSS_WEEKLY_PAY",
    "HOURLY_RATE",
    "HOURPAY",
}


def read_data(period: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    df = df.assign(SOURCE_PERIOD=period)
    return df


def coalesce_columns(df: pd.DataFrame, options: list[str]) -> pd.Series:
    existing = [c for c in options if c in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index)

    result = df[existing[0]]
    for col in existing[1:]:
        result = result.fillna(df[col])
    return result


def build_harmonized(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    harmonized = pd.DataFrame(index=df.index)
    mapping_rows = []

    for canonical, raw_options in CANONICAL_MAP.items():
        used = [c for c in raw_options if c in df.columns]
        harmonized[canonical] = coalesce_columns(df, raw_options)
        mapping_rows.append(
            {
                "canonical": canonical,
                "raw_options": ", ".join(raw_options),
                "raw_used": ", ".join(used) if used else "<missing in source>",
            }
        )

    harmonized["SOURCE_PERIOD"] = df["SOURCE_PERIOD"]

    for col in NUMERIC_CANONICAL_COLS:
        harmonized[col] = pd.to_numeric(harmonized[col], errors="coerce")

    for col in harmonized.columns:
        if col in NUMERIC_CANONICAL_COLS:
            continue
        if pd.api.types.is_object_dtype(harmonized[col]):
            harmonized[col] = harmonized[col].astype("string").str.strip()

    mapping_df = pd.DataFrame(mapping_rows)
    return harmonized, mapping_df


def variable_selection_log(harmonized: pd.DataFrame, mapping_df: pd.DataFrame) -> str:
    missing_pct = (harmonized.isna().mean() * 100).sort_values(ascending=False)

    selected = []
    excluded = []

    for col in harmonized.columns:
        miss = float(missing_pct[col])
        if col == "SOURCE_PERIOD":
            selected.append((col, miss, "Critical provenance field for split-period comparisons."))
        elif col in {"YEAR", "AGE", "LABOUR_STATUS", "OUTCOME_STATUS", "FULLTIME_PARTTIME", "GROSS_WEEKLY_PAY", "HOURLY_RATE", "HIGHEST_QUAL"}:
            selected.append((col, miss, "Core explanatory/outcome candidate for EDA and modeling."))
        elif miss <= 85.0:
            selected.append((col, miss, "Kept as potentially useful contextual feature."))
        else:
            excluded.append((col, miss, "Very high missingness; keep out of initial core analysis."))

    lines = [
        "# Stage 2: Variable Selection Log",
        "",
        "## Scope",
        "- This log is generated from observed data structure and missingness.",
        "- Assessment brief and data dictionaries should be used to refine code meanings in narrative write-up.",
        "- Where dictionary interpretation is uncertain, variables are retained by statistical utility rather than semantic certainty.",
        "",
        "## Column Mapping Decisions",
        "| Canonical | Raw Options | Raw Used |",
        "|---|---|---|",
    ]

    for _, row in mapping_df.iterrows():
        lines.append(f"| {row['canonical']} | {row['raw_options']} | {row['raw_used']} |")

    lines.extend(["", "## Selected Variables (Initial Core)", "| Variable | Missing % | Rationale |", "|---|---:|---|"])
    for col, miss, why in selected:
        lines.append(f"| {col} | {miss:.2f} | {why} |")

    lines.extend(["", "## Excluded from Initial Core", "| Variable | Missing % | Rationale |", "|---|---:|---|"])
    for col, miss, why in excluded:
        lines.append(f"| {col} | {miss:.2f} | {why} |")

    lines.extend(
        [
            "",
            "## Stage 2 Assumptions",
            "- YEAR is treated as numeric calendar year derived from FILEYEAR.",
            "- Geographic columns are harmonized across census naming changes via coalescing.",
            "- Semantic recoding of categorical codes (for example, specific ILODEFR classes) is deferred until dictionary-driven modeling setup.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    frames = []
    mapping_per_period = []

    for period, path in FILES:
        raw = read_data(period, path)
        harmonized, mapping_df = build_harmonized(raw)
        frames.append(harmonized)

        period_mapping = mapping_df.copy()
        period_mapping.insert(0, "period", period)
        mapping_per_period.append(period_mapping)

    combined = pd.concat(frames, ignore_index=True)
    mapping_all = pd.concat(mapping_per_period, ignore_index=True)

    missing_report = (combined.isna().mean() * 100).round(2).reset_index()
    missing_report.columns = ["variable", "missing_pct"]
    missing_report = missing_report.sort_values("missing_pct", ascending=False)

    core_cols = [
        "PERSON_ID",
        "SOURCE_PERIOD",
        "YEAR",
        "WEIGHT",
        "AGE",
        "SEX",
        "LABOUR_STATUS",
        "OUTCOME_STATUS",
        "FULLTIME_PARTTIME",
        "GROSS_WEEKLY_PAY",
        "HOURLY_RATE",
        "HOURPAY",
        "HIGHEST_QUAL",
        "HEALTH_LIMITATION",
        "ETHNICITY",
        "COUNTRY_NAME",
        "COUNTRY_CODE",
        "REGION_CODE",
        "COMBINED_AUTHORITY",
        "ITL2_CODE",
        "ITL3_CODE",
    ]

    core_existing = [c for c in core_cols if c in combined.columns]
    core_df = combined[core_existing].copy()

    core_df.to_csv(OUTPUT_DIR / "stage2_harmonized_core.csv", index=False)
    core_df.head(5000).to_csv(OUTPUT_DIR / "stage2_harmonized_core_sample_5000.csv", index=False)
    mapping_all.to_csv(OUTPUT_DIR / "stage2_column_mapping.csv", index=False)
    missing_report.to_csv(OUTPUT_DIR / "stage2_missingness_core.csv", index=False)

    log_text = variable_selection_log(core_df, mapping_all.drop_duplicates(subset=["canonical", "raw_options", "raw_used"]))
    (OUTPUT_DIR / "stage2_variable_selection_log.md").write_text(log_text, encoding="utf-8")

    print("Wrote:")
    print((OUTPUT_DIR / "stage2_harmonized_core.csv").as_posix())
    print((OUTPUT_DIR / "stage2_harmonized_core_sample_5000.csv").as_posix())
    print((OUTPUT_DIR / "stage2_column_mapping.csv").as_posix())
    print((OUTPUT_DIR / "stage2_missingness_core.csv").as_posix())
    print((OUTPUT_DIR / "stage2_variable_selection_log.md").as_posix())


if __name__ == "__main__":
    main()
