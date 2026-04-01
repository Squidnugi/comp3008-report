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


def read_and_standardize(period: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Normalize column names across files to reduce case-based mismatches.
    df.columns = [c.upper() for c in df.columns]
    df["SOURCE_PERIOD"] = period
    return df


def summarize_dataframe(df: pd.DataFrame, period: str) -> dict:
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    year_col = "FILEYEAR" if "FILEYEAR" in df.columns else None
    year_min = year_max = None
    if year_col:
        years = pd.to_numeric(df[year_col], errors="coerce")
        if years.notna().any():
            year_min = int(years.min())
            year_max = int(years.max())

    return {
        "period": period,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_top10": missing_pct.head(10).round(2),
        "year_min": year_min,
        "year_max": year_max,
    }


def markdown_table_from_series(series: pd.Series) -> str:
    lines = ["| Variable | Missing % |", "|---|---:|"]
    for k, v in series.items():
        lines.append(f"| {k} | {v:.2f} |")
    return "\n".join(lines)


def main() -> None:
    dataframes = {}
    summaries = []

    for period, path in FILES:
        df = read_and_standardize(period, path)
        dataframes[period] = df
        summaries.append(summarize_dataframe(df, period))

    cols_a = set(dataframes["2019-2021"].columns)
    cols_b = set(dataframes["2022-2024"].columns)
    shared_cols = sorted(cols_a & cols_b)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)

    combined = pd.concat([dataframes["2019-2021"], dataframes["2022-2024"]], ignore_index=True, sort=False)

    # Save machine-readable diagnostics for downstream stages.
    pd.Series(shared_cols, name="shared_columns").to_csv(OUTPUT_DIR / "stage1_shared_columns.csv", index=False)
    pd.Series(only_a, name="only_in_2019_2021").to_csv(OUTPUT_DIR / "stage1_only_in_2019_2021.csv", index=False)
    pd.Series(only_b, name="only_in_2022_2024").to_csv(OUTPUT_DIR / "stage1_only_in_2022_2024.csv", index=False)

    summary_lines = [
        "# Stage 1: Dataset Understanding",
        "",
        "## Files Profiled",
        f"- {FILES[0][1].as_posix()}",
        f"- {FILES[1][1].as_posix()}",
        "",
        "## High-Level Shape",
    ]

    for s in summaries:
        year_span = "Unknown"
        if s["year_min"] is not None and s["year_max"] is not None:
            year_span = f"{s['year_min']} to {s['year_max']}"

        summary_lines.extend(
            [
                f"### {s['period']}",
                f"- Rows: {s['rows']:,}",
                f"- Columns (including SOURCE_PERIOD): {s['cols']:,}",
                f"- Duplicate rows: {s['duplicate_rows']:,}",
                f"- FILEYEAR span: {year_span}",
                "- Top 10 missing variables:",
                markdown_table_from_series(s["missing_top10"]),
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Cross-File Compatibility",
            f"- Shared columns after standardization: {len(shared_cols)}",
            f"- Columns only in 2019-2021: {len(only_a)}",
            f"- Columns only in 2022-2024: {len(only_b)}",
            "",
            "## Combined Dataset Snapshot",
            f"- Combined rows: {combined.shape[0]:,}",
            f"- Combined columns: {combined.shape[1]:,}",
            "",
            "## Stage 1 Interpretation Notes",
            "- Several variables are fully missing (100%) and should likely be excluded or treated as structurally unavailable.",
            "- There are schema shifts between periods (new census2021 fields and renamed geography/ID fields) that require harmonization in Stage 2.",
            "- FILEYEAR should be treated as numeric year, not a timestamp.",
        ]
    )

    (OUTPUT_DIR / "stage1_overview.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Wrote:")
    print((OUTPUT_DIR / "stage1_overview.md").as_posix())
    print((OUTPUT_DIR / "stage1_shared_columns.csv").as_posix())
    print((OUTPUT_DIR / "stage1_only_in_2019_2021.csv").as_posix())
    print((OUTPUT_DIR / "stage1_only_in_2022_2024.csv").as_posix())


if __name__ == "__main__":
    main()
