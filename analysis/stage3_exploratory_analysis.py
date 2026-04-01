from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE_DIR / "analysis_outputs" / "stage2_harmonized_core.csv"
OUT_DIR = BASE_DIR / "analysis_outputs"
FIG_DIR = OUT_DIR / "stage3_figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def save_distribution_plots(df: pd.DataFrame) -> dict:
    results = {}

    age = pd.to_numeric(df["AGE"], errors="coerce").dropna()
    plt.figure(figsize=(10, 5))
    sns.histplot(age, bins=40, kde=True, color="#1f77b4")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    p = FIG_DIR / "eda_age_distribution.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    results["age_distribution"] = p.name

    pay = pd.to_numeric(df["GROSS_WEEKLY_PAY"], errors="coerce")
    pay = pay[(pay > 0) & (pay < pay.quantile(0.99))]
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=pay, color="#2ca02c")
    plt.title("Gross Weekly Pay (Trimmed at 99th Percentile)")
    plt.ylabel("Gross Weekly Pay")
    p = FIG_DIR / "eda_pay_boxplot.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    results["pay_boxplot"] = p.name

    return results


def save_groupwise_plots(df: pd.DataFrame) -> dict:
    results = {}

    temp = df.copy()
    temp["YEAR"] = pd.to_numeric(temp["YEAR"], errors="coerce")
    temp = temp.dropna(subset=["YEAR", "LABOUR_STATUS"])
    temp["YEAR"] = temp["YEAR"].astype(int)

    top_status = temp["LABOUR_STATUS"].value_counts().head(5).index
    temp = temp[temp["LABOUR_STATUS"].isin(top_status)]

    counts = temp.groupby(["YEAR", "LABOUR_STATUS"]).size().reset_index(name="count")
    totals = counts.groupby("YEAR")["count"].transform("sum")
    counts["pct"] = 100 * counts["count"] / totals

    pivot = counts.pivot(index="YEAR", columns="LABOUR_STATUS", values="pct").fillna(0)
    pivot.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="tab20")
    plt.title("Labour Status Composition by Year (Top 5 Categories)")
    plt.xlabel("Year")
    plt.ylabel("Share (%)")
    plt.legend(title="LABOUR_STATUS", bbox_to_anchor=(1.02, 1), loc="upper left")
    p = FIG_DIR / "eda_labour_status_by_year.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    results["labour_status_by_year"] = p.name

    return results


def save_correlation_plot(df: pd.DataFrame) -> dict:
    results = {}

    numeric_cols = ["YEAR", "WEIGHT", "AGE", "GROSS_WEEKLY_PAY", "HOURLY_RATE", "HOURPAY"]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    corr = numeric.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap (Numeric Core Variables)")
    p = FIG_DIR / "eda_numeric_correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    results["numeric_correlation_heatmap"] = p.name

    return results


def save_unsupervised_plot(df: pd.DataFrame) -> dict:
    results = {}

    features = ["AGE", "WEIGHT", "GROSS_WEEKLY_PAY", "HOURLY_RATE", "HOURPAY"]
    work = df[features].apply(pd.to_numeric, errors="coerce")

    # Use a capped sample for clustering speed while keeping broad representation.
    sample_n = min(50000, len(work))
    work = work.sample(n=sample_n, random_state=42)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(work)
    X_scaled = scaler.fit_transform(X_imp)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set2", s=8, alpha=0.5)
    plt.title("KMeans Segments Visualized in PCA Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    p = FIG_DIR / "eda_kmeans_pca_segments.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()

    results["kmeans_pca_segments"] = p.name
    results["kmeans_inertia"] = float(kmeans.inertia_)
    results["pca_explained_variance_ratio"] = [float(x) for x in pca.explained_variance_ratio_]

    return results


def write_summary(df: pd.DataFrame, figure_map: dict, cluster_info: dict) -> None:
    miss = (df.isna().mean() * 100).sort_values(ascending=False)

    age = pd.to_numeric(df["AGE"], errors="coerce")
    pay = pd.to_numeric(df["GROSS_WEEKLY_PAY"], errors="coerce")

    lines = [
        "# Stage 3: Exploratory Data Analysis",
        "",
        "## Methods Applied",
        "1. Distribution and outlier analysis (age and pay).",
        "2. Groupwise comparative analysis (labour-status composition by year).",
        "3. Correlation analysis on numeric features.",
        "4. Unsupervised profiling via KMeans, visualized with PCA.",
        "",
        "## Dataset Snapshot",
        f"- Rows analyzed: {len(df):,}",
        f"- Columns analyzed: {df.shape[1]}",
        f"- AGE mean (std): {age.mean():.2f} ({age.std():.2f})",
        f"- Gross weekly pay mean (std, raw): {pay.mean(skipna=True):.2f} ({pay.std(skipna=True):.2f})",
        "",
        "## Highest Missingness in Core Fields",
        "| Variable | Missing % |",
        "|---|---:|",
    ]

    for k, v in miss.head(8).items():
        lines.append(f"| {k} | {v:.2f} |")

    lines.extend(
        [
            "",
            "## Generated Figures",
            f"- Age distribution: stage3_figures/{figure_map['age_distribution']}",
            f"- Pay boxplot: stage3_figures/{figure_map['pay_boxplot']}",
            f"- Labour status by year: stage3_figures/{figure_map['labour_status_by_year']}",
            f"- Correlation heatmap: stage3_figures/{figure_map['numeric_correlation_heatmap']}",
            f"- KMeans + PCA: stage3_figures/{figure_map['kmeans_pca_segments']}",
            "",
            "## Unsupervised Profiling Notes",
            f"- KMeans inertia (k=4): {cluster_info['kmeans_inertia']:.2f}",
            f"- PCA explained variance ratio (PC1, PC2): {cluster_info['pca_explained_variance_ratio'][0]:.4f}, {cluster_info['pca_explained_variance_ratio'][1]:.4f}",
            "",
            "## Interpretation Pointers for Report",
            "- Distinguish structural missingness from random missingness before drawing causal claims.",
            "- Treat wage-related variables cautiously due to high missingness.",
            "- Use labour-status year composition as a descriptive trend view; predictive claims should wait for formal modeling in Stage 4.",
        ]
    )

    (OUT_DIR / "stage3_eda_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    figs = {}
    figs.update(save_distribution_plots(df))
    figs.update(save_groupwise_plots(df))
    figs.update(save_correlation_plot(df))
    cluster_info = save_unsupervised_plot(df)
    figs["kmeans_pca_segments"] = cluster_info["kmeans_pca_segments"]

    write_summary(df, figs, cluster_info)

    print("Wrote:")
    print((OUT_DIR / "stage3_eda_summary.md").as_posix())
    for name in sorted(FIG_DIR.glob("*.png")):
        print(name.as_posix())


if __name__ == "__main__":
    main()
