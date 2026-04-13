from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")

try:
    from pmdarima import auto_arima

    ARIMA_AVAILABLE = True
except ImportError:
    auto_arima = None
    ARIMA_AVAILABLE = False

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Report_Data"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / "Report Data"
OUT_DIR = BASE_DIR / "analysis_outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

FILE_A = DATA_DIR / "AnnualPopulationSurvey_Jan2019_Dec2021.csv"
FILE_B = DATA_DIR / "AnnualPopulationSurvey_Jan2022_Dec2024.csv"
HARMONISED_PATH = DATA_DIR / "harmonised_aps_data.csv"

STATUS_LABELS = {
    "1": "In Employment",
    "2": "ILO Unemployed",
    "3": "Economically Inactive",
    "4": "Government-Supported Training",
}

REGION_NAMES = {
    "E12000001": "North East",
    "E12000002": "North West",
    "E12000003": "Yorkshire & Humber",
    "E12000004": "East Midlands",
    "E12000005": "West Midlands",
    "E12000006": "East of England",
    "E12000007": "London",
    "E12000008": "South East",
    "E12000009": "South West",
    "W92000004": "Wales",
    "S92000003": "Scotland",
    "N92000002": "Northern Ireland",
}

CLUSTER_FEATURES = ["AGE", "WEIGHT", "GROSS_WEEKLY_PAY", "HOURLY_RATE"]
NUM_FEATURES = ["AGE", "YEAR", "WEIGHT"]
CAT_FEATURES = ["SEX", "HIGHEST_QUAL", "ETHNICITY", "COUNTRY_CODE", "REGION_CODE", "LABOUR_STATUS"]
TARGET = "GROSS_WEEKLY_PAY"
RANDOM_STATE = 42


def load_csv(path: Path, period_label: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.upper()
    df["SOURCE_PERIOD"] = period_label
    return df


def coalesce(df: pd.DataFrame, *cols: str) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(pd.NA, index=df.index)

    result = df[existing[0]].copy()
    for col in existing[1:]:
        result = result.fillna(df[col])
    return result


def build_reg_pipeline(model) -> Pipeline:
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", num_pipe, NUM_FEATURES),
            ("cat", cat_pipe, CAT_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def harmonise_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()

    df["PERSON_ID"] = coalesce(raw, "IDREF")
    df["SOURCE_PERIOD"] = raw["SOURCE_PERIOD"]
    df["YEAR"] = pd.to_numeric(coalesce(raw, "FILEYEAR"), errors="coerce")
    df["WEIGHT"] = pd.to_numeric(coalesce(raw, "NPWT22C"), errors="coerce")
    df["AGE"] = pd.to_numeric(coalesce(raw, "AGE"), errors="coerce")
    df["SEX"] = coalesce(raw, "SEX").astype("string")
    df["LABOUR_STATUS"] = coalesce(raw, "ILODEFR").astype("string")
    df["OUTCOME_STATUS"] = coalesce(raw, "IOUTCOME").astype("string")
    df["FULLTIME_PARTTIME"] = coalesce(raw, "FTPT").astype("string")
    df["GROSS_WEEKLY_PAY"] = pd.to_numeric(coalesce(raw, "GRSSWK"), errors="coerce")
    df["HOURLY_RATE"] = pd.to_numeric(coalesce(raw, "HRRATE"), errors="coerce")
    df["HIGHEST_QUAL"] = coalesce(raw, "HIQUAL15", "HIQUAL22").astype("string")
    df["HEALTH_LIMITATION"] = coalesce(raw, "HEALYR").astype("string")
    df["ETHNICITY"] = coalesce(raw, "ETH11EW").astype("string")
    df["COUNTRY_CODE"] = coalesce(raw, "CTRY9D").astype("string")
    df["COUNTRY_NAME"] = coalesce(raw, "COUNTRY").astype("string")
    df["REGION_CODE"] = coalesce(raw, "GOR9D", "GOR9DCENSUS2021").astype("string")
    df["COMBINED_AUTHORITY"] = coalesce(
        raw,
        "COMBINEDAUTHORITIES",
        "COMBINEDAUTHORITIESCENSUS2021",
    ).astype("string")
    df["ITL2_CODE"] = coalesce(raw, "ITL221", "ITL221CENSUS2021", "ITL225CENSUS2021").astype("string")
    df["ITL3_CODE"] = coalesce(raw, "ITL321", "ITL321CENSUS2021", "ITL325CENSUS2021").astype("string")

    return df


def save_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = (df.isna().mean() * 100).sort_values(ascending=False).round(2)
    miss_df = miss.reset_index()
    miss_df.columns = ["Variable", "Missing %"]
    miss_df.to_csv(OUT_DIR / "stage2_missingness_core.csv", index=False)
    return miss_df


def print_data_quality(df: pd.DataFrame) -> None:
    print("=== Data Quality Summary ===")
    print(f"Total rows            : {len(df):,}")
    print(f"Duplicate rows        : {df.duplicated().sum():,}")
    if df["YEAR"].notna().any():
        print(f"Year range            : {int(df['YEAR'].min())} - {int(df['YEAR'].max())}")
    else:
        print("Year range            : unavailable")
    print("Rows per year:")
    print(df["YEAR"].value_counts().sort_index().to_string())
    print(f"\nUnique REGION_CODE values : {df['REGION_CODE'].nunique()}")
    print(f"Unique COUNTRY_NAME values: {df['COUNTRY_NAME'].dropna().unique()}")


def save_fig(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda_descriptives(df: pd.DataFrame) -> None:
    numeric_vars = ["AGE", "GROSS_WEEKLY_PAY", "HOURLY_RATE", "WEIGHT"]
    print(df[numeric_vars].describe().round(2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    age = df["AGE"].dropna()
    sns.histplot(age, bins=40, kde=True, ax=axes[0], color="#4878CF")
    axes[0].set_title("Age Distribution of APS Respondents (2019-2024)")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].axvline(age.mean(), color="red", linestyle="--", label=f"Mean: {age.mean():.1f}")
    axes[0].legend()

    pay = df["GROSS_WEEKLY_PAY"].dropna()
    pay_trimmed = pay[pay <= pay.quantile(0.99)]
    sns.histplot(pay_trimmed, bins=50, kde=True, ax=axes[1], color="#6ACC65")
    axes[1].set_title("Gross Weekly Pay Distribution (trimmed at 99th pct)")
    axes[1].set_xlabel("Gross Weekly Pay (£)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(pay_trimmed.median(), color="red", linestyle="--", label=f"Median: £{pay_trimmed.median():.0f}")
    axes[1].legend()

    plt.tight_layout()
    save_fig(fig, "eda1_distributions.png")


def run_labour_status_timeseries(df: pd.DataFrame) -> None:
    ts = df.dropna(subset=["YEAR", "LABOUR_STATUS"]).copy()
    ts["YEAR"] = ts["YEAR"].astype(int)
    ts["STATUS_LABEL"] = ts["LABOUR_STATUS"].map(STATUS_LABELS).fillna("Other")

    grouped = ts.groupby(["YEAR", "STATUS_LABEL"]).size().reset_index(name="count")
    totals = grouped.groupby("YEAR")["count"].transform("sum")
    grouped["pct"] = 100 * grouped["count"] / totals
    pivot = grouped.pivot(index="YEAR", columns="STATUS_LABEL", values="pct").fillna(0)

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(ax=ax, marker="o", linewidth=2)
    ax.axvspan(2019.5, 2021.5, alpha=0.08, color="red", label="COVID period (2020-2021)")
    ax.set_title("Labour Status Composition by Year (2019-2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of Respondents (%)")
    ax.set_xticks(sorted(ts["YEAR"].unique()))
    ax.legend(title="Labour Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    save_fig(fig, "eda2_labour_status_timeseries.png")


def run_regional_comparison(df: pd.DataFrame) -> None:
    reg = df.dropna(subset=["YEAR", "REGION_CODE", "LABOUR_STATUS"]).copy()
    reg["YEAR"] = reg["YEAR"].astype(int)
    reg["REGION_NAME"] = reg["REGION_CODE"].map(REGION_NAMES).fillna(reg["REGION_CODE"].astype(str))
    reg["EMPLOYED"] = reg["LABOUR_STATUS"].isin(["1", "4"]).astype(int)

    emp_rate = (
        reg.groupby(["YEAR", "REGION_NAME"])["EMPLOYED"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"EMPLOYED": "Employment Rate (%)"})
    )

    pivot_reg = emp_rate.pivot(index="REGION_NAME", columns="YEAR", values="Employment Rate (%)")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        pivot_reg,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Employment Rate (%)"},
    )
    ax.set_title("Employment Rate (%) by Region and Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Region")
    plt.tight_layout()
    save_fig(fig, "eda3_regional_employment_heatmap.png")


def run_kmeans_clustering(df: pd.DataFrame) -> None:
    cluster_work = df[CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce")
    sample_n = min(50_000, len(cluster_work))
    cluster_sample = cluster_work.sample(n=sample_n, random_state=RANDOM_STATE)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(cluster_sample))

    inertias = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(x)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker="o")
    ax.set_title("Elbow Plot - Choosing Number of Clusters (k)")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia")
    plt.tight_layout()
    save_fig(fig, "eda4a_elbow.png")

    k_chosen = 4
    kmeans = KMeans(n_clusters=k_chosen, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(x)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    x_2d = pca.fit_transform(x)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=labels, cmap="Set2", s=8, alpha=0.5)
    ax.set_title(f"K-Means Clusters (k={k_chosen}) - Visualised in PCA Space")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    legend_handles, _ = scatter.legend_elements()
    ax.legend(legend_handles, [f"Cluster {i}" for i in range(k_chosen)], title="Cluster", loc="best")
    plt.tight_layout()
    save_fig(fig, "eda4b_kmeans_pca.png")

    cluster_sample_with_labels = cluster_sample.copy()
    cluster_sample_with_labels["Cluster"] = labels
    print("\nCluster centroids (original scale):")
    print(cluster_sample_with_labels.groupby("Cluster").mean(numeric_only=True).round(2))


def run_arima_forecast(df: pd.DataFrame) -> tuple[float | None, pd.Series | None]:
    if not ARIMA_AVAILABLE:
        print("Skipping ARIMA - install pmdarima first: pip install pmdarima")
        return None, None

    arima_df = df.dropna(subset=["YEAR", "LABOUR_STATUS"]).copy()
    arima_df["YEAR"] = arima_df["YEAR"].astype(int)
    arima_df["EMPLOYED"] = arima_df["LABOUR_STATUS"].isin(["1", "4"]).astype(int)

    annual_emp = arima_df.groupby("YEAR")["EMPLOYED"].mean().mul(100).sort_index()
    print("Annual employment rate series:")
    print(annual_emp.round(2).to_string())

    train_ts = annual_emp[annual_emp.index <= 2022]
    test_ts = annual_emp[annual_emp.index >= 2023]
    future_horizon = 3
    last_year = int(annual_emp.index.max())
    future_years = np.arange(last_year + 1, last_year + 1 + future_horizon)

    try:
        backtest_model = auto_arima(
            train_ts,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        print(f"\nSelected ARIMA order (backtest): {backtest_model.order}")
        test_forecast = pd.Series(backtest_model.predict(n_periods=len(test_ts)), index=test_ts.index)
        test_forecast = pd.to_numeric(test_forecast, errors="coerce")
        if test_forecast.isna().any():
            fallback_value = float(train_ts.dropna().iloc[-1]) if train_ts.dropna().size > 0 else float(annual_emp.dropna().iloc[-1])
            test_forecast = test_forecast.fillna(fallback_value)

        # Refit on all observed years, then forecast beyond 2024.
        full_model = auto_arima(
            annual_emp,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        future_pred, future_ci = full_model.predict(n_periods=future_horizon, return_conf_int=True)
        future_forecast = pd.Series(future_pred, index=future_years)
        future_lower = pd.Series(future_ci[:, 0], index=future_years)
        future_upper = pd.Series(future_ci[:, 1], index=future_years)
        future_forecast = pd.to_numeric(future_forecast, errors="coerce")
        future_lower = pd.to_numeric(future_lower, errors="coerce")
        future_upper = pd.to_numeric(future_upper, errors="coerce")
        if future_forecast.isna().any():
            fallback_value = float(annual_emp.dropna().iloc[-1])
            future_forecast = future_forecast.fillna(fallback_value)
        future_lower = future_lower.fillna(future_forecast)
        future_upper = future_upper.fillna(future_forecast)
    except Exception as exc:
        print(f"ARIMA fitting failed: {exc}")
        test_forecast = pd.Series([train_ts.iloc[-1]] * len(test_ts), index=test_ts.index)
        future_forecast = pd.Series([annual_emp.iloc[-1]] * future_horizon, index=future_years)
        future_lower = future_forecast.copy()
        future_upper = future_forecast.copy()
        print("Using a naive persistence forecast instead.")

    eval_df = pd.concat([test_ts.rename("actual"), test_forecast.rename("pred")], axis=1).dropna()
    if len(eval_df) == 0:
        arima_mae = float("nan")
        print("ARIMA MAE on 2023-2024 backtest: unavailable (no valid overlap after cleaning)")
    else:
        arima_mae = mean_absolute_error(eval_df["actual"], eval_df["pred"])
        print(f"ARIMA MAE on 2023-2024 backtest: {arima_mae:.3f} percentage points")
    print("Future ARIMA forecast:")
    print(future_forecast.round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(annual_emp.index, annual_emp.values, marker="o", label="Actual", color="#4878CF")
    ax.plot(
        test_ts.index,
        test_forecast.values,
        marker="s",
        linestyle="--",
        label="ARIMA Backtest Forecast (2023-2024)",
        color="#E06C75",
    )
    ax.plot(
        future_years,
        future_forecast.values,
        marker="D",
        linestyle="--",
        linewidth=2,
        label=f"ARIMA Future Forecast ({future_years[0]}-{future_years[-1]})",
        color="#55A868",
    )
    ax.fill_between(
        future_years,
        future_lower.values,
        future_upper.values,
        color="#55A868",
        alpha=0.15,
        label="Future forecast interval",
    )
    ax.axvspan(2022.5, 2024.5, alpha=0.07, color="grey", label="Backtest window")
    ax.axvspan(future_years[0] - 0.5, future_years[-1] + 0.5, alpha=0.07, color="#55A868", label="Future window")
    ax.set_title("ARIMA Forecast of UK Employment Rate (Annual)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Employment Rate (%)")
    ax.set_xticks(sorted(list(annual_emp.index) + list(future_years)))
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "model1_arima_forecast.png")

    return float(arima_mae), annual_emp


def prepare_regression_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    model_df = df[NUM_FEATURES + CAT_FEATURES + [TARGET]].copy()
    model_df[TARGET] = pd.to_numeric(model_df[TARGET], errors="coerce")
    model_df = model_df.dropna(subset=[TARGET])

    lo, hi = model_df[TARGET].quantile([0.01, 0.99])
    model_df = model_df[(model_df[TARGET] >= lo) & (model_df[TARGET] <= hi)]

    print(f"Rows available for regression: {len(model_df):,}")
    print(f"Target range after trimming: £{lo:.2f} - £{hi:.2f}")

    train = model_df[model_df["YEAR"] <= 2023]
    test = model_df[model_df["YEAR"] == 2024]

    if len(test) < 100:
        split_year = model_df["YEAR"].quantile(0.8)
        train = model_df[model_df["YEAR"] <= split_year]
        test = model_df[model_df["YEAR"] > split_year]
        print(f"Fallback split used: train <= {split_year}, test > {split_year}")

    X_train = train[NUM_FEATURES + CAT_FEATURES]
    y_train = train[TARGET]
    X_test = test[NUM_FEATURES + CAT_FEATURES]
    y_test = test[TARGET]

    print(f"\nTrain rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    return X_train, y_train, X_test, y_test, model_df


def run_ridge_regression(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[Pipeline, dict]:
    ridge_pipe = build_reg_pipeline(Ridge(alpha=1.0))

    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    for col in CAT_FEATURES:
        X_train_clean[col] = X_train_clean[col].astype("object").replace({pd.NA: np.nan})
        X_test_clean[col] = X_test_clean[col].astype("object").replace({pd.NA: np.nan})

    ridge_pipe.fit(X_train_clean, y_train)
    pred_ridge = ridge_pipe.predict(X_test_clean)

    ridge_metrics = {
        "MAE": mean_absolute_error(y_test, pred_ridge),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_ridge)),
        "R2": r2_score(y_test, pred_ridge),
    }
    print("Ridge Regression metrics:")
    for key, value in ridge_metrics.items():
        print(f"  {key}: {value:.4f}")

    return ridge_pipe, ridge_metrics


def run_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[Pipeline, dict]:
    max_train = 200_000

    X_train_rf = X_train.copy()
    X_test_rf = X_test.copy()
    for col in CAT_FEATURES:
        X_train_rf[col] = X_train_rf[col].astype("object").replace({pd.NA: np.nan})
        X_test_rf[col] = X_test_rf[col].astype("object").replace({pd.NA: np.nan})

    if len(X_train_rf) > max_train:
        x_tr_sub = X_train_rf.sample(n=max_train, random_state=RANDOM_STATE)
        y_tr_sub = y_train.loc[x_tr_sub.index]
    else:
        x_tr_sub, y_tr_sub = X_train_rf, y_train

    rf_pipe = build_reg_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    rf_pipe.fit(x_tr_sub, y_tr_sub)

    pred_rf = rf_pipe.predict(X_test_rf)
    rf_metrics = {
        "MAE": mean_absolute_error(y_test, pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_rf)),
        "R2": r2_score(y_test, pred_rf),
    }
    print("Random Forest metrics:")
    for key, value in rf_metrics.items():
        print(f"  {key}: {value:.4f}")

    return rf_pipe, rf_metrics


def save_rf_diagnostics(rf_pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    X_test_rf = X_test.copy()
    for col in CAT_FEATURES:
        X_test_rf[col] = X_test_rf[col].astype("object").replace({pd.NA: np.nan})

    pred_rf = rf_pipe.predict(X_test_rf)

    rng = np.random.default_rng(RANDOM_STATE)
    plot_idx = rng.choice(len(y_test), size=min(6000, len(y_test)), replace=False)
    y_plot = np.array(y_test.iloc[plot_idx])
    p_plot = pred_rf[plot_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_plot, p_plot, alpha=0.2, s=10, color="#4878CF")
    lim = [min(y_plot.min(), p_plot.min()), max(y_plot.max(), p_plot.max())]
    axes[0].plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Gross Weekly Pay (£)")
    axes[0].set_ylabel("Predicted Gross Weekly Pay (£)")
    axes[0].set_title("Random Forest: Actual vs Predicted Pay")
    axes[0].legend()

    residuals = p_plot - y_plot
    axes[1].scatter(y_plot, residuals, alpha=0.2, s=10, color="#E06C75")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Actual Gross Weekly Pay (£)")
    axes[1].set_ylabel("Residual (Predicted - Actual)")
    axes[1].set_title("Random Forest: Residual Plot")

    plt.tight_layout()
    save_fig(fig, "model3_rf_actual_vs_predicted.png")

    pre_step = rf_pipe.named_steps["pre"]
    cat_names = pre_step.named_transformers_["cat"]["encode"].get_feature_names_out(CAT_FEATURES)
    all_names = NUM_FEATURES + list(cat_names)
    importances = rf_pipe.named_steps["model"].feature_importances_

    imp_df = (
        pd.DataFrame({"Feature": all_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax, palette="viridis")
    ax.set_title("Top 15 Feature Importances - Random Forest (Gross Weekly Pay)")
    ax.set_xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    save_fig(fig, "model3_rf_feature_importance.png")


def write_model_comparison(arima_mae: float | None, ridge_metrics: dict, rf_metrics: dict) -> None:
    rows = [
        {
            "Model": "ARIMA",
            "Target": "Annual employment rate (%)",
            "MAE": f"{arima_mae:.3f} pp" if arima_mae is not None else "N/A",
            "RMSE": "N/A (univariate TS)",
            "R²": "N/A (univariate TS)",
            "Suitability": "Appropriate for temporal trend; limited by small series (n=6).",
        },
        {
            "Model": "Ridge Regression",
            "Target": "Gross weekly pay (£)",
            "MAE": f"£{ridge_metrics['MAE']:.2f}",
            "RMSE": f"£{ridge_metrics['RMSE']:.2f}",
            "R²": f"{ridge_metrics['R2']:.4f}",
            "Suitability": "Interpretable linear baseline; assumes linearity which may not hold.",
        },
        {
            "Model": "Random Forest",
            "Target": "Gross weekly pay (£)",
            "MAE": f"£{rf_metrics['MAE']:.2f}",
            "RMSE": f"£{rf_metrics['RMSE']:.2f}",
            "R²": f"{rf_metrics['R2']:.4f}",
            "Suitability": "Captures non-linear patterns; less interpretable, higher compute cost.",
        },
    ]

    comparison_df = pd.DataFrame(rows).set_index("Model")
    comparison_df.to_csv(OUT_DIR / "model_comparison.csv")
    print("Model Comparison Table:")
    print(comparison_df)


def main() -> None:
    print("Loading source files...")
    df_a = load_csv(FILE_A, "2019-2021")
    df_b = load_csv(FILE_B, "2022-2024")

    print(f"File A shape: {df_a.shape}  ({df_a.shape[0]:,} rows, {df_a.shape[1]} columns)")
    print(f"File B shape: {df_b.shape}  ({df_b.shape[0]:,} rows, {df_b.shape[1]} columns)")

    raw = pd.concat([df_a, df_b], ignore_index=True, sort=False)
    print(f"\nCombined shape: {raw.shape}  ({raw.shape[0]:,} rows, {raw.shape[1]} columns)")

    df = harmonise_data(raw)
    del raw, df_a, df_b

    print(f"Harmonised DataFrame: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.dtypes.to_string())
    print("Checks:")
    print(df["REGION_CODE"].value_counts().head(20))
    print(df["LABOUR_STATUS"].value_counts().head(10))

    df.to_csv(HARMONISED_PATH, index=False)
    print(f"Saved -> {HARMONISED_PATH}")

    miss_df = save_missingness_table(df)
    print(miss_df.head(20).to_string(index=False))

    print_data_quality(df)

    run_eda_descriptives(df)
    run_labour_status_timeseries(df)
    run_regional_comparison(df)
    run_kmeans_clustering(df)

    arima_mae, _ = run_arima_forecast(df)

    X_train, y_train, X_test, y_test, _ = prepare_regression_frame(df)
    ridge_pipe, ridge_metrics = run_ridge_regression(X_train, y_train, X_test, y_test)
    rf_pipe, rf_metrics = run_random_forest(X_train, y_train, X_test, y_test)
    save_rf_diagnostics(rf_pipe, X_test, y_test)

    write_model_comparison(arima_mae, ridge_metrics, rf_metrics)

    print("\nDone.")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
