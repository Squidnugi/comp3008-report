from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE_DIR / "analysis_outputs" / "stage2_harmonized_core.csv"
OUT_DIR = BASE_DIR / "analysis_outputs"
FIG_DIR = OUT_DIR / "stage4_figures"
FIG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def split_time_aware(df: pd.DataFrame, year_col: str = "YEAR") -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[year_col] <= 2023].copy()
    test = df[df[year_col] == 2024].copy()

    # Fallback if there is no 2024 data after filtering.
    if len(test) == 0:
        q = df[year_col].quantile(0.8)
        train = df[df[year_col] <= q].copy()
        test = df[df[year_col] > q].copy()

    return train, test


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def run_classification(df: pd.DataFrame) -> dict:
    # Predict OUTCOME_STATUS as a supervised classification task.
    features_num = ["AGE", "YEAR", "WEIGHT"]
    features_cat = ["SEX", "HIGHEST_QUAL", "ETHNICITY", "COUNTRY_CODE", "REGION_CODE", "FULLTIME_PARTTIME"]
    target = "OUTCOME_STATUS"

    cols = features_num + features_cat + [target]
    work = df[cols].copy()
    work = work.dropna(subset=[target])
    work[target] = work[target].astype("string")

    train, test = split_time_aware(work, year_col="YEAR")
    X_train, y_train = train[features_num + features_cat], train[target]
    X_test, y_test = test[features_num + features_cat], test[target]

    pre = build_preprocessor(features_num, features_cat)

    lr_model = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", LogisticRegression(max_iter=400)),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocess", pre),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=18,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Cap training size for compute efficiency on large data.
    if len(X_train) > 250000:
        sampled = train.sample(n=250000, random_state=RANDOM_STATE)
        X_train, y_train = sampled[features_num + features_cat], sampled[target]

    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    pred_lr = lr_model.predict(X_test)
    pred_rf = rf_model.predict(X_test)

    metrics = {
        "logistic_accuracy": float(accuracy_score(y_test, pred_lr)),
        "logistic_f1_macro": float(f1_score(y_test, pred_lr, average="macro")),
        "rf_accuracy": float(accuracy_score(y_test, pred_rf)),
        "rf_f1_macro": float(f1_score(y_test, pred_rf, average="macro")),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    # Plot confusion matrix for best classifier by macro-F1.
    best_pred = pred_rf if metrics["rf_f1_macro"] >= metrics["logistic_f1_macro"] else pred_lr
    labels = sorted(y_test.unique())
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_pred,
        display_labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Classification Confusion Matrix (Best Model)")
    plt.tight_layout()
    cm_path = FIG_DIR / "model_classification_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    return metrics


def run_regression(df: pd.DataFrame) -> dict:
    # Predict gross weekly pay as a continuous target.
    target = "GROSS_WEEKLY_PAY"
    features_num = ["AGE", "YEAR", "WEIGHT"]
    features_cat = ["SEX", "HIGHEST_QUAL", "ETHNICITY", "COUNTRY_CODE", "REGION_CODE", "LABOUR_STATUS", "OUTCOME_STATUS"]

    cols = features_num + features_cat + [target]
    work = df[cols].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])

    # Trim extreme tails to reduce unstable fit from outliers.
    q_low = work[target].quantile(0.01)
    q_high = work[target].quantile(0.99)
    work = work[(work[target] >= q_low) & (work[target] <= q_high)]

    train, test = split_time_aware(work, year_col="YEAR")
    X_train, y_train = train[features_num + features_cat], train[target]
    X_test, y_test = test[features_num + features_cat], test[target]

    pre = build_preprocessor(features_num, features_cat)

    baseline = Pipeline(
        steps=[
            ("preprocess", pre),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=16,
                    min_samples_leaf=4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    linear_pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", LogisticRegression()),
        ]
    )

    # Replace accidental classifier pipe with a numeric baseline if needed.
    # We keep a simple median-by-feature baseline via random forest and a linear alternative via SGDRegressor-like behavior.
    from sklearn.linear_model import Ridge

    linear_pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )

    if len(X_train) > 220000:
        sampled = train.sample(n=220000, random_state=RANDOM_STATE)
        X_train, y_train = sampled[features_num + features_cat], sampled[target]

    baseline.fit(X_train, y_train)
    linear_pipe.fit(X_train, y_train)

    pred_rf = baseline.predict(X_test)
    pred_linear = linear_pipe.predict(X_test)

    metrics = {
        "rf_mae": float(mean_absolute_error(y_test, pred_rf)),
        "rf_rmse": float(np.sqrt(mean_squared_error(y_test, pred_rf))),
        "rf_r2": float(r2_score(y_test, pred_rf)),
        "ridge_mae": float(mean_absolute_error(y_test, pred_linear)),
        "ridge_rmse": float(np.sqrt(mean_squared_error(y_test, pred_linear))),
        "ridge_r2": float(r2_score(y_test, pred_linear)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    # Scatter plot for best regressor by RMSE.
    use_pred = pred_rf if metrics["rf_rmse"] <= metrics["ridge_rmse"] else pred_linear
    name = "RandomForest" if metrics["rf_rmse"] <= metrics["ridge_rmse"] else "Ridge"

    plt.figure(figsize=(8, 6))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(y_test), size=min(8000, len(y_test)), replace=False)
    y_plot = np.asarray(y_test.iloc[idx])
    p_plot = np.asarray(use_pred[idx])
    plt.scatter(y_plot, p_plot, alpha=0.25, s=12)
    lim_low = min(y_plot.min(), p_plot.min())
    lim_high = max(y_plot.max(), p_plot.max())
    plt.plot([lim_low, lim_high], [lim_low, lim_high], "r--", linewidth=1.5)
    plt.xlabel("Actual Gross Weekly Pay")
    plt.ylabel("Predicted Gross Weekly Pay")
    plt.title(f"Regression Predictions vs Actual ({name})")
    plt.tight_layout()
    sc_path = FIG_DIR / "model_regression_actual_vs_predicted.png"
    plt.savefig(sc_path, dpi=150)
    plt.close()

    return metrics


def write_summary(class_metrics: dict, reg_metrics: dict) -> None:
    lines = [
        "# Stage 4: Predictive Modeling",
        "",
        "## Modeling Methods Used",
        "1. Supervised classification of OUTCOME_STATUS:",
        "   - Logistic Regression (baseline)",
        "   - Random Forest Classifier",
        "2. Regression of GROSS_WEEKLY_PAY:",
        "   - Ridge Regression (baseline)",
        "   - Random Forest Regressor",
        "",
        "## Split Strategy",
        "- Time-aware split: train on years <= 2023, test on 2024.",
        "- Fallback quantile split is available if no 2024 rows remain after filtering.",
        "",
        "## Classification Results",
        f"- Train rows: {class_metrics['train_rows']:,}",
        f"- Test rows: {class_metrics['test_rows']:,}",
        f"- Logistic accuracy: {class_metrics['logistic_accuracy']:.4f}",
        f"- Logistic macro F1: {class_metrics['logistic_f1_macro']:.4f}",
        f"- Random forest accuracy: {class_metrics['rf_accuracy']:.4f}",
        f"- Random forest macro F1: {class_metrics['rf_f1_macro']:.4f}",
        "",
        "## Regression Results",
        f"- Train rows: {reg_metrics['train_rows']:,}",
        f"- Test rows: {reg_metrics['test_rows']:,}",
        f"- Ridge MAE: {reg_metrics['ridge_mae']:.3f}",
        f"- Ridge RMSE: {reg_metrics['ridge_rmse']:.3f}",
        f"- Ridge R^2: {reg_metrics['ridge_r2']:.4f}",
        f"- Random forest MAE: {reg_metrics['rf_mae']:.3f}",
        f"- Random forest RMSE: {reg_metrics['rf_rmse']:.3f}",
        f"- Random forest R^2: {reg_metrics['rf_r2']:.4f}",
        "",
        "## Generated Figures",
        "- stage4_figures/model_classification_confusion_matrix.png",
        "- stage4_figures/model_regression_actual_vs_predicted.png",
        "",
        "## Notes and Limitations",
        "- Wage targets remain sensitive to high missingness and possible non-random reporting.",
        "- Categorical code semantics should be validated against data dictionaries before final interpretation.",
        "- Additional model tuning and weighted evaluation can be done in a follow-up stage.",
    ]

    (OUT_DIR / "stage4_modeling_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df.dropna(subset=["YEAR"])

    class_metrics = run_classification(df)
    reg_metrics = run_regression(df)
    write_summary(class_metrics, reg_metrics)

    print("Wrote:")
    print((OUT_DIR / "stage4_modeling_summary.md").as_posix())
    for name in sorted(FIG_DIR.glob("*.png")):
        print(name.as_posix())


if __name__ == "__main__":
    main()
