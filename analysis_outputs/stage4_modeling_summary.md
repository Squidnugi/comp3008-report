# Stage 4: Predictive Modeling

## Modeling Methods Used
1. Supervised classification of OUTCOME_STATUS:
   - Logistic Regression (baseline)
   - Random Forest Classifier
2. Regression of GROSS_WEEKLY_PAY:
   - Ridge Regression (baseline)
   - Random Forest Regressor

## Split Strategy
- Time-aware split: train on years <= 2023, test on 2024.
- Fallback quantile split is available if no 2024 rows remain after filtering.

## Classification Results
- Train rows: 250,000
- Test rows: 114,375
- Logistic accuracy: 0.9956
- Logistic macro F1: 0.7963
- Random forest accuracy: 0.9994
- Random forest macro F1: 0.9993

## Regression Results
- Train rows: 174,079
- Test rows: 33,656
- Ridge MAE: 157.385
- Ridge RMSE: 193.534
- Ridge R^2: 0.2194
- Random forest MAE: 149.240
- Random forest RMSE: 182.614
- Random forest R^2: 0.3050

## Generated Figures
- stage4_figures/model_classification_confusion_matrix.png
- stage4_figures/model_regression_actual_vs_predicted.png

## Notes and Limitations
- Wage targets remain sensitive to high missingness and possible non-random reporting.
- Categorical code semantics should be validated against data dictionaries before final interpretation.
- Additional model tuning and weighted evaluation can be done in a follow-up stage.