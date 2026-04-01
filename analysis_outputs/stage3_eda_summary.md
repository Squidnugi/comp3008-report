# Stage 3: Exploratory Data Analysis

## Methods Applied
1. Distribution and outlier analysis (age and pay).
2. Groupwise comparative analysis (labour-status composition by year).
3. Correlation analysis on numeric features.
4. Unsupervised profiling via KMeans, visualized with PCA.

## Dataset Snapshot
- Rows analyzed: 750,936
- Columns analyzed: 21
- AGE mean (std): 45.19 (24.16)
- Gross weekly pay mean (std, raw): 510.04 (232.06)

## Highest Missingness in Core Fields
| Variable | Missing % |
|---|---:|
| HOURLY_RATE | 88.81 |
| HOURPAY | 72.22 |
| GROSS_WEEKLY_PAY | 72.06 |
| HEALTH_LIMITATION | 62.69 |
| FULLTIME_PARTTIME | 54.86 |
| HIGHEST_QUAL | 35.01 |
| COMBINED_AUTHORITY | 32.54 |
| ETHNICITY | 16.24 |

## Generated Figures
- Age distribution: stage3_figures/eda_age_distribution.png
- Pay boxplot: stage3_figures/eda_pay_boxplot.png
- Labour status by year: stage3_figures/eda_labour_status_by_year.png
- Correlation heatmap: stage3_figures/eda_numeric_correlation_heatmap.png
- KMeans + PCA: stage3_figures/eda_kmeans_pca_segments.png

## Unsupervised Profiling Notes
- KMeans inertia (k=4): 116204.12
- PCA explained variance ratio (PC1, PC2): 0.3320, 0.2151

## Interpretation Pointers for Report
- Distinguish structural missingness from random missingness before drawing causal claims.
- Treat wage-related variables cautiously due to high missingness.
- Use labour-status year composition as a descriptive trend view; predictive claims should wait for formal modeling in Stage 4.