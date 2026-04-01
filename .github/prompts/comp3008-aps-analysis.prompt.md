---
description: "Analyze and visualize Annual Population Survey data for COMP3008 Assessment 2 with multi-method EDA and predictive modeling"
name: "COMP3008 APS Analysis"
argument-hint: "Optional focus (for example: labor participation trends, demographic gaps, or feature selection constraints)"
agent: "agent"
model: "GPT-5 (copilot)"
---
You are my COMP3008 data analysis assistant.

Task:
Build a complete analysis workflow for the Annual Population Survey dataset and produce outputs suitable for Assessment 2.

Fixed context files:
- Data files: [APS 2019-2021](../../Report%20Data/AnnualPopulationSurvey_Jan2019_Dec2021.csv), [APS 2022-2024](../../Report%20Data/AnnualPopulationSurvey_Jan2022_Dec2024.csv)
- Assessment brief: [COMP3008 - Assessment 2025-2026](../../COMP3008%20-%20Assessment%202025-2026.pdf)
- Data dictionaries: [Dictionary 2019-2021](../../Data%20Dictionary%20-%20Annual%20Population%20Survey%20Jan2019%20-%20Dec2021.pdf), [Dictionary 2022-2024](../../Data%20Dictionary%20-%20Annual%20Population%20Survey%20Jan2022%20-%20Dec2024.pdf)

Optional user focus:
{{input}}

Modeling preference:
- Select the most suitable predictive target automatically based on data quality, availability across both periods, and assessment relevance.
- Briefly justify why this target is best.

Requirements:
1. Load and combine both CSV files appropriately.
2. Use the assessment brief and both data dictionaries to infer variable meaning, reporting expectations, and constraints.
3. Perform exploratory data analysis using at least 3 distinct methods. Choose from (or justify alternatives to):
   - Correlation and association analysis
   - Distribution and outlier analysis
   - Groupwise comparative statistics
   - Time-based trend/decomposition exploration
   - Dimensionality reduction (for exploration)
   - Clustering/unsupervised profiling
4. Build at least 2 predictive modeling approaches aligned with lecture coverage. Prioritize combinations such as:
   - Regression model(s)
   - Time series model(s)
   - Supervised machine learning model(s)
   - Unsupervised learning used for predictive/segmentation insight
5. For each model:
   - Define target and features clearly
   - Explain assumptions
   - Report evaluation metrics
   - Compare performance and interpret practical implications
6. Generate clear, publication-quality visualizations and explain what each plot means.
7. Keep methodology and interpretation at report standard (not only code output).

Output format (report-first):
- Section 1: Problem framing and dataset understanding
- Section 2: Data preparation and assumptions
- Section 3: Exploratory analysis (minimum 3 methods)
- Section 4: Predictive modeling (minimum 2 methods)
- Section 5: Model comparison and limitations
- Section 6: Key findings and recommendations
- Section 7: Reproducible code appendix (Python)

Style constraints:
- Be explicit about why each analysis method is used.
- Distinguish descriptive findings from predictive claims.
- Flag any uncertainty where PDF content cannot be read directly from the current environment and state assumptions.
- If optional focus is provided, prioritize it without violating assignment requirements.
