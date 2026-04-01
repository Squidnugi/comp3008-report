# Stage 2: Variable Selection Log

## Scope
- This log is generated from observed data structure and missingness.
- Assessment brief and data dictionaries should be used to refine code meanings in narrative write-up.
- Where dictionary interpretation is uncertain, variables are retained by statistical utility rather than semantic certainty.

## Column Mapping Decisions
| Canonical | Raw Options | Raw Used |
|---|---|---|
| PERSON_ID | IDREF | IDREF |
| YEAR | FILEYEAR | FILEYEAR |
| WEIGHT | NPWT22C | NPWT22C |
| COUNTRY_CODE | CTRY9D | CTRY9D |
| REGION_CODE | GOR9D, GOR9DCENSUS2021 | GOR9D |
| COMBINED_AUTHORITY | COMBINEDAUTHORITIES, COMBINEDAUTHORITIESCENSUS2021 | COMBINEDAUTHORITIES |
| ITL2_CODE | ITL221, ITL221CENSUS2021, ITL225CENSUS2021 | ITL221 |
| ITL3_CODE | ITL321, ITL321CENSUS2021, ITL325CENSUS2021 | ITL321 |
| AGE | AGE | AGE |
| SEX | SEX | SEX |
| LABOUR_STATUS | ILODEFR | ILODEFR |
| OUTCOME_STATUS | IOUTCOME | IOUTCOME |
| FULLTIME_PARTTIME | FTPT | FTPT |
| GROSS_WEEKLY_PAY | GRSSWK | GRSSWK |
| HOURLY_RATE | HRRATE | HRRATE |
| HOURPAY | HOURPAY | HOURPAY |
| HIGHEST_QUAL | HIQUAL15, HIQUAL22 | HIQUAL15 |
| HEALTH_LIMITATION | HEALYR | HEALYR |
| ETHNICITY | ETH11EW | ETH11EW |
| COUNTRY_NAME | COUNTRY | COUNTRY |
| REGION_CODE | GOR9D, GOR9DCENSUS2021 | GOR9DCENSUS2021 |
| COMBINED_AUTHORITY | COMBINEDAUTHORITIES, COMBINEDAUTHORITIESCENSUS2021 | COMBINEDAUTHORITIESCENSUS2021 |
| ITL2_CODE | ITL221, ITL221CENSUS2021, ITL225CENSUS2021 | ITL221CENSUS2021, ITL225CENSUS2021 |
| ITL3_CODE | ITL321, ITL321CENSUS2021, ITL325CENSUS2021 | ITL321CENSUS2021, ITL325CENSUS2021 |
| HIGHEST_QUAL | HIQUAL15, HIQUAL22 | HIQUAL15, HIQUAL22 |

## Selected Variables (Initial Core)
| Variable | Missing % | Rationale |
|---|---:|---|
| PERSON_ID | 0.00 | Kept as potentially useful contextual feature. |
| SOURCE_PERIOD | 0.00 | Critical provenance field for split-period comparisons. |
| YEAR | 0.00 | Core explanatory/outcome candidate for EDA and modeling. |
| WEIGHT | 0.00 | Kept as potentially useful contextual feature. |
| AGE | 0.00 | Core explanatory/outcome candidate for EDA and modeling. |
| SEX | 0.00 | Kept as potentially useful contextual feature. |
| LABOUR_STATUS | 0.00 | Core explanatory/outcome candidate for EDA and modeling. |
| OUTCOME_STATUS | 0.00 | Core explanatory/outcome candidate for EDA and modeling. |
| FULLTIME_PARTTIME | 54.86 | Core explanatory/outcome candidate for EDA and modeling. |
| GROSS_WEEKLY_PAY | 72.06 | Core explanatory/outcome candidate for EDA and modeling. |
| HOURLY_RATE | 88.81 | Core explanatory/outcome candidate for EDA and modeling. |
| HOURPAY | 72.22 | Kept as potentially useful contextual feature. |
| HIGHEST_QUAL | 35.01 | Core explanatory/outcome candidate for EDA and modeling. |
| HEALTH_LIMITATION | 62.69 | Kept as potentially useful contextual feature. |
| ETHNICITY | 16.24 | Kept as potentially useful contextual feature. |
| COUNTRY_NAME | 0.00 | Kept as potentially useful contextual feature. |
| COUNTRY_CODE | 0.00 | Kept as potentially useful contextual feature. |
| REGION_CODE | 0.00 | Kept as potentially useful contextual feature. |
| COMBINED_AUTHORITY | 32.54 | Kept as potentially useful contextual feature. |
| ITL2_CODE | 0.00 | Kept as potentially useful contextual feature. |
| ITL3_CODE | 0.00 | Kept as potentially useful contextual feature. |

## Excluded from Initial Core
| Variable | Missing % | Rationale |
|---|---:|---|

## Stage 2 Assumptions
- YEAR is treated as numeric calendar year derived from FILEYEAR.
- Geographic columns are harmonized across census naming changes via coalescing.
- Semantic recoding of categorical codes (for example, specific ILODEFR classes) is deferred until dictionary-driven modeling setup.