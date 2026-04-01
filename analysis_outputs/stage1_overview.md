# Stage 1: Dataset Understanding

## Files Profiled
- X:/COMP3008 Report/Report Data/AnnualPopulationSurvey_Jan2019_Dec2021.csv
- X:/COMP3008 Report/Report Data/AnnualPopulationSurvey_Jan2022_Dec2024.csv

## High-Level Shape
### 2019-2021
- Rows: 430,347
- Columns (including SOURCE_PERIOD): 527
- Duplicate rows: 0
- FILEYEAR span: 2019 to 2021
- Top 10 missing variables:
| Variable | Missing % |
|---|---:|
| UNDY989 | 100.00 |
| UNDY987 | 100.00 |
| UNDY988 | 100.00 |
| UNDY986 | 100.00 |
| TPBN1309 | 100.00 |
| ERNCM10 | 100.00 |
| TPBN1310 | 100.00 |
| TPBN1307 | 100.00 |
| TPBN1308 | 100.00 |
| JB2T105 | 100.00 |

### 2022-2024
- Rows: 320,589
- Columns (including SOURCE_PERIOD): 474
- Duplicate rows: 0
- FILEYEAR span: 2022 to 2024
- Top 10 missing variables:
| Variable | Missing % |
|---|---:|
| UNDY989 | 100.00 |
| UNDY987 | 100.00 |
| UNDY988 | 100.00 |
| ERNCM10 | 100.00 |
| ERNCM09 | 100.00 |
| ERNCM08 | 100.00 |
| DISBEN8 | 100.00 |
| ERNCM07 | 100.00 |
| TPBN1309 | 100.00 |
| TPBN1310 | 100.00 |

## Cross-File Compatibility
- Shared columns after standardization: 450
- Columns only in 2019-2021: 77
- Columns only in 2022-2024: 24

## Combined Dataset Snapshot
- Combined rows: 750,936
- Combined columns: 551

## Stage 1 Interpretation Notes
- Several variables are fully missing (100%) and should likely be excluded or treated as structurally unavailable.
- There are schema shifts between periods (new census2021 fields and renamed geography/ID fields) that require harmonization in Stage 2.
- FILEYEAR should be treated as numeric year, not a timestamp.