# Data Card — College Experience Dataset

## Source
Longitudinal study, 5 years, two student generations (2017, 2018).

## Data Groups
| Group | Files | Description |
|---|---|---|
| EMA | general_ema.csv, covid_ema | Weekly self-assessment surveys |
| Sensing | sensing.csv | Mobile sensor data — activity, mobility, sleep, phone usage, audio |
| Demographics | TBD | Student demographic information |

## Known Limitations
- Irregular surveys (not exactly once per week)
- Android/iOS availability differs per column
- COVID period may introduce distributional shift
- Student dropout expected, 123 out of 217 finished the whole 4 years

## Target Variable (planned)
Composite mental health score (0–100%) from PHQ4, stress, social level, SSE.