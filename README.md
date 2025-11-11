URC score prediction

Features

Halftime & fulltime home/away score prediction (non-negative, clipped).
Time-aware cross-validation (TimeSeriesSplit).
One-Hot Encoding for RF/ET; native categoricals tried for LGBM.
Parameters tuned with GridSearchCV.

Data

Source: data/processed/matches_with_weather_features24.csv
Key features: teams, venue, weather (wx_*).

Current models

Halftime (ETR): criterion=absolute_error, max_depth=12, min_samples_leaf=4, max_features='sqrt', n_estimators=600
Fulltime (ETR): criterion=squared_error, max_depth=12, min_samples_leaf=2, max_features='sqrt', n_estimators=1000
Metric: mean absolute error (MAE) averaged over home/away targets with TimeSeriesSplit.