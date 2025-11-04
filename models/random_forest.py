# Random Forest with 5-fold time-aware CV for URC scores 
# - Features: Home_team, Away_team, Venue, wx_temp_c, wx_summary, time_bucket,
#             is_in_south_africa, is_main_home_stadium
# - Targets:  Fulltime_score_home, Fulltime_score_away, Halftime_score_home, Halftime_score_away

import pandas as pd
import numpy as np
import re

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# --- function to convert datetime ---
def parse_sa(s: str):
    if pd.isna(s):
        return pd.NaT
    s = re.sub(r'(\b\d{1,2})(st|nd|rd|th)\b', r'\1', s)     # 20th -> 20
    s = s.replace("SAST", "").strip()                       # drop SAST
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")  # tz-naive
    return dt

# --- Load data ---
path = r"C:\Betting\URC\data\processed\matches_with_weather_features.csv"
df = pd.read_csv(path, converters={"Date_time": parse_sa})

# Parse datetime and sort for time-aware CV
df["Date_time"] = pd.to_datetime(df["Date_time"])
df = df.sort_values("Date_time").reset_index(drop=True)

# --- Select features and targets ---
X_cols = ["Home_team","Away_team","Venue","wx_temp_c","wx_summary",
          "time_bucket","is_in_south_africa","is_main_home_stadium"]

yt = df[[
    "Halftime_score_home","Halftime_score_away",
    "Fulltime_score_home","Fulltime_score_away"
]]

X = df[X_cols]

Y_ht = yt.iloc[:, [0, 1]].values   # HT_home, HT_away
Y_ft = yt.iloc[:, [2, 3]].values   # FT_home, FT_away

# --- Preprocessing ---
cat = ["Home_team", "Away_team", "Venue", "wx_summary", "time_bucket", "is_in_south_africa", "is_main_home_stadium"]
num = ["wx_temp_c"]

# OneHotEncoder
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", "passthrough", num),
])

# --- Model ---
rf_ht = RandomForestRegressor(n_estimators=600, min_samples_leaf=2, n_jobs=-1, max_depth=12, max_features="sqrt", random_state=42)
rf_ft = RandomForestRegressor(n_estimators=600, min_samples_leaf=2, n_jobs=-1, max_depth=12, max_features="sqrt", random_state=42)

pipe_ht = Pipeline([("prep", prep), ("rf", rf_ht)])
pipe_ft = Pipeline([("prep", prep), ("rf", rf_ft)])

# --- 5-fold TimeSeriesSplit CV ---
tscv = TimeSeriesSplit(n_splits=5)

def cv_mae(pipe, X, Y):
    total_abs = np.zeros(Y.shape[1]); total_n = 0
    for tr, te in tscv.split(X):
        pipe.fit(X.iloc[tr], Y[tr])
        pred = np.clip(pipe.predict(X.iloc[te]), 0, None)
        total_abs += np.sum(np.abs(Y[te] - pred), axis=0)
        total_n += len(te)
    return total_abs / total_n   # per-target MAE

mae_ht = cv_mae(pipe_ht, X, Y_ht)  # [MAE_HT_home, MAE_HT_away]
mae_ft = cv_mae(pipe_ft, X, Y_ft)  # [MAE_FT_home, MAE_FT_away]

print(f"HT MAE home/away: {mae_ht[0]:.3f} / {mae_ht[1]:.3f}")
print(f"FT MAE home/away: {mae_ft[0]:.3f} / {mae_ft[1]:.3f}")

# Refit on full data for deployment:
pipe_ht.fit(X, Y_ht)
pipe_ft.fit(X, Y_ft)
