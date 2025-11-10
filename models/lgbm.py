import pandas as pd
import numpy as np
import re

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor
from pathlib import Path

# --- function to convert datetime ---
def parse_sa(s: str):
    if pd.isna(s):
        return pd.NaT
    s = re.sub(r'(\b\d{1,2})(st|nd|rd|th)\b', r'\1', s)     # 20th -> 20
    s = s.replace("SAST", "").strip()                       # drop SAST
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")  # tz-naive
    return dt

# --- Load data ---
REPO_ROOT = Path.cwd()
DATA = REPO_ROOT / "data" / "processed" / "matches_with_weather_features24.csv"
df = pd.read_csv(DATA)

# Parse datetime and sort for time-aware CV
df["Date_time"] = df["Date_time"].apply(parse_sa)
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
cat = ["Home_team", "Away_team", "Venue", "wx_summary", "time_bucket",
       "is_in_south_africa", "is_main_home_stadium"]
num = ["wx_temp_c"]

# OneHotEncoder (LightGBM handles scipy sparse matrices fine)
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", "passthrough", num),
])

# --- LightGBM Models (wrapped for multi-output) ---
# These are sensible starting hyperparams; tune as needed.
lgbm_base_ht = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,           # let num_leaves control complexity
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.3,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm_base_ft = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=95,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm_ht = MultiOutputRegressor(lgbm_base_ht, n_jobs=None)
lgbm_ft = MultiOutputRegressor(lgbm_base_ft, n_jobs=None)

pipe_ht = Pipeline([("prep", prep), ("lgbm", lgbm_ht)])
pipe_ft = Pipeline([("prep", prep), ("lgbm", lgbm_ft)])

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
