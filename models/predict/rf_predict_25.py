import pandas as pd
import numpy as np
import re

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# ---------- helpers ----------
def parse_sa(s: str):
    if pd.isna(s):
        return pd.NaT
    s = re.sub(r'(\b\d{1,2})(st|nd|rd|th)\b', r'\1', str(s))  # 20th -> 20
    s = s.replace("SAST", "").strip()                         # drop SAST
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

# ---------- paths ----------
IN_PATH  = "data/processed/matches_with_weather25.csv"
OUT_PATH = "data/processed/matches_with_weather25_with_preds.csv"

# ---------- load ----------
df = pd.read_csv(IN_PATH, converters={"Date_time": parse_sa})
df["Date_time"] = pd.to_datetime(df["Date_time"], errors="coerce")

# ---------- feature/target setup ----------
X_cols = [
    "Home_team","Away_team","Venue","wx_temp_c","wx_summary",
    "time_bucket","is_in_south_africa","is_main_home_stadium"
]

t_ht_h, t_ht_a = "Halftime_score_home","Halftime_score_away"
t_ft_h, t_ft_a = "Fulltime_score_home","Fulltime_score_away"

# Prediction columns (order required)
p_ft_h = "Fulltime_score_home_predicted"
p_ft_a = "Fulltime_score_away_predicted"
p_ht_h = "Halftime_score_home_predicted"
p_ht_a = "Halftime_score_away_predicted"

# Ensure any missing feature columns exist (safe defaults)
for c in ["wx_summary","time_bucket"]:
    if c not in df.columns:
        df[c] = ""
for c in ["is_in_south_africa","is_main_home_stadium"]:
    if c not in df.columns:
        df[c] = 0
if "wx_temp_c" not in df.columns:
    df["wx_temp_c"] = np.nan

X = df[X_cols].copy()

# ---------- mask: train & predict ONLY on rows with FT score present ----------
use_mask = df[t_ft_h].notna()   # True = past completed; False = future/postponed

# ---------- preprocess & models ----------
cat = ["Home_team","Away_team","Venue","wx_summary","time_bucket",
       "is_in_south_africa","is_main_home_stadium"]
num = ["wx_temp_c"]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", SimpleImputer(strategy="median"), num),
])

rf_params = dict(n_estimators=600, min_samples_leaf=2, n_jobs=-1,
                 max_depth=12, max_features="sqrt", random_state=42)

pipe_ht = Pipeline([("prep", clone(prep)), ("rf", RandomForestRegressor(**rf_params))])
pipe_ft = Pipeline([("prep", clone(prep)), ("rf", RandomForestRegressor(**rf_params))])

# ---------- fit on past completed only ----------
if use_mask.sum() == 0:
    raise SystemExit("No completed rows to train on (Fulltime_score_home empty everywhere).")

Y_ht = df.loc[use_mask, [t_ht_h, t_ht_a]].values
Y_ft = df.loc[use_mask, [t_ft_h, t_ft_a]].values

pipe_ht.fit(X.loc[use_mask], Y_ht)
pipe_ft.fit(X.loc[use_mask], Y_ft)

# ---------- predict ONLY for rows we trained on (ignore future/postponed) ----------
pred_ht = np.clip(pipe_ht.predict(X.loc[use_mask]), 0, None)  # (n_used, 2)
pred_ft = np.clip(pipe_ft.predict(X.loc[use_mask]), 0, None)  # (n_used, 2)

# Prepare/clear prediction columns
for col in [p_ft_h, p_ft_a, p_ht_h, p_ht_a]:
    df[col] = pd.Series(dtype="float")

# Write predictions into those rows; others remain blank
df.loc[use_mask, p_ft_h] = pred_ft[:, 0]
df.loc[use_mask, p_ft_a] = pred_ft[:, 1]
df.loc[use_mask, p_ht_h] = pred_ht[:, 0]
df.loc[use_mask, p_ht_a] = pred_ht[:, 1]

# (Optional) move the 4 prediction cols to the front
pred_cols = [p_ft_h, p_ft_a, p_ht_h, p_ht_a]
df = df.reindex(columns=pred_cols + [c for c in df.columns if c not in pred_cols])

# ---------- save ----------
df.to_csv(OUT_PATH, index=False)
print(f"Saved predictions -> {OUT_PATH}")