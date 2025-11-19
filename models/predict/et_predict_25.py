import pandas as pd
import numpy as np
import re

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# helpers 
def parse_sa(s: str):
    if pd.isna(s):
        return pd.NaT
    s = re.sub(r'(\b\d{1,2})(st|nd|rd|th)\b', r'\1', str(s))  # 20th -> 20
    s = s.replace("SAST", "").strip()                         # drop SAST
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

# paths 
IN_PATH  = "data/processed/matches_with_weather_features25.csv"
OUT_PATH = "models/predict/25preds_et.csv"

# load 
df = pd.read_csv(IN_PATH, converters={"Date_time": parse_sa})
df["Date_time"] = pd.to_datetime(df["Date_time"], errors="coerce")
original_cols = list(df.columns)  # keep exact original order

# names 
X_cols = [
    "Home_team","Away_team","Venue","wx_temp_c","wx_summary",
    "time_bucket","is_in_south_africa","is_main_home_stadium"
]

t_ht_h, t_ht_a = "Halftime_score_home","Halftime_score_away"
t_ft_h, t_ft_a = "Fulltime_score_home","Fulltime_score_away"

# Prediction column names (order at the very front)
p_ft_h = "Fulltime_score_home_predicted"
p_ft_a = "Fulltime_score_away_predicted"
p_ht_h = "Halftime_score_home_predicted"
p_ht_a = "Halftime_score_away_predicted"

# build X WITHOUT modifying df 
def get(col, default):
    return df[col] if col in df.columns else pd.Series(default, index=df.index)

X = pd.DataFrame({
    "Home_team": get("Home_team", ""),
    "Away_team": get("Away_team", ""),
    "Venue": get("Venue", ""),
    "wx_temp_c": get("wx_temp_c", np.nan),
    "wx_summary": get("wx_summary", ""),
    "time_bucket": get("time_bucket", ""),
    "is_in_south_africa": get("is_in_south_africa", 0),
    "is_main_home_stadium": get("is_main_home_stadium", 0),
})

# Train & predict ONLY on rows with FT present (past completed)
use_mask = df[t_ft_h].notna()

# preprocess & models 
cat = ["Home_team","Away_team","Venue","wx_summary","time_bucket",
       "is_in_south_africa","is_main_home_stadium"]
num = ["wx_temp_c"]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", SimpleImputer(strategy="median"), num),
])

ef_ht_params = dict(n_estimators=600, criterion="absolute_error", min_samples_leaf=4, n_jobs=-1,
                 max_depth=12, max_features="sqrt", random_state=42)

ef_ft_params = dict(n_estimators=1000, criterion="squared_error", min_samples_leaf=2, n_jobs=-1,
                 max_depth=12, max_features="sqrt", random_state=42)

pipe_ht = Pipeline([("prep", clone(prep)), ("et", ExtraTreesRegressor(**ef_ht_params))])
pipe_ft = Pipeline([("prep", clone(prep)), ("et", ExtraTreesRegressor(**ef_ft_params))])

if use_mask.sum() == 0:
    raise SystemExit("No completed rows to train on.")

Y_ht = df.loc[use_mask, [t_ht_h, t_ht_a]].values
Y_ft = df.loc[use_mask, [t_ft_h, t_ft_a]].values

pipe_ht.fit(X.loc[use_mask], Y_ht)
pipe_ft.fit(X.loc[use_mask], Y_ft)

# Predict only for completed rows; others stay NaN
pred_ht = np.clip(pipe_ht.predict(X), 0, None)
pred_ft = np.clip(pipe_ft.predict(X), 0, None)

# Set preds to NaN where we shouldn't predict (future/postponed)
pred_ht[~use_mask.values, :] = np.nan
pred_ft[~use_mask.values, :] = np.nan

# build a separate preds_df (does not alter df) 
preds_df = pd.DataFrame({
    p_ft_h: pred_ft[:, 0],
    p_ft_a: pred_ft[:, 1],
    p_ht_h: pred_ht[:, 0],
    p_ht_a: pred_ht[:, 1],
}, index=df.index)

# prepend preds to the original df, preserving everything else 
out_df = pd.concat([preds_df, df[original_cols]], axis=1)

out_df.to_csv(OUT_PATH, index=False)
print(f"Saved predictions -> {OUT_PATH}")