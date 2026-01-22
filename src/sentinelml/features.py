
# Feature engineering utilities for security event logs
import pandas as pd
import numpy as np
from scipy import stats
# Convert a pandas Series to datetime, coercing errors and using UTC
def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates events into per-user 5-minute windows (high-signal for cybersecurity).
    Returns:
      X: feature table (numeric)
      meta: keys + labels for debugging
    """
    d = df.copy()
    # Convert timestamps to datetime and drop rows with invalid timestamps
    d["timestamp"] = _to_dt(d["timestamp"])
    d = d.dropna(subset=["timestamp"])
    # Create a 'window' column for 5-minute time bins
    d["window"] = d["timestamp"].dt.floor("5min")

    # Group by user and window for aggregation
    grp = d.groupby(["user_id", "window"], as_index=False)

    # Aggregate features for each user-window
    feat = grp.agg(
        event_count=("event_type", "size"),  # total events
        login_count=("event_type", lambda x: int((x == "login").sum())),  # login events
        fail_login_count=("auth_result", lambda x: int((x == "fail").sum())),  # failed logins
        unique_resources=("resource", "nunique"),  # unique resources accessed
        unique_event_types=("event_type", "nunique"),  # unique event types
        total_bytes=("bytes_transferred", "sum"),  # total bytes transferred
        max_bytes=("bytes_transferred", "max"),  # max bytes in a single event
        mean_bytes=("bytes_transferred", "mean"),  # mean bytes per event
        std_bytes=("bytes_transferred", "std"),  # std dev of bytes
        mean_lat=("lat", "mean"),  # mean latitude
        mean_lon=("lon", "mean"),  # mean longitude
        unique_countries=("country", "nunique"),  # unique countries
    )

    # Fill NaNs in std_bytes (e.g., if only one event in window)
    feat["std_bytes"] = feat["std_bytes"].fillna(0.0)

    # Derived rate features
    feat["fail_rate"] = np.where(feat["login_count"] > 0, feat["fail_login_count"] / feat["login_count"], 0.0)
    feat["bytes_per_event"] = np.where(feat["event_count"] > 0, feat["total_bytes"] / feat["event_count"], 0.0)
        # ---- Poisson deviation feature (login burst "surprise") ----
    # Estimate per-user baseline lambda (expected logins per 5-min window)
    # Using mean login_count across windows for that user.
    user_lambda = feat.groupby("user_id")["login_count"].transform("mean")

    # Avoid lambda=0 causing degenerate Poisson
    user_lambda = user_lambda.clip(lower=1e-6)

    # Tail probability P(K >= k | lambda) = sf(k-1)
    # Surprise score = -log(tail_prob) (bigger = more surprising)
    tail_prob = stats.poisson.sf(feat["login_count"] - 1, mu=user_lambda)

    # Numerical stability
    tail_prob = np.clip(tail_prob, 1e-300, 1.0)
    feat["login_poisson_surprise"] = -np.log(tail_prob)

    # Labels for evaluation (not used by model)
    lbl = grp.agg(
        injected_anoms=("is_injected_anomaly", "max"),  # was an anomaly injected?
        anomaly_type=("anomaly_type", lambda x: x.dropna().iloc[0] if x.dropna().shape[0] else None),  # anomaly type
    )
    # Merge features and labels for metadata
    meta = feat[["user_id", "window"]].merge(lbl, on=["user_id", "window"], how="left")

    # Drop non-numeric columns for model input
    X = feat.drop(columns=["user_id", "window"])
    # Replace any remaining NaNs or infs with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, meta

def build_event_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight per-event features for quick streaming demo.
    (Not used in main pipeline.)
    """
    d = events_df.copy()
    d["is_login"] = (d["event_type"] == "login").astype(int)
    d["is_fail"] = (d["auth_result"] == "fail").astype(int)
    d["is_export"] = (d["resource"] == "/export").astype(int)
    X = d[["is_login", "is_fail", "is_export", "bytes_transferred"]].copy()
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
