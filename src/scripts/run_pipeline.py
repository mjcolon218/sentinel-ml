
# Import necessary libraries
import pandas as pd
from sentinelml.log_generator import generate_logs  # For generating synthetic security logs
from sentinelml.features import build_feature_table  # For feature engineering
from sentinelml.model import train_isolation_forest, anomaly_scores, save_model  # Model utilities

# Path to store the generated raw logs
RAW_PATH = "data/raw/security_logs.jsonl"

if __name__ == "__main__":
    # Step 1: Generate synthetic security logs and save to file
    generate_logs(out_path=RAW_PATH, n_events=8000, anomaly_rate=0.02, seed=42)

    # Step 2: Load the generated logs into a DataFrame
    df = pd.read_json(RAW_PATH, lines=True)

    # Step 3: Build feature table (X: features, meta: metadata)
    X, meta = build_feature_table(df)

    # Step 4: Train Isolation Forest model for anomaly detection
    model = train_isolation_forest(X, seed=42)

    # Step 5: Save the trained model to disk
    save_model(model, "models/isoforest.joblib")

    # Step 6: Compute anomaly scores for each window
    scores = anomaly_scores(model, X)
    meta = meta.copy()  # Avoid modifying original metadata
    meta["score"] = scores  # Add scores to metadata
    import numpy as np

# Step 7: Set alert threshold at the 99th percentile (top 1% most anomalous)
p = 99  # alert on top 1% most anomalous windows
threshold = float(np.percentile(meta["score"], p))

# Step 8: Flag windows as alerts if their score exceeds the threshold
meta["is_alert"] = meta["score"] >= threshold
alert_rate = meta["is_alert"].mean()  # Fraction of windows flagged as alerts

# Step 9: Print alert threshold and alert rate
print(f"\nAlert threshold (p{p}): {threshold:.6f}")
print(f"Alert rate: {alert_rate:.2%} of windows flagged")

# Step 10: Evaluate alert quality
alerts = meta[meta["is_alert"]].copy()
if len(alerts) > 0:
    # Precision: fraction of alerts that are true injected anomalies
    precision = float((alerts["injected_anoms"] == True).mean())
    print(f"Precision among alerts: {precision:.2%} (injected anomalies)")

    # Show top 10 most anomalous alerts
    print("\nTop 10 alerts:")
    print(alerts.sort_values("score", ascending=False)
              .head(10)[["user_id","window","score","injected_anoms","anomaly_type"]]
              .to_string(index=False))

    # Show top 15 most anomalous windows overall
    top = meta.sort_values("score", ascending=False).head(15)
    print("\nTop alerts (most anomalous windows):")
    print(top[["user_id","window","score","injected_anoms","anomaly_type"]].to_string(index=False))

    # Sanity check: hit-rate of injected anomalies in top 15
    hit_rate = float((top["injected_anoms"] == True).mean())
    print(f"\nInjected anomaly hit-rate in Top-15: {hit_rate:.2%}")
