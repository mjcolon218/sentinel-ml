
# Model utilities for anomaly detection
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_isolation_forest(X: pd.DataFrame, seed: int = 42) -> Pipeline:
    """
    Trains a pipeline: StandardScaler -> IsolationForest.
    - StandardScaler: normalizes features to mean=0, std=1
    - IsolationForest: detects anomalies in feature space
    Returns a fitted sklearn Pipeline.
    """
    model = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("iso", IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=0.02,  # expected anomaly fraction
            random_state=seed,
            n_jobs=-1
        ))
    ])
    model.fit(X)
    return model

def anomaly_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Computes anomaly scores for each sample.
    Higher score = more anomalous (inverts sklearn's convention).
    """
    # IsolationForest: decision_function higher => more normal
    normality = model.decision_function(X)
    return (-normality)  # invert so higher = more anomalous

def save_model(model: Pipeline, path: str) -> None:
    """
    Save a trained model pipeline to disk using joblib.
    """
    joblib.dump(model, path)

def load_model(path: str) -> Pipeline:
    """
    Load a trained model pipeline from disk.
    """
    return joblib.load(path)
