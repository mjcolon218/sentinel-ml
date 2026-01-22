
# Configuration dataclass for pipeline settings
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 42  # Random seed for reproducibility
    # Paths for data and model storage
    raw_path: str = "data/raw/security_logs.jsonl"  # Path to raw security logs
    model_path: str = "models/isoforest.joblib"     # Path to save/load trained model
