
# Synthetic log generator for security events
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import numpy as np

# Event types and resources for synthetic logs
EVENT_TYPES = ["login", "file_access", "api_call", "process_start", "dns_query"]
RESOURCES = ["/admin", "/billing", "/export", "/settings", "/api/v1/data", "/api/v1/auth"]
AUTH_RESULTS = ["success", "fail"]

# User profile for log generation
@dataclass
class UserProfile:
    user_id: str
    base_rate_per_min: float  # Poisson rate for events
    home_country: str
    home_lat: float
    home_lon: float

# Generate a random IPv4 address
def _rand_ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


# Compute haversine distance (km) between two lat/lon points
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    # lightweight haversine
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return float(2*R*np.arcsin(np.sqrt(a)))


# Create a list of synthetic users with home locations and event rates
def _make_users(rng: random.Random, n_users: int) -> List[UserProfile]:
    # small set of "home" locations
    homes = [
        ("US", 40.7128, -74.0060),  # NYC
        ("US", 39.2904, -76.6122),  # Baltimore
        ("GB", 51.5072, -0.1276),   # London
        ("DE", 52.5200, 13.4050),   # Berlin
        ("IN", 28.6139, 77.2090),   # Delhi
    ]
    users: List[UserProfile] = []
    for i in range(n_users):
        c, lat, lon = homes[rng.randrange(len(homes))]
        # base event rate ~ Poisson (events/min)
        base = rng.uniform(0.05, 0.8)  # 0.05 to 0.8 events/min
        users.append(UserProfile(user_id=f"user_{i:03d}", base_rate_per_min=base,
                                home_country=c, home_lat=lat, home_lon=lon))
    return users

def generate_logs(out_path: str, n_events: int = 5000, anomaly_rate: float = 0.02, seed: int = 42) -> None:
    """
    Writes JSONL security telemetry with injected anomalies:
    - impossible travel (geo jump)
    - brute force (login fail bursts)
    - data exfil (huge bytes on /export)
    Args:
        out_path: Output file path for JSONL logs
        n_events: Number of events to generate
        anomaly_rate: Fraction of events to inject anomalies
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    # Create synthetic users
    users = _make_users(rng, n_users=80)

    # Start time for logs (6 hours ago)
    start = datetime.now(timezone.utc) - timedelta(hours=6)
    current = start

    # Write events to output file
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n_events):
            user = users[rng.randrange(len(users))]

            # Poisson inter-arrival: expected events/min => expected seconds between events
            lam = max(user.base_rate_per_min, 1e-3)
            delta_seconds = np_rng.exponential(scale=60.0/lam)
            current += timedelta(seconds=float(delta_seconds))

            # Choose event type, resource, and IP
            event_type = rng.choice(EVENT_TYPES)
            resource = rng.choice(RESOURCES)
            ip = _rand_ip(rng)

            # Baseline geo near home
            lat = user.home_lat + rng.uniform(-0.25, 0.25)
            lon = user.home_lon + rng.uniform(-0.25, 0.25)
            country = user.home_country

            # Auth result: mostly success except for logins
            auth_result = "success" if (event_type != "login") else rng.choices(AUTH_RESULTS, weights=[0.85, 0.15])[0]
            bytes_tx = int(max(0, np_rng.normal(loc=2000, scale=1200)))

            # Decide if this event is an injected anomaly
            is_anom = rng.random() < anomaly_rate
            anom_type = None

            # Inject anomalies
            if is_anom:
                pick = rng.random()
                if pick < 0.34:
                    # Impossible travel: jump far away
                    far = [("JP", 35.6762, 139.6503), ("BR", -23.5505, -46.6333), ("AU", -33.8688, 151.2093)]
                    country, lat, lon = far[rng.randrange(len(far))]
                    event_type = "login"
                    auth_result = "success"
                    anom_type = "impossible_travel"
                elif pick < 0.67:
                    # Brute force burst: many failed logins, short inter-arrival
                    event_type = "login"
                    auth_result = "fail"
                    bytes_tx = int(max(0, np_rng.normal(loc=500, scale=200)))
                    anom_type = "bruteforce"
                    # Compress time to simulate burst
                    current -= timedelta(seconds=rng.uniform(0, 10))
                else:
                    # Data exfiltration: large bytes on export
                    event_type = rng.choice(["api_call", "file_access"])
                    resource = "/export"
                    bytes_tx = int(np_rng.integers(200_000, 3_000_000))
                    anom_type = "data_exfil"

            # Build event record
            rec: Dict[str, object] = {
                "timestamp": current.isoformat(),
                "user_id": user.user_id,
                "event_type": event_type,
                "resource": resource,
                "ip_address": ip,
                "country": country,
                "lat": lat,
                "lon": lon,
                "auth_result": auth_result,
                "bytes_transferred": bytes_tx,
                "is_injected_anomaly": bool(is_anom),
                "anomaly_type": anom_type,
            }
            # Write event as JSON line
            f.write(json.dumps(rec) + "\n")
