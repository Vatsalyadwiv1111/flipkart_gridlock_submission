"""
backend/ml.py — Advanced ML pipelines for ParkWatch AI.

Four models, all trained on the ACTIVE dataset and cached per dataset version
(so they retrain automatically after a CSV upload / CCTV detection):

  1. predict_eta        — Random-Forest regressor: tow-truck response time (min).
  2. predict_propensity — Decision-Tree classifier: most-likely violation type.
  3. emerging_hotspots  — KMeans clustering: clusters whose density grew >20% WoW.
  4. economic_forecast  — 30-day economic-loss time-series (Prophet → seasonal fallback).

scikit-learn powers 1–3 (already a dependency). Prophet is optional for #4.
Every function degrades gracefully if a library or enough data is missing.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEPOT = (12.9774, 77.5713)  # Majestic depot
AVG_SPEED_KMPH = 27.0
MAX_PARKING_DELAY_MIN = 8.0

# Model cache keyed by dataset version so we retrain only when data changes.
_cache: dict = {"version": None, "eta": None, "prop": None}

import datetime as _dt

_live_buffer: list[dict] = []
RETRAIN_THRESHOLD = 50

_model_status: dict = {
    "eta":        {"trained": False, "engine": "none", "rows": 0, "last_trained": None},
    "propensity": {"trained": False, "engine": "none", "rows": 0, "last_trained": None},
    "hotspots":   {"trained": False, "rows": 0, "last_trained": None},
    "forecast":   {"trained": False, "engine": "none", "last_trained": None},
    "retrain_threshold": RETRAIN_THRESHOLD,
}


def get_model_status() -> dict:
    return {**_model_status, "live_buffered": len(_live_buffer)}


def append_live_detection(row: dict) -> bool:
    _live_buffer.append(row)
    return len(_live_buffer) >= RETRAIN_THRESHOLD


def flush_live_buffer(base_df: pd.DataFrame) -> pd.DataFrame:
    global _live_buffer
    if not _live_buffer:
        return base_df
    live_df = pd.DataFrame(_live_buffer)
    _live_buffer = []
    now = pd.Timestamp.now()
    for col, default in [("created_datetime", now), ("cis", 50.0),
                         ("hour", now.hour), ("is_rush_hour", 0),
                         ("is_weekend", 0), ("vehicle_size_score", 3),
                         ("violation_severity", 3), ("junction_factor", 1.0),
                         ("time_factor", 1.5)]:
        if col not in live_df.columns:
            live_df[col] = default
    if "hour" in live_df.columns and live_df["hour"].isna().any():
        live_df["created_datetime"] = pd.to_datetime(
            live_df.get("created_datetime", now), errors="coerce"
        )
        live_df["hour"] = live_df["created_datetime"].dt.hour.fillna(12).astype(int)
    return pd.concat([base_df, live_df], ignore_index=True)


def _haversine_km(a, b):
    R = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dphi = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    x = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def _is_rush(h: int) -> int:
    return 1 if (7 <= h <= 10 or 16 <= h <= 20) else 0


# ════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — Tow-truck response-time predictor (regression)
# ════════════════════════════════════════════════════════════════════════════
def _train_eta(df: pd.DataFrame):
    """Train a regressor on (distance, location CIS, rush-hour) → ETA.

    Tries XGBoost first; falls back to RandomForest if xgboost is not installed.
    Ground-truth ETA is derived from a physical model (base travel + parking
    delay) plus noise, so the model learns the response surface."""
    sample = df.dropna(subset=["latitude", "longitude", "cis"])
    if len(sample) > 8000:
        sample = sample.sample(8000, random_state=42)
    if sample.empty:
        return None

    lat = sample["latitude"].to_numpy()
    lon = sample["longitude"].to_numpy()
    cis = sample["cis"].to_numpy()
    rush = sample["is_rush_hour"].to_numpy() if "is_rush_hour" in sample.columns else np.zeros(len(sample))
    dist = np.array([_haversine_km(DEPOT, (la, lo)) * 1.4 for la, lo in zip(lat, lon)])  # road factor

    base = dist / AVG_SPEED_KMPH * 60
    delay = cis / 100 * MAX_PARKING_DELAY_MIN
    rush_pen = rush * 2.5
    rng = np.random.default_rng(42)
    y = base + delay + rush_pen + rng.normal(0, 1.0, len(sample))

    X = np.column_stack([dist, cis, rush])
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             n_jobs=1, random_state=42, tree_method="hist")
        engine = "xgboost"
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=120, max_depth=10,
                                      n_jobs=1, random_state=42)
        engine = "randomforest"
    except Exception:  # noqa: BLE001
        return None
    model.fit(X, np.maximum(1.0, y))
    _model_status["eta"] = {"trained": True, "engine": engine,
                            "rows": len(sample), "last_trained": _dt.datetime.now().isoformat()}
    return model


def predict_eta(df, zone_stats, version, lat, lon, cis_density=None, hour=None, origin=None):
    """Predict tow-truck ETA (minutes) from `origin` (depot) to (lat, lon)."""
    if _cache["version"] != version:
        _cache.update(version=version, eta=_train_eta(df), prop=None)
    model = _cache["eta"]
    depot = tuple(origin) if origin else DEPOT
    dist = _haversine_km(depot, (lat, lon)) * 1.4
    if cis_density is None:
        cis_density = min(zone_stats, key=lambda z: _haversine_km((lat, lon), (z["lat"], z["lon"])))["avgCis"] if zone_stats else 50.0
    h = pd.Timestamp.now().hour if hour is None else int(hour)
    rush = _is_rush(h)
    if model is not None:
        eta = float(model.predict([[dist, cis_density, rush]])[0])
    else:  # physical fallback
        eta = dist / AVG_SPEED_KMPH * 60 + cis_density / 100 * MAX_PARKING_DELAY_MIN + rush * 2.5
    return {
        "eta_min": round(eta, 1),
        "distance_km": round(dist, 1),
        "cis_density": round(float(cis_density), 1),
        "rush_hour": bool(rush),
        "origin": [round(depot[0], 5), round(depot[1], 5)],
        "destination": [round(lat, 5), round(lon, 5)],
    }


def train_all(df, version) -> dict:
    """Eagerly (re)train all models on the active dataset — called on CSV upload."""
    _cache.update(version=version, eta=_train_eta(df), prop=_train_propensity(df))
    em = emerging_hotspots(df)
    _model_status["hotspots"] = {"trained": True, "rows": len(df), "last_trained": _dt.datetime.now().isoformat()}
    ec = economic_forecast(df, days=30)
    _model_status["forecast"] = {"trained": True, "engine": ec.get("engine", "none"), "last_trained": _dt.datetime.now().isoformat()}
    return {
        "etaTrained": _cache["eta"] is not None,
        "propensityTrained": _cache["prop"] is not None,
        "emergingZones": len(em),
        "economicEngine": ec["engine"],
        "etaEngine": _model_status["eta"]["engine"],
        "propensityEngine": _model_status["propensity"]["engine"],
    }


# ════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — Violation-type propensity (multiclass classification)
# ════════════════════════════════════════════════════════════════════════════
def _train_propensity(df: pd.DataFrame):
    need = {"vehicle_size_score", "hour", "police_station", "violation_type"}
    if not need.issubset(df.columns):
        return None
    sample = df.dropna(subset=["violation_type"])
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=42)
    if sample["violation_type"].nunique() < 2:
        return None

    stations = sorted(sample["police_station"].dropna().astype(str).unique())
    scode = {s: i for i, s in enumerate(stations)}
    X = np.column_stack([
        sample["vehicle_size_score"].to_numpy(),
        sample["hour"].to_numpy(),
        sample["police_station"].astype(str).map(scode).fillna(0).to_numpy(),
    ])
    y = sample["violation_type"].astype(str).to_numpy()
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=200, max_depth=8,
                                 min_child_samples=25, random_state=42, verbose=-1, n_jobs=1)
        engine = "lightgbm"
    except ImportError:
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=25, random_state=42)
        engine = "decisiontree"
    except Exception:  # noqa: BLE001
        return None
    clf.fit(X, y)
    _model_status["propensity"] = {"trained": True, "engine": engine,
                                   "rows": len(sample), "last_trained": _dt.datetime.now().isoformat()}
    # typical vehicle size per station (feature value at inference)
    veh = sample.groupby("police_station", observed=True)["vehicle_size_score"].mean().to_dict()
    return {"clf": clf, "scode": scode, "veh": veh, "veh_mean": float(sample["vehicle_size_score"].mean())}


def predict_propensity(df, version, station, hour):
    if _cache["version"] != version or _cache.get("prop") is None:
        _cache.update(version=version, prop=_train_propensity(df))
        if _cache.get("eta") is None:
            _cache["eta"] = _train_eta(df)
    m = _cache["prop"]
    if not m:
        return {"violation_type": "No Parking", "confidence": 0.0, "available": False}
    code = m["scode"].get(str(station), 0)
    veh = m["veh"].get(str(station), m["veh_mean"])
    proba = m["clf"].predict_proba([[veh, int(hour), code]])[0]
    classes = m["clf"].classes_
    idx = int(np.argmax(proba))
    top = [
        {"type": str(classes[i]).title(), "confidence": round(float(proba[i]) * 100, 1)}
        for i in np.argsort(proba)[::-1][:3]
    ]
    return {"violation_type": str(classes[idx]).title(), "confidence": round(float(proba[idx]) * 100, 1), "top": top, "available": True}


# ════════════════════════════════════════════════════════════════════════════
#  MODEL 3 — Emerging-hotspot predictor (spatiotemporal clustering)
# ════════════════════════════════════════════════════════════════════════════
def emerging_hotspots(df: pd.DataFrame):
    """DBSCAN spatial clusters whose violation density grew >20% week-over-week.

    Returns dicts with both API-spec fields (latitude, longitude, growth_rate)
    and the short aliases the map component uses (lat, lon, growth)."""
    d = df.dropna(subset=["latitude", "longitude", "created_datetime"])
    if len(d) < 30:
        return []
    if len(d) > 20000:
        d = d.sample(20000, random_state=42)

    def _emit(lat, lon, growth_pct, recent, prior):
        return {
            "latitude": round(float(lat), 5),
            "longitude": round(float(lon), 5),
            "growth_rate": round(growth_pct),
            # aliases for the frontend map
            "lat": float(lat),
            "lon": float(lon),
            "growth": round(growth_pct),
            "recent": int(recent),
            "prior": int(prior),
        }

    try:
        from sklearn.cluster import DBSCAN
    except Exception:  # noqa: BLE001
        return []

    coords = d[["latitude", "longitude"]].to_numpy()
    labels = DBSCAN(eps=0.012, min_samples=8).fit_predict(coords)  # ~1.3km radius
    d = d.assign(_cluster=labels)
    clusters = [c for c in set(labels) if c != -1]
    if not clusters:  # fall back to KMeans if DBSCAN finds no dense cores
        try:
            from sklearn.cluster import KMeans

            k = max(3, min(8, len(d) // 200))
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(coords)
            d = d.assign(_cluster=labels)
            clusters = list(range(k))
        except Exception:  # noqa: BLE001
            return []

    max_date = d["created_datetime"].max()
    last = d[d["created_datetime"] > max_date - pd.Timedelta(days=7)]
    prev = d[(d["created_datetime"] > max_date - pd.Timedelta(days=14)) & (d["created_datetime"] <= max_date - pd.Timedelta(days=7))]

    out = []
    for c in clusters:
        members = d[d["_cluster"] == c]
        lat, lon = members["latitude"].mean(), members["longitude"].mean()
        lc = int((last["_cluster"] == c).sum())
        pc = int((prev["_cluster"] == c).sum())
        growth = (lc - pc) / pc if pc > 0 else (1.0 if lc > 0 else 0.0)
        if growth > 0.2 and lc >= 3:
            out.append(_emit(lat, lon, growth * 100, lc, pc))

    if not out:  # surface densest recent clusters as watch zones
        order = sorted(clusters, key=lambda c: -int((last["_cluster"] == c).sum()) if len(last) else -len(d[d["_cluster"] == c]))
        for c in order[:2]:
            members = d[d["_cluster"] == c]
            out.append(_emit(members["latitude"].mean(), members["longitude"].mean(), 0, len(members), 0))
    return sorted(out, key=lambda z: -z["growth_rate"])[:5]


# ════════════════════════════════════════════════════════════════════════════
#  MODEL 4 — Economic-loss forecast (time series)
# ════════════════════════════════════════════════════════════════════════════
def economic_forecast(df: pd.DataFrame, days: int = 30):
    """Forecast daily economic loss (₹) for the next `days` (Prophet → seasonal)."""
    d = df.dropna(subset=["created_datetime", "cis"]).copy()
    if d.empty:
        return {"engine": "none", "history": [], "points": []}
    d["date"] = d["created_datetime"].dt.normalize()
    # daily cost ≈ violations × 0.5 person-hr × ₹320, CIS-weighted
    daily = d.groupby("date").apply(lambda g: len(g) * 0.5 * 320 * (1 + g["cis"].mean() / 100), include_groups=False).reset_index(name="loss")
    daily = daily.sort_values("date")
    history = [{"date": r["date"].strftime("%Y-%m-%d"), "loss": round(float(r["loss"])), "type": "history"} for _, r in daily.tail(30).iterrows()]

    last_date = daily["date"].max()
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(days)]

    # Prophet removed to save 150MB+ RAM in Docker containers.

    # Seasonal fallback: linear trend + weekday offset.
    y = daily["loss"].to_numpy(dtype=float)
    x = np.arange(len(y))
    if len(y) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, float(y[0]) if len(y) else 0.0
    resid = y - (slope * x + intercept)
    wd_offset = {}
    for wd, grp in daily.assign(wd=daily["date"].dt.dayofweek, _r=resid).groupby("wd"):
        wd_offset[int(wd)] = float(grp["_r"].mean())
    pts = []
    for i, ts in enumerate(future_dates):
        base = slope * (len(y) + i) + intercept + wd_offset.get(ts.dayofweek, 0.0)
        pts.append({"date": ts.strftime("%Y-%m-%d"), "loss": round(max(0, base)), "type": "forecast"})
    return {"engine": "seasonal", "history": history, "points": pts}
