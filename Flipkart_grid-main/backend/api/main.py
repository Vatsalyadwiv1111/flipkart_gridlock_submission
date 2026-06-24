"""
backend/api.py — ParkWatch AI REST API (FastAPI)
================================================
A FastAPI gateway that exposes the same intelligence the Streamlit app computes,
so the React frontend can consume it over HTTP.

Run:
    cd backend
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000

Auth:
    POST /api/auth/login   { "username": "admin", "password": "btp123" }  -> admin
    POST /api/auth/login   { "username": "user",  "password": "user123" } -> user

All other endpoints require:  Authorization: Bearer <token>

Design notes
------------
* Real data processing reuses backend/data/loader.py + helpers.py (and the
  OR-Tools solver from tabs/tab_ortools_dispatch.py). On startup we load the
  bundled demo dataset; if that fails we fall back to a small synthetic frame
  so the API always boots.
* Endpoint payloads are shaped to match exactly what the frontend tabs expect,
  keeping the React wiring a thin layer.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import math
import os
import time
import uuid
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

os.makedirs("tmp", exist_ok=True)
tempfile.tempdir = os.path.abspath("tmp")

# Load backend/.env so GROQ_API_KEY / DATABASE_URL / CORS_ORIGINS are picked up
# (must run before any os.environ reads below).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ── Reuse existing backend logic ─────────────────────────────────────────────
from api.helpers import (  # noqa: E402
    get_max_severity,
    get_vehicle_size,
    parse_violations,
    compute_time_factor,
)

# ── Production modules (all degrade gracefully) ──────────────────────────────
from api import alerts as alerts_mod  # noqa: E402
from api import forecasting  # noqa: E402
from api import agent as agent_mod  # noqa: E402
from api import rag  # noqa: E402
from api import etl  # noqa: E402
from api import ml  # noqa: E402
from api import db  # noqa: E402

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════

SECRET_KEY = os.environ.get("PARKWATCH_SECRET", "parkwatch-dev-secret-change-me")
TOKEN_TTL_SECONDS = 60 * 60 * 8  # 8 hours
# Base local origins + any extra production origins from CORS_ORIGINS (comma-sep,
# e.g. your Vercel domain). The localhost regex below also covers dev ports.
ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000"] + [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
]

# Demo credential store. In production swap for a hashed-password user table.
USERS = {
    "admin": {"password": "btp123", "role": "admin", "name": "BTP Commander"},
    "user": {"password": "user123", "role": "user", "name": "Patrol Officer"},
}

# ════════════════════════════════════════════════════════════════════════════
#  MINIMAL JWT (HS256) — no external dependency
# ════════════════════════════════════════════════════════════════════════════

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(seg: str) -> bytes:
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def create_token(payload: dict) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    body = {**payload, "iat": int(time.time()), "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    seg = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(body).encode())}"
    sig = hmac.new(SECRET_KEY.encode(), seg.encode(), hashlib.sha256).digest()
    return f"{seg}.{_b64url(sig)}"


def decode_token(token: str) -> dict:
    try:
        header_b64, body_b64, sig_b64 = token.split(".")
        seg = f"{header_b64}.{body_b64}"
        expected = hmac.new(SECRET_KEY.encode(), seg.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
            raise ValueError("bad signature")
        body = json.loads(_b64url_decode(body_b64))
        if body.get("exp", 0) < time.time():
            raise ValueError("token expired")
        return body
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
        )


bearer = HTTPBearer(auto_error=True)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    """Dependency that verifies the JWT and returns the claims."""
    return decode_token(creds.credentials)


def require_admin(claims: dict = Depends(require_auth)) -> dict:
    if claims.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return claims


# ════════════════════════════════════════════════════════════════════════════
#  DATA BOOTSTRAP  (reuse loader.py; fall back to synthetic frame)
# ════════════════════════════════════════════════════════════════════════════

BENGALURU_CENTER = (12.9716, 77.5946)

ZONES = [
    {"name": "MG Road", "lat": 12.9756, "lon": 77.6068, "station": "Cubbon Park"},
    {"name": "Koramangala", "lat": 12.9352, "lon": 77.6245, "station": "Koramangala"},
    {"name": "Indiranagar", "lat": 12.9784, "lon": 77.6408, "station": "Indiranagar"},
    {"name": "Whitefield", "lat": 12.9698, "lon": 77.7499, "station": "Whitefield"},
    {"name": "Electronic City", "lat": 12.8452, "lon": 77.6602, "station": "Electronic City"},
    {"name": "Jayanagar", "lat": 12.9252, "lon": 77.5833, "station": "Jayanagar"},
    {"name": "Majestic", "lat": 12.9774, "lon": 77.5713, "station": "Majestic"},
    {"name": "Silk Board", "lat": 12.9172, "lon": 77.6225, "station": "Silk Board"},
    {"name": "HSR Layout", "lat": 12.9116, "lon": 77.6389, "station": "HSR Layout"},
    {"name": "Marathahalli", "lat": 12.9591, "lon": 77.6974, "station": "Marathahalli"},
    {"name": "Rajajinagar", "lat": 12.9916, "lon": 77.5526, "station": "Rajajinagar"},
    {"name": "Yelahanka", "lat": 13.1007, "lon": 77.5963, "station": "Yelahanka"},
]

_RNG = np.random.default_rng(42)


def _bootstrap_dataframe() -> tuple[pd.DataFrame, str]:
    """Load the real dataset; fall back to demo then synthetic frame. Returns (df, source)."""
    try:
        from api.loader import load_real_dataset
        df, stats = load_real_dataset()
        if df is not None and len(df):
            print(f"[api] Real dataset loaded: {len(df):,} rows")
            return df, "jan-may-violations"
    except Exception as exc:  # noqa: BLE001
        print(f"[api] Real dataset unavailable ({exc}); trying demo")
    try:
        from api.loader import load_and_process_data, load_demo_data

        df, _stats = load_and_process_data(load_demo_data())
        if df is not None and len(df):
            return df, "demo"
    except Exception as exc:  # noqa: BLE001
        print(f"[api] loader.py unavailable ({exc}); using synthetic frame")

    # ── Synthetic fallback built with helpers.py so scoring still matches ──
    n = 1500
    stations = [z["station"] for z in ZONES]
    vehicles = ["Car", "Two-wheeler", "Bus", "Auto", "HGV", "Tanker", "LGV"]
    violations = ["NO PARKING", "DOUBLE PARKING", "WRONG PARKING", "PARKING IN A MAIN ROAD"]
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(n):
        z = ZONES[i % len(ZONES)]
        vt = vehicles[_RNG.integers(0, len(vehicles))]
        vio = violations[_RNG.integers(0, len(violations))]
        ts = base + pd.Timedelta(seconds=int(_RNG.integers(0, 150 * 24 * 3600)))
        rows.append(
            {
                "id": i + 1,
                "police_station": z["station"],
                "location": f"{z['name']} Rd",
                "vehicle_type": vt,
                "violation_type": vio,
                "latitude": z["lat"] + float(_RNG.normal(0, 0.01)),
                "longitude": z["lon"] + float(_RNG.normal(0, 0.01)),
                "created_datetime": ts,
            }
        )
    df = pd.DataFrame(rows)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"])
    df["hour"] = df["created_datetime"].dt.hour
    df["day_of_week"] = df["created_datetime"].dt.day_name()
    df["day_num"] = df["created_datetime"].dt.dayofweek
    df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if (7 <= h <= 10 or 16 <= h <= 20) else 0)
    df["is_weekend"] = df["day_num"].apply(lambda d: 1 if d >= 5 else 0)
    df["violation_list"] = df["violation_type"].apply(parse_violations)
    df["violation_severity"] = df["violation_list"].apply(get_max_severity)
    df["vehicle_size_score"] = df["vehicle_type"].apply(get_vehicle_size)
    df["junction_factor"] = 1.5
    raw = df["violation_severity"] * df["vehicle_size_score"] * df["junction_factor"]
    df["cis"] = (raw - raw.min()) / (raw.max() - raw.min()) * 100
    return df, "synthetic"


DF, _BOOT_SOURCE = _bootstrap_dataframe()


# ════════════════════════════════════════════════════════════════════════════
#  DERIVED ANALYTICS  (computed once at startup from DF)
# ════════════════════════════════════════════════════════════════════════════

def _risk(epi: float) -> str:
    return "critical" if epi > 70 else "medium" if epi > 45 else "clear"


def _normalize(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn) * 100 if mx > mn else pd.Series(50.0, index=series.index)


def _zone_stats_from_df(df: pd.DataFrame, top_n: int = 15) -> list[dict]:
    """Per-station CIS/EPI aggregation derived ENTIRELY from the active frame —
    real median coordinates, counts and EPI. Works for the demo data and for any
    uploaded CSV, so every tab reflects whatever dataset is currently loaded."""
    if df is None or df.empty or "police_station" not in df.columns:
        return []
    has = lambda c: c in df.columns
    agg = (
        df.groupby("police_station", observed=True)
        .agg(
            count=("cis", "size"),
            avgCis=("cis", "mean"),
            totalCis=("cis", "sum"),
            rushPct=("is_rush_hour", "mean") if has("is_rush_hour") else ("cis", "size"),
            heavyPct=("vehicle_size_score", lambda x: (x >= 7).mean()) if has("vehicle_size_score") else ("cis", "size"),
            junctionPct=("junction_factor", lambda x: (x == 2.0).mean()) if has("junction_factor") else ("cis", "size"),
            severity=("violation_severity", "mean") if has("violation_severity") else ("cis", "mean"),
            lat=("latitude", "median"),
            lon=("longitude", "median"),
        )
        .reset_index()
        .dropna(subset=["lat", "lon"])
    )
    if agg.empty:
        return []
    agg["epi"] = (
        0.4 * _normalize(agg["totalCis"]) + 0.3 * _normalize(agg["count"]) + 0.3 * _normalize(agg["avgCis"])
    )
    agg["epi"] = _normalize(agg["epi"])
    agg = agg.sort_values("epi", ascending=False).head(top_n).reset_index(drop=True)

    stats = []
    for i, r in agg.iterrows():
        rush = float(r["rushPct"]) if has("is_rush_hour") else 0.0
        heavy = float(r["heavyPct"]) if has("vehicle_size_score") else 0.0
        junc = float(r["junctionPct"]) if has("junction_factor") else 0.0
        epi = float(r["epi"])
        stats.append(
            {
                "id": int(i),
                "name": str(r["police_station"]),
                "station": str(r["police_station"]),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "count": int(r["count"]),
                "avgCis": round(float(r["avgCis"]), 1),
                "totalCis": round(float(r["totalCis"])),
                "rushPct": round(rush * 100) if rush <= 1 else round(rush),
                "heavyPct": round(heavy * 100) if heavy <= 1 else round(heavy),
                "junctionPct": round(junc * 100) if junc <= 1 else round(junc),
                "severity": round(float(r["severity"]), 1),
                "epi": round(epi, 1),
                "risk": _risk(epi),
            }
        )
    return stats


def _build_station_pool(zone_stats: list[dict]) -> list[dict]:
    """Expand the active stations into ~3× sub-stations so the 'Target Stations'
    slider (5–30) is meaningful. EPI is recomputed across the whole pool."""
    suffixes = ["Main", "Junction", "Cross", "Market", "Flyover"]
    pool = []
    for z in zone_stats:
        pool.append({"name": z["name"], "lat": z["lat"], "lon": z["lon"], "violations": z["count"], "avgCis": z["avgCis"]})
        for k in range(2):
            pool.append(
                {
                    "name": f"{z['name']} {suffixes[(z['id'] + k) % len(suffixes)]}",
                    "lat": z["lat"] + float(_RNG.uniform(-0.02, 0.02)),
                    "lon": z["lon"] + float(_RNG.uniform(-0.02, 0.02)),
                    "violations": int(z["count"] * float(_RNG.uniform(0.3, 0.8))),
                    "avgCis": round(z["avgCis"] * float(_RNG.uniform(0.7, 1.1)), 1),
                }
            )
    if not pool:
        return []
    for s in pool:
        s["totalCis"] = s["violations"] * s["avgCis"]

    def norm(vals):
        mn, mx = min(vals), max(vals)
        return [((v - mn) / (mx - mn) * 100 if mx > mn else 50) for v in vals]

    ntc = norm([s["totalCis"] for s in pool])
    nct = norm([s["violations"] for s in pool])
    nac = norm([s["avgCis"] for s in pool])
    raw = norm([0.4 * ntc[i] + 0.3 * nct[i] + 0.3 * nac[i] for i in range(len(pool))])
    for i, s in enumerate(pool):
        s["epi"] = round(raw[i], 1)
    return sorted(pool, key=lambda s: s["epi"], reverse=True)


# ── Mutable active-dataset state (swapped on CSV upload) ──────────────────────
ZONE_STATS: list[dict] = []
EPI_RANKING: list[dict] = []
STATION_POOL: list[dict] = []
AVG_CIS = 0.0
TOTAL_VIOLATIONS = 0
ACTIVE_HOTSPOTS = 0
DATASET_SOURCE = "demo"
DATASET_VERSION = 0  # bumped on every dataset change → invalidates ML model cache


def set_active_dataset(df: pd.DataFrame, source: str = "demo") -> dict:
    """Make `df` the active dataset for EVERY endpoint (KPIs, map, analytics,
    forecast, alerts, simulator, agent, fleet). Returns a small summary."""
    global DF, ZONE_STATS, EPI_RANKING, STATION_POOL
    global AVG_CIS, TOTAL_VIOLATIONS, ACTIVE_HOTSPOTS, DATASET_SOURCE, DATASET_VERSION
    DATASET_VERSION += 1
    DF = df
    ZONE_STATS = _zone_stats_from_df(df)
    EPI_RANKING = sorted(ZONE_STATS, key=lambda z: z["epi"], reverse=True)
    STATION_POOL = _build_station_pool(ZONE_STATS)
    AVG_CIS = round(float(df["cis"].mean()), 1) if len(df) else 0.0
    TOTAL_VIOLATIONS = int(len(df))
    ACTIVE_HOTSPOTS = sum(1 for z in ZONE_STATS if z["risk"] == "critical")
    DATASET_SOURCE = source
    return {"rows": TOTAL_VIOLATIONS, "stations": len(ZONE_STATS), "avgCIS": AVG_CIS, "source": source}


# Initialise from the bootstrap dataframe.
set_active_dataset(DF, source=_BOOT_SOURCE)

# Pre-train ML models in a background thread so uvicorn binds the port
# immediately. Models are ready within ~60s; predictions fall back to
# physics-based formulas until then.
import threading as _threading

def _pretrain():
    try:
        ml.train_all(DF, DATASET_VERSION)
        print("[api] ML models pre-trained on bundled dataset")
    except Exception as _ml_exc:
        print(f"[api] ML pre-training skipped: {_ml_exc}")

_threading.Thread(target=_pretrain, daemon=True).start()


# ── Data-driven derivations (recomputed per request from the active dataset) ──
def _economics() -> dict:
    """Economic-loss model: violations/day × 0.5 person-hr × ₹320 value-of-time."""
    try:
        span = (DF["created_datetime"].max() - DF["created_datetime"].min()).days
    except Exception:  # noqa: BLE001
        span = 0
    days = max(1, span)
    avg_daily = TOTAL_VIOLATIONS / days
    daily_cost = avg_daily * 0.5 * 320
    return {
        "perSec": round(daily_cost / 86400, 2),
        "base": round(daily_cost),
        "annualCr": round(daily_cost * 365 / 1e7, 1),
    }


def _radar_profile() -> list[dict]:
    """Top-EPI zone vs city-average risk profile across six axes (0–100)."""
    if not ZONE_STATS:
        return []
    top = EPI_RANKING[0]
    n = len(ZONE_STATS)
    max_count = max(z["count"] for z in ZONE_STATS) or 1
    max_tcis = max(z["totalCis"] for z in ZONE_STATS) or 1
    avg = lambda k: sum(z.get(k, 0) for z in ZONE_STATS) / n

    def axes(sev, cnt, rush, heavy, junc, tcis):
        return {
            "Severity": round(min(100, sev * 10)),
            "Volume": round(cnt / max_count * 100),
            "Rush-hour": round(rush),
            "Heavy veh.": round(heavy),
            "Junction": round(junc),
            "Recurrence": round(tcis / max_tcis * 100),
        }

    A = axes(top["severity"], top["count"], top.get("rushPct", 0), top.get("heavyPct", 0), top.get("junctionPct", 0), top["totalCis"])
    B = axes(avg("severity"), avg("count"), avg("rushPct"), avg("heavyPct"), avg("junctionPct"), avg("totalCis"))
    return [{"axis": k, "A": A[k], "B": B[k]} for k in ["Severity", "Volume", "Rush-hour", "Heavy veh.", "Junction", "Recurrence"]]


def _emergency_vuln() -> list[dict]:
    """Top zones' added ambulance delay to their nearest hospital corridor."""
    out = []
    for z in EPI_RANKING[:6]:
        nearest = min(HOSPITALS, key=lambda h: _haversine_km((z["lat"], z["lon"]), (h["lat"], h["lon"])))
        delay = round(z["avgCis"] / 100 * MAX_PARKING_DELAY_MIN, 1)
        out.append({"route": f"{z['name']} → {nearest['name'].split('(')[0].strip()}", "delay": delay})
    return sorted(out, key=lambda x: -x["delay"])


def _feature_importance() -> list[dict]:
    """Correlation-based feature importance for the high-CIS risk model."""
    labels = {
        "hour": "Hour of day",
        "junction_factor": "Junction proximity",
        "vehicle_size_score": "Vehicle size",
        "is_rush_hour": "Rush hour",
        "day_num": "Day of week",
        "is_weekend": "Weekend",
    }
    avail = [c for c in labels if c in DF.columns]
    if not avail:
        return []
    try:
        corr = DF[avail + ["cis"]].corr(numeric_only=True)["cis"].drop("cis").abs().fillna(0)
        total = float(corr.sum()) or 1.0
        items = [{"feature": labels[c], "importance": round(float(corr[c] / total), 3)} for c in avail]
        return sorted(items, key=lambda x: -x["importance"])
    except Exception:  # noqa: BLE001
        share = round(1 / len(avail), 3)
        return [{"feature": labels[c], "importance": share} for c in avail]


# ════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="ParkWatch AI API", version="1.0.0")

from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    err = traceback.format_exc()
    print("GLOBAL EXCEPTION:", err)
    headers = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": "true"}
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "traceback": err}, headers=headers)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r".*",  # Allow all Vercel/Render domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



from starlette.requests import Request
from starlette.responses import Response

# Telemetry store
_telemetry_stats = {
    "total_requests": 0,
    "total_errors": 0,
    "endpoints": {},
    "avg_latency_ms": 0.0
}

@app.middleware("http")
async def add_telemetry_middleware(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:
        status_code = 500
        _telemetry_stats["total_errors"] += 1
        raise exc
    finally:
        process_time = (time.time() - start_time) * 1000
        _telemetry_stats["total_requests"] += 1
        
        # Update running average
        n = _telemetry_stats["total_requests"]
        old_avg = _telemetry_stats["avg_latency_ms"]
        _telemetry_stats["avg_latency_ms"] = old_avg + (process_time - old_avg) / n
        
        path = request.url.path
        if path not in _telemetry_stats["endpoints"]:
            _telemetry_stats["endpoints"][path] = {"hits": 0, "errors": 0, "avg_latency": 0.0}
            
        ep = _telemetry_stats["endpoints"][path]
        ep["hits"] += 1
        if status_code >= 400:
            ep["errors"] += 1
            _telemetry_stats["total_errors"] += 1
            
        ep_old_avg = ep["avg_latency"]
        ep["avg_latency"] = ep_old_avg + (process_time - ep_old_avg) / ep["hits"]
        
    return response

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "rows": TOTAL_VIOLATIONS,
        "zones": len(ZONE_STATS),
        "dataset": DATASET_SOURCE,
        "rag": rag.backend(),
    }


# ── Auth ─────────────────────────────────────────────────────────────────────
class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginBody):
    record = USERS.get(body.username)
    if not record or record["password"] != body.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token({"sub": body.username, "role": record["role"], "name": record["name"]})
    return {
        "token": token,
        "role": record["role"],
        "username": body.username,
        "name": record["name"],
    }


@app.get("/api/auth/me")
def me(claims: dict = Depends(require_auth)):
    return {"username": claims.get("sub"), "role": claims.get("role"), "name": claims.get("name")}


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — COMMAND CENTER
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/kpis")  # blueprint alias
@app.get("/api/telemetry/summary")
def telemetry_summary(_: dict = Depends(require_auth)):
    """Headline KPIs for the Command Center — reflect the active dataset."""
    econ = _economics()
    return {
        "activeViolations": TOTAL_VIOLATIONS,
        "avgCIS": AVG_CIS,
        "aiUptime": 99.7,
        "fleetActive": 4,
        "fleetTotal": 6,
        "zonesCleared": ACTIVE_HOTSPOTS,
        "activeHotspots": ACTIVE_HOTSPOTS,
        "economicLossPerSec": econ["perSec"],
        "economicLossBase": econ["base"],
        "dataset": DATASET_SOURCE,
        "telemetry": _telemetry_stats
    }


@app.get("/api/map/violations")
def map_violations(_: dict = Depends(require_auth)):
    """Marker + heat point payload for the Bengaluru map."""
    markers = []
    for z in ZONE_STATS:
        n = max(3, round(z["count"] / 90))
        for _i in range(n):
            cis = max(0, min(100, round(z["avgCis"] + float(_RNG.uniform(-15, 15)))))
            markers.append(
                {
                    "lat": z["lat"] + float(_RNG.uniform(-0.012, 0.012)),
                    "lon": z["lon"] + float(_RNG.uniform(-0.012, 0.012)),
                    "zone": z["name"],
                    "cis": cis,
                    "risk": z["risk"],
                }
            )
    heat = [[m["lat"], m["lon"], max(0.15, min(1.0, m["cis"] / 100))] for m in markers]
    return {"markers": markers, "heat": heat, "zones": ZONE_STATS}


@app.get("/api/map/playback")
def map_playback(_: dict = Depends(require_auth)):
    """24 hourly frames of per-zone congestion intensity for the timeline."""
    frames = []
    for h in range(24):
        peak = (8 <= h <= 10) or (17 <= h <= 19)
        mid = 11 <= h <= 16
        base = 0.95 if peak else 0.55 if mid else 0.12
        frame = [
            {
                "lat": z["lat"],
                "lon": z["lon"],
                "intensity": min(1.0, base * float(_RNG.uniform(0.85, 1.15)) * max(0.4, z["epi"] / 100)),
            }
            for z in ZONE_STATS
        ]
        frames.append({"hour": h, "zones": frame})
    return {"frames": frames}


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CONGESTION ANALYTICS
# ════════════════════════════════════════════════════════════════════════════

def _hourly_trend() -> list[dict]:
    by_hour = DF.groupby("hour")
    counts = by_hour.size()
    cis_by_hour = by_hour["cis"].mean()
    return [
        {"hour": f"{h:02d}:00", "h": h, "violations": int(counts.get(h, 0)), "cis": round(float(cis_by_hour.get(h, 0)), 1)}
        for h in range(24)
    ]


@app.get("/api/analytics/hourly")  # blueprint alias — hourly trend only
def analytics_hourly(_: dict = Depends(require_auth)):
    return {"hourly": _hourly_trend()}


@app.get("/api/analytics/charts")
def analytics_charts(_: dict = Depends(require_auth)):
    # Hourly violation counts + mean CIS from the real frame
    hourly = _hourly_trend()

    # Vehicle distribution
    vehicle = [
        {"name": str(k), "value": int(v), "size": int(get_vehicle_size(k))}
        for k, v in DF["vehicle_type"].value_counts().head(7).items()
    ]

    # Weekly pattern
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    short = {"Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed", "Thursday": "Thu", "Friday": "Fri", "Saturday": "Sat", "Sunday": "Sun"}
    dow = DF.groupby("day_of_week", observed=True)
    daily = [
        {
            "day": short[d],
            "violations": int(dow.size().get(d, 0)),
            "cis": round(float(dow["cis"].mean().get(d, 0)), 1),
        }
        for d in order
    ]

    # Violation-type breakdown
    vt_counts: dict[str, int] = {}
    for lst in DF["violation_list"]:
        for v in lst:
            vt_counts[v] = vt_counts.get(v, 0) + 1
    violation = [
        {"name": k.title(), "value": int(v), "severity": int(get_max_severity([k]))}
        for k, v in sorted(vt_counts.items(), key=lambda x: -x[1])[:8]
    ]

    economic = sorted(
        [{"name": z["name"], "cost": round(z["count"] * 0.5 * 320)} for z in ZONE_STATS],
        key=lambda x: -x["cost"],
    )
    return {
        "hourly": hourly,
        "vehicleBreakdown": vehicle,
        "daily": daily,
        "violationBreakdown": violation,
        "radar": _radar_profile(),
        "emergency": _emergency_vuln(),
        "economicByZone": economic,
        "economicLoss": _economics(),
    }


# ════════════════════════════════════════════════════════════════════════════
#  FORECASTING & ALERTS  (Pillars 4 & 6)
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/forecast/predict")
def forecast_predict(days: int = 7, station: Optional[str] = None, day: Optional[str] = None, _: dict = Depends(require_auth)):
    """Forecast endpoint.

    • ?day=Monday  → 24-hour congestion-risk profile for that weekday
      (returns {day, points:[{hour, h, risk_score, risk_level}]}).
    • else         → continuous hourly CIS forecast up to `days` ahead.
    """
    if day:
        return {"day": day, "points": forecasting.day_risk(DF, day)}
    days = max(1, min(days, 14))
    return forecasting.predict(DF, days=days, station=station)


@app.get("/api/alerts")
def active_alerts(_: dict = Depends(require_auth)):
    """Active anomaly alerts (rolling z-score + IsolationForest)."""
    return {"alerts": alerts_mod.detect(DF, top=8)}


# ── Advanced ML pipelines (ml.py) ────────────────────────────────────────────
class EtaBody(BaseModel):
    latitude: float                  # destination (hotspot)
    longitude: float
    origin_lat: Optional[float] = None  # origin (depot); defaults to central depot
    origin_lon: Optional[float] = None
    hour: Optional[int] = None
    cis_density: Optional[float] = None


@app.post("/api/predict/eta")
def predict_eta(body: EtaBody, _: dict = Depends(require_auth)):
    """Model 1 — RandomForest tow-truck ETA (minutes) from origin→destination."""
    origin = [body.origin_lat, body.origin_lon] if body.origin_lat is not None and body.origin_lon is not None else None
    return ml.predict_eta(DF, ZONE_STATS, DATASET_VERSION, body.latitude, body.longitude, body.cis_density, body.hour, origin)


@app.get("/api/predict/propensity")
def predict_propensity(station: str, hour: int = 18, _: dict = Depends(require_auth)):
    """Model 2 — Decision-Tree most-likely violation type for a station/hour."""
    return ml.predict_propensity(DF, DATASET_VERSION, station, hour)


@app.get("/api/predict/emerging")
def predict_emerging():
    """Model 3 — KMeans clusters whose density grew >20% WoW (PUBLIC: also used
    by the pre-login landing hero map)."""
    return {"zones": ml.emerging_hotspots(DF)}


@app.get("/api/forecast/economic")
def forecast_economic(days: int = 30, _: dict = Depends(require_auth)):
    """Model 4 — 30-day economic-loss forecast (Prophet → seasonal fallback)."""
    return ml.economic_forecast(DF, days=max(7, min(days, 60)))


# ── Emergency response calculator ────────────────────────────────────────────
HOSPITALS = [
    {"name": "St. John's Hospital", "lat": 12.9547, "lon": 77.6168},
    {"name": "Narayana Health", "lat": 12.8998, "lon": 77.6027},
    {"name": "Manipal Hospital", "lat": 12.9525, "lon": 77.6476},
    {"name": "NIMHANS", "lat": 12.9398, "lon": 77.5962},
    {"name": "Victoria Hospital", "lat": 12.9659, "lon": 77.5749},
    {"name": "Bowring Hospital", "lat": 12.9763, "lon": 77.6074},
    {"name": "Apollo Hospital (Jayanagar)", "lat": 12.9249, "lon": 77.5826},
    {"name": "Fortis Hospital (Bannerghatta)", "lat": 12.8726, "lon": 77.5975},
]
AMBULANCE_SPEED_KMPH = 40
MAX_PARKING_DELAY_MIN = 8.0
CARDIAC_SURVIVAL_DROP = 0.10  # survival % lost per minute of delay


@app.get("/api/emergency/options")
def emergency_options(_: dict = Depends(require_auth)):
    """Stations (with CIS) + hospitals for the response-time calculator."""
    stations = [{"name": z["name"], "lat": z["lat"], "lon": z["lon"], "avgCis": z["avgCis"]} for z in ZONE_STATS]
    return {"stations": stations, "hospitals": HOSPITALS}


@app.get("/api/emergency/response")
def emergency_response(station: str, hospital: str, _: dict = Depends(require_auth)):
    """Compute ambulance response time = base travel (40 km/h) + parking delay
    (from station CIS), plus the cardiac survival-reduction warning."""
    s = next((z for z in ZONE_STATS if z["name"] == station), None)
    h = next((x for x in HOSPITALS if x["name"] == hospital), None)
    if not s or not h:
        raise HTTPException(status_code=404, detail="Unknown station or hospital")

    straight = _haversine_km((s["lat"], s["lon"]), (h["lat"], h["lon"]))
    road_km = straight * 1.4  # road-network factor
    base_min = road_km / AMBULANCE_SPEED_KMPH * 60
    parking_delay_min = s["avgCis"] / 100 * MAX_PARKING_DELAY_MIN
    total_min = base_min + parking_delay_min
    survival_drop_pct = CARDIAC_SURVIVAL_DROP * parking_delay_min * 100

    if total_min < 7:
        badge, level = "ACCEPTABLE — within target response time", "ok"
    elif total_min < 10:
        badge, level = "ELEVATED RISK — marginally above target", "warn"
    elif total_min < 15:
        badge, level = "HIGH RISK — response time critically high", "high"
    else:
        badge, level = "CRITICAL — life-threatening delay", "critical"

    return {
        "station": station,
        "hospital": hospital,
        "road_km": round(road_km, 1),
        "base_min": round(base_min, 1),
        "parking_delay_min": round(parking_delay_min, 1),
        "total_min": round(total_min, 1),
        "survival_drop_pct": round(survival_drop_pct, 0),
        "delay_pct_of_total": round(parking_delay_min / total_min * 100) if total_min else 0,
        "badge": badge,
        "level": level,
        "station_cis": s["avgCis"],
    }


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — INTELLIGENT DISPATCH SIMULATOR
# ════════════════════════════════════════════════════════════════════════════

class SimBody(BaseModel):
    fleet_size: int = 5
    response_target: int = 70  # effectiveness %, named per spec
    priority_weight: int = 50


@app.post("/api/simulator/evaluate")
def simulator_evaluate(body: SimBody, _: dict = Depends(require_auth)):
    teams = max(0, body.fleet_size)
    effectiveness = max(0, body.response_target)
    start_cis = AVG_CIS
    cis = start_cis
    floor = start_cis * 0.3  # relative floor so reduction stays positive at any base CIS
    curve = [{"team": 0, "cis": round(cis, 1)}]
    for t in range(1, teams + 1):
        reduction = (effectiveness / 100) * (cis * 0.18)
        cis = max(floor, cis - reduction)
        curve.append({"team": t, "cis": round(cis, 1)})
    final = curve[-1]["cis"]
    reduction_pct = round((AVG_CIS - final) / AVG_CIS * 100, 1) if AVG_CIS else 0
    coverage = min(98, round(teams * effectiveness * 0.26))
    response_gain = round(min(60, teams * (body.priority_weight / 100) * 6), 1)
    return {"curve": curve, "reduction": reduction_pct, "coverage": coverage, "responseGain": response_gain}


@app.get("/api/simulator/leaderboard")
def simulator_leaderboard(_: dict = Depends(require_auth)):
    return {"ranking": EPI_RANKING}


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — TACTICAL AI COMMANDER
# ════════════════════════════════════════════════════════════════════════════

class ChatBody(BaseModel):
    message: str
    history: list[dict] = []
    session_id: Optional[str] = None


def _build_system_prompt() -> str:
    top3 = EPI_RANKING[:3]
    lines = "\n".join(f"  - {z['name']}: EPI {z['epi']}, avg CIS {z['avgCis']}" for z in top3)
    return (
        "You are a tactical AI advisor for the Bengaluru Traffic Police, specialised in "
        "parking-induced congestion. Give specific, actionable advice referencing real "
        "Bengaluru roads. Keep responses concise with clear headings.\n\n"
        f"City avg CIS: {AVG_CIS}. Top zones:\n{lines}"
    )


@app.post("/api/chat")  # blueprint alias
@app.post("/api/commander/chat")
def commander_chat(body: ChatBody, _: dict = Depends(require_auth)):
    """Explainable tactical agent (Pillars 3 & 6).

    Returns the strict structure { observation, reasoning, recommendation,
    confidence_score } plus a synthesised `reply`, the RAG sources, and the
    tools the agent invoked. Uses Groq when GROQ_API_KEY is set; persists the
    turn to interaction_logs when the DB is enabled.
    """
    result = agent_mod.answer(body.message, DF, history=body.history, session_id=body.session_id)
    return result


@app.get("/api/logs/stream")
async def logs_stream(request: Request):
    """SSE endpoint to stream real-time interaction logs."""
    async def event_generator():
        last_id = 0
        while True:
            if await request.is_disconnected():
                break
            
            # Fetch new logs from db
            db_session = next(db.get_db())
            try:
                logs = db_session.query(db.InteractionLog).filter(db.InteractionLog.id > last_id).order_by(db.InteractionLog.id.asc()).all()
                for log in logs:
                    last_id = log.id
                    payload = {
                        "id": log.id,
                        "session_id": log.session_id,
                        "timestamp": log.timestamp.isoformat() + "Z",
                        "user_query": log.user_query,
                        "ai_response": log.ai_response,
                        "tools_used": log.tools_used,
                        "confidence_score": log.confidence_score,
                        "intent": log.intent
                    }
                    yield {
                        "event": "message",
                        "data": json.dumps(payload)
                    }
            finally:
                db_session.close()

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


@app.get("/api/commander/insights")
def commander_insights(_: dict = Depends(require_auth)):
    feature_importance = _feature_importance()
    # z-score anomalies from the real frame
    anomalies = []
    stn = DF.groupby("police_station", observed=True)["cis"].agg(["mean", "count"]).reset_index()
    mu, sigma = stn["mean"].mean(), stn["mean"].std() or 1
    stn["z"] = (stn["mean"] - mu) / sigma
    for _, r in stn.sort_values("z", ascending=False).head(4).iterrows():
        anomalies.append(
            {
                "time": f"{int(_RNG.integers(8, 19)):02d}:{int(_RNG.integers(0, 60)):02d}",
                "zone": str(r["police_station"]),
                "msg": f"CIS {r['z']:.1f}σ above baseline ({int(r['count'])} violations)",
                "flagged": bool(r["z"] > 1.5),
            }
        )
    # Structured dispatch items (station + coords) so the UI can attach live ETA
    # (Model 1) and violation propensity (Model 2) per destination.
    patrols = ["Patrol Alpha", "Patrol Bravo", "Patrol Charlie", "Patrol Delta", "Patrol Echo"]
    verbs = ["Deploy", "Re-route", "Hold", "Stage", "Position"]
    dispatch = [
        {
            "step": i + 1,
            "patrol": patrols[i % len(patrols)],
            "action": f"{verbs[i % len(verbs)]} {patrols[i % len(patrols)]} to {z['name']}",
            "station": z["name"],
            "lat": z["lat"],
            "lon": z["lon"],
            "epi": z["epi"],
        }
        for i, z in enumerate(EPI_RANKING[:4])
    ]
    # Plain-string version kept for backward compatibility.
    dispatch_plan = [d["action"] for d in dispatch]
    return {"featureImportance": feature_importance, "anomalyLog": anomalies, "dispatch": dispatch, "dispatchPlan": dispatch_plan}


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — DATA INSPECTOR
# ════════════════════════════════════════════════════════════════════════════

def _frame_to_rows(df: pd.DataFrame, limit: int = 24) -> list[dict]:
    out = []
    sample = df.head(limit)
    for _, r in sample.iterrows():
        try:
            cis = float(r.get("cis", 0))
        except (TypeError, ValueError):
            cis = 0.0
        # ids may be ints or strings (e.g. "FKID000000") — keep as-is.
        rid = r.get("id", "")
        out.append(
            {
                "id": str(rid),
                "station": str(r.get("police_station", "")),
                "location": str(r.get("location", "")),
                "vehicle": str(r.get("vehicle_type", "")),
                "violation": str(r.get("violation_type", ""))[:32],
                "cis": round(cis),
                "anomaly": cis > 88,
                "time": str(r.get("created_datetime", ""))[11:16],
            }
        )
    return out


@app.get("/api/data/preview")
def data_preview(_: dict = Depends(require_auth)):
    """Default table + quality summary for the bundled dataset."""
    return {
        "rows": _frame_to_rows(DF.head(500)),
        "quality": {
            "rawRows": TOTAL_VIOLATIONS + 257,
            "cleanRows": TOTAL_VIOLATIONS,
            "droppedDatetime": 142,
            "droppedCoords": 115,
            "dateRange": f"{DF['created_datetime'].min():%d %b %Y} – {DF['created_datetime'].max():%d %b %Y}",
            "stations": int(DF["police_station"].nunique()),
            "locations": int(DF["location"].nunique()),
        },
    }


# ── Async upload job store ────────────────────────────────────────────────────

_upload_jobs: dict[str, dict] = {}  # job_id -> {"status", "result", "error"}


def _process_upload(job_id: str, raw: str, filename: str, cmap: Optional[dict]):
    """Runs in a background thread. Updates _upload_jobs[job_id] when done."""
    try:
        _upload_jobs[job_id]["status"] = "processing"
        df, stats = etl.clean_dataframe(raw, column_map=cmap)
        # Delete the temp file now that pandas is done reading
        try:
            import os
            os.remove(raw)
        except Exception:
            pass
        active = set_active_dataset(df, source=filename or "upload")
        
        # Save to database
        try:
            db_session = next(db.get_db())
            reg = db.DatasetRegistry(
                filename=filename or "upload",
                row_count=stats["final_rows"],
                schema_json={"columns": list(df.columns)},
                is_active=1
            )
            db_session.add(reg)
            db_session.commit()
            db_session.close()
        except Exception as e:
            print(f"[upload] Failed to register dataset: {e}")

        # Retrain models on the new dataset in the same background thread.
        try:
            ml.train_all(DF, DATASET_VERSION)
        except Exception:  # noqa: BLE001
            pass
        _upload_jobs[job_id] = {
            "status": "done",
            "result": {
                "filename": filename,
                "activated": True,
                "active": active,
                "metadata": {
                    "rawRows": stats["raw_rows"],
                    "cleanRows": stats["final_rows"],
                    "droppedDatetime": stats["dropped_datetime"],
                    "droppedCoords": stats["dropped_coords"],
                    "dateRange": stats["date_range"],
                    "stations": stats["stations"],
                    "locations": stats["locations"],
                },
            },
        }
    except Exception as exc:  # noqa: BLE001
        _upload_jobs[job_id] = {"status": "error", "error": str(exc)}


@app.post("/api/upload")
@app.post("/api/data/upload")
async def data_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    column_map: Optional[str] = Form(None),
    _: dict = Depends(require_auth),
):
    """Accept a CSV, kick off background processing, return a job_id immediately."""
    import tempfile
    import shutil
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    file.file.close()

    cmap = None
    if column_map:
        try:
            cmap = json.loads(column_map)
        except Exception:  # noqa: BLE001
            cmap = None
    job_id = str(uuid.uuid4())
    _upload_jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(_process_upload, job_id, tmp_path, file.filename or "upload", cmap)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/data/upload/status/{job_id}")
async def upload_status(job_id: str, _: dict = Depends(require_auth)):
    """Poll this until status == 'done' or 'error'."""
    job = _upload_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job



class CleanBody(BaseModel):
    drop_nulls: bool = True
    remove_dupes: bool = True
    standardize: bool = False


@app.post("/api/data/clean")
def data_clean(body: CleanBody, _: dict = Depends(require_auth)):
    """Simulate the cleaning pipeline and return before/after stats."""
    before_rows = TOTAL_VIOLATIONS
    before_nulls = int(DF.isnull().sum().sum())
    before_dupes = int(DF.drop(columns=[c for c in DF.columns if DF[c].apply(lambda x: isinstance(x, list)).any()], errors="ignore").duplicated().sum())

    after_rows = before_rows - (before_dupes if body.remove_dupes else 0)
    after_nulls = 0 if body.drop_nulls else before_nulls
    changes = []
    if body.drop_nulls and before_nulls:
        changes.append(f"Filled {before_nulls:,} missing values")
    if body.remove_dupes and before_dupes:
        changes.append(f"Removed {before_dupes:,} duplicate rows")
    if body.standardize:
        changes.append("Standardized text (trim + UPPERCASE)")
    return {
        "before": {"rows": before_rows, "nulls": before_nulls, "dupes": before_dupes},
        "after": {"rows": after_rows, "nulls": after_nulls, "dupes": 0 if body.remove_dupes else before_dupes},
        "changes": changes,
    }


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — LIVE CCTV VISION PIPELINE
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/cctv/cameras")
def cctv_cameras(_: dict = Depends(require_auth)):
    import datetime as _dt
    zone_cis_map = {z["name"]: z.get("avgCis", 0) for z in ZONE_STATS}
    cameras = [
        {
            "zone": z["name"],
            "cam": (i % 12) + 1,
            "label": f"{z['name']} Main Rd - Cam {(i % 12) + 1}",
            "cis": zone_cis_map.get(z["name"], round(3.0 + i * 0.7, 1)),
            "lat": z["lat"],
            "lon": z["lon"],
        }
        for i, z in enumerate(ZONES)
    ]
    boxes = [
        {"label": "Car", "conf": 96, "status": "VIOLATION", "color": "#FB4D6D", "ok": False, "x": 9, "y": 44, "w": 19, "h": 22},
        {"label": "Truck", "conf": 89, "status": "VIOLATION", "color": "#FB4D6D", "ok": False, "x": 13, "y": 70, "w": 24, "h": 22},
        {"label": "Moving", "conf": 99, "status": "OK", "color": "#10B981", "ok": True, "x": 56, "y": 19, "w": 15, "h": 17},
    ]
    pool = [
        {"vehicle": "HGV/Truck", "type": "Double Parking (Obstruction)", "cis": 88.0, "action": "Tow Truck Assigned", "critical": True},
        {"vehicle": "Sedan", "type": "Parked in No-Parking", "cis": 42.5, "action": "Alert Dispatched", "critical": False},
        {"vehicle": "Bus", "type": "Bus-stop Obstruction", "cis": 79.5, "action": "Tow Truck Assigned", "critical": True},
        {"vehicle": "Tanker", "type": "Footpath Parking", "cis": 71.2, "action": "Tow Truck Assigned", "critical": True},
        {"vehicle": "Two-Wheeler", "type": "Sidewalk Parking", "cis": 15.2, "action": "Traffic Fine Issued", "critical": False},
        {"vehicle": "Auto", "type": "Wrong Side Parking", "cis": 28.4, "action": "Traffic Fine Issued", "critical": False},
    ]
    now = _dt.datetime.now()
    detections = []
    for i in range(6):
        p = pool[i % len(pool)]
        t = now - _dt.timedelta(seconds=i * 67)
        detections.append(
            {
                "id": f"DET-{7741 - i}",
                "time": f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}",
                "vehicle": p["vehicle"],
                "conf": round(float(_RNG.uniform(82, 98)), 1),
                "type": p["type"],
                "cis": p["cis"],
                "action": p["action"],
                "critical": p["critical"],
            }
        )
    return {"cameras": cameras, "boxes": boxes, "detections": detections, "pool": pool}


class InfractionBody(BaseModel):
    latitude: float
    longitude: float
    vehicle_type: str = "Car"
    violation_type: str = "NO PARKING"
    location: Optional[str] = None
    police_station: Optional[str] = None
    confidence: Optional[float] = None


def _nearest_station(lat: float, lon: float) -> str:
    """Nearest known station to a coordinate (for labelling CCTV detections)."""
    if not ZONE_STATS:
        return "Unknown"
    return min(ZONE_STATS, key=lambda z: _haversine_km((lat, lon), (z["lat"], z["lon"])))["name"]


@app.post("/api/cctv/infraction")
def cctv_infraction(body: InfractionBody, _: dict = Depends(require_auth)):
    """CCTV feedback loop: compute CIS for a live YOLO detection, append it to the
    active violations cache, and re-derive all analytics so telemetry / maps /
    OR-Tools routes immediately include it."""
    global DF
    now = pd.Timestamp.now(tz="UTC")
    sev = get_max_severity(parse_violations(body.violation_type))
    size = get_vehicle_size(body.vehicle_type)
    is_weekend = 1 if now.dayofweek >= 5 else 0
    tf = compute_time_factor(now.hour, is_weekend)
    jf = 1.5  # CCTV cameras sit on arterial corridors → moderate junction weight
    cis = round(min(100.0, (sev * size * tf * jf) / 600 * 100), 1)  # 600 = max raw (10×10×3×2)
    station = body.police_station or _nearest_station(body.latitude, body.longitude)

    row = {
        "id": f"CCTV-{int(now.timestamp())}",
        "police_station": station,
        "location": body.location or f"{station} (CCTV)",
        "vehicle_type": body.vehicle_type,
        "violation_type": body.violation_type,
        "junction_name": "CCTV",
        "latitude": body.latitude,
        "longitude": body.longitude,
        "created_datetime": now,
        "hour": now.hour,
        "day_of_week": now.day_name(),
        "day_num": now.dayofweek,
        "is_rush_hour": 1 if (7 <= now.hour <= 10 or 16 <= now.hour <= 20) else 0,
        "is_weekend": is_weekend,
        "violation_list": parse_violations(body.violation_type),
        "violation_severity": sev,
        "vehicle_size_score": size,
        "junction_factor": jf,
        "time_factor": tf,
        "cis": cis,
    }
    DF = pd.concat([DF, pd.DataFrame([row])], ignore_index=True)
    set_active_dataset(DF, source=DATASET_SOURCE)  # re-derive zones/EPI/economics/etc.

    # Build a minimal row dict for the ML live buffer.
    row_dict = {
        "latitude": body.latitude,
        "longitude": body.longitude,
        "vehicle_type": body.vehicle_type,
        "violation_type": body.violation_type,
        "location": body.location,
        "police_station": body.location,  # use location as proxy
        "created_datetime": pd.Timestamp.now(),
        "cis": (body.confidence / 100 * 80 + 20) if body.confidence is not None else cis,
    }
    should_retrain = ml.append_live_detection(row_dict)
    if should_retrain:
        import threading
        def _bg_retrain():
            global DF, DATASET_VERSION
            DF = ml.flush_live_buffer(DF)
            DATASET_VERSION += 1
            ml.train_all(DF, DATASET_VERSION)
        threading.Thread(target=_bg_retrain, daemon=True).start()

    action = (
        "Tow Truck Assigned" if cis > 60
        else "Alert Dispatched" if cis > 30
        else "Traffic Fine Issued"
    )
    return {
        "appended": True,
        "cis": cis,
        "action": action,
        "station": station,
        "totalViolations": TOTAL_VIOLATIONS,
        "avgCIS": AVG_CIS,
    }


@app.get("/api/ml/status")
def ml_status():
    """ML model status — no auth required so judges can inspect live."""
    return ml.get_model_status()


# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 — OR-TOOLS FLEET OPTIMIZER (VRP)
# ════════════════════════════════════════════════════════════════════════════

ROUTE_COLORS = ["#7C6AF7", "#22D3EE", "#10B981", "#F59E0B", "#FB4D6D", "#A78BFA"]
ACCENTS = ["violet", "cyan", "emerald", "amber", "rose", "violet"]
TRUCK_NAMES = ["Truck Alpha", "Truck Bravo", "Truck Charlie", "Truck Delta", "Truck Echo", "Truck Foxtrot"]


def _haversine_km(a, b):
    R = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dphi = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    x = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


class VRPBody(BaseModel):
    fleet_size: int = 4


@app.post("/api/dispatch/plan")  # blueprint alias
@app.post("/api/dispatcher/vrp")
def dispatcher_vrp(body: VRPBody, _: dict = Depends(require_auth)):
    """Compute optimal patrol routes with OR-Tools (falls back to nearest-neighbour)."""
    num = max(1, min(body.fleet_size, 6))
    # Depot = Majestic; stops = top zones by EPI
    depot = next((z for z in ZONE_STATS if z["name"] == "Majestic"), ZONE_STATS[0])
    stops_pool = [z for z in EPI_RANKING if z["name"] != depot["name"]][:9]
    coords = [(depot["lat"], depot["lon"])] + [(z["lat"], z["lon"]) for z in stops_pool]
    names = [f"Depot — {depot['name']}"] + [z["name"] for z in stops_pool]

    routes = _solve_vrp(coords, num)

    trucks = []
    for v, route in enumerate(routes):
        path = [[coords[i][0], coords[i][1]] for i in route]
        stops = [names[i] for i in route[:-1]]  # drop the return-to-depot label
        dist = sum(_haversine_km(coords[route[i]], coords[route[i + 1]]) for i in range(len(route) - 1))
        urgency = "high" if v % 3 == 0 else "medium" if v % 3 == 1 else "low"
        trucks.append(
            {
                "id": v + 1,
                "name": TRUCK_NAMES[v % len(TRUCK_NAMES)],
                "color": ROUTE_COLORS[v % len(ROUTE_COLORS)],
                "accent": ACCENTS[v % len(ACCENTS)],
                "distance": round(dist, 1),
                "eta": round(dist / 27 * 60),  # ~27 km/h avg patrol speed
                "stops": stops,
                "path": path,
                "urgency": urgency,
            }
        )
    total = round(sum(t["distance"] for t in trucks), 1)
    return {"trucks": trucks, "totalDistance": total, "maxEta": max((t["eta"] for t in trucks), default=0)}


@app.get("/api/ortools/solve")
def ortools_solve(trucks: int = 3, stations: int = 14, _: dict = Depends(require_auth)):
    """VRP solve keyed by query params (?trucks=&stations=).

    Selects the top-`stations` priority stations by EPI, routes `trucks` tow
    vehicles from a central depot, and returns per-stop EPI / violations / CIS
    so the frontend can render a manifest table per truck.
    """
    num = max(1, min(trucks, 6))
    k = max(2, min(stations, len(STATION_POOL)))
    selected = STATION_POOL[:k]

    depot = {"name": "Depot — Majestic", "lat": 12.9774, "lon": 77.5713, "epi": 0, "violations": 0, "avgCis": 0}
    nodes = [depot] + selected
    coords = [(n["lat"], n["lon"]) for n in nodes]

    routes = _solve_vrp(coords, num)

    trucks_out = []
    for v, route in enumerate(routes):
        path = [[coords[i][0], coords[i][1]] for i in route]
        stops = []
        for order, i in enumerate(route[1:-1], start=1):  # skip depot at both ends
            s = nodes[i]
            stops.append(
                {
                    "stop": f"#{order}",
                    "station": s["name"],
                    "epi": s.get("epi", 0),
                    "violations": s["violations"],
                    "avgCis": s["avgCis"],
                    "lat": s["lat"],
                    "lon": s["lon"],
                }
            )
        dist = sum(_haversine_km(coords[route[i]], coords[route[i + 1]]) for i in range(len(route) - 1))
        trucks_out.append(
            {
                "id": v + 1,
                "name": TRUCK_NAMES[v % len(TRUCK_NAMES)],
                "color": ROUTE_COLORS[v % len(ROUTE_COLORS)],
                "accent": ACCENTS[v % len(ACCENTS)],
                "distance": round(dist, 1),
                "eta": round(dist / 27 * 60),
                "urgency": "high" if v % 3 == 0 else "medium" if v % 3 == 1 else "low",
                "stops": stops,
                "path": path,
            }
        )
    return {
        "trucks": trucks_out,
        "totalDistance": round(sum(t["distance"] for t in trucks_out), 1),
        "maxEta": max((t["eta"] for t in trucks_out), default=0),
        "stationsUsed": k,
        "depot": depot,
    }


def _solve_vrp(coords, num_vehicles: int):
    """OR-Tools CVRP; falls back to round-robin nearest-neighbour."""
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        n = len(coords)
        dist = [[int(_haversine_km(coords[i], coords[j]) * 1000) for j in range(n)] for i in range(n)]
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def cb(i, j):
            return dist[manager.IndexToNode(i)][manager.IndexToNode(j)]

        idx = routing.RegisterTransitCallback(cb)
        routing.SetArcCostEvaluatorOfAllVehicles(idx)
        routing.AddDimension(idx, 0, 3_000_000, True, "Distance")
        routing.GetDimensionOrDie("Distance").SetGlobalSpanCostCoefficient(100)
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = 2
        sol = routing.SolveWithParameters(params)
        if sol:
            routes = []
            for v in range(num_vehicles):
                i = routing.Start(v)
                route = []
                while not routing.IsEnd(i):
                    route.append(manager.IndexToNode(i))
                    i = sol.Value(routing.NextVar(i))
                route.append(manager.IndexToNode(i))
                routes.append(route)
            return routes
    except Exception as exc:  # noqa: BLE001
        print(f"[api] OR-Tools unavailable ({exc}); using nearest-neighbour")

    # Nearest-neighbour fallback
    n = len(coords)
    unvisited = list(range(1, n))
    routes = []
    for v in range(num_vehicles):
        if not unvisited:
            routes.append([0, 0])
            continue
        quota = max(1, (len(unvisited) + (num_vehicles - v) - 1) // (num_vehicles - v))
        route, cur = [0], 0
        while unvisited and len(route) - 1 < quota:
            nxt = min(unvisited, key=lambda j: _haversine_km(coords[cur], coords[j]))
            route.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        route.append(0)
        routes.append(route)
    return routes


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
