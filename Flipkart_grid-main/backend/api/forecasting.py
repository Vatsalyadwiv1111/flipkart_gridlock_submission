"""
backend/forecasting.py — Continuous congestion forecasting (Pillar 4).

Predicts hourly average CIS up to N days ahead with weekly + daily seasonality.

Engine selection (best available wins, all optional):
  1. Prophet            — additive seasonality, holiday-aware
  2. XGBoost Regressor  — gradient-boosted on calendar features
  3. Seasonal-naive     — (day-of-week × hour) profile  [always available]

The seasonal fallback is pure pandas/numpy so /api/forecast/predict always works.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one mean-CIS value per calendar hour."""
    s = df.dropna(subset=["created_datetime"]).copy()
    s["ts"] = s["created_datetime"].dt.floor("h")
    g = s.groupby("ts")["cis"].mean().reset_index()
    return g


def _seasonal_profile(df: pd.DataFrame) -> dict:
    """Mean CIS keyed by (day_of_week, hour)."""
    p = df.groupby([df["created_datetime"].dt.dayofweek, df["created_datetime"].dt.hour])["cis"].mean()
    return {(int(d), int(h)): float(v) for (d, h), v in p.items()}


_DOW = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def day_risk(df: pd.DataFrame, day_name: str) -> list[dict]:
    """24-hour congestion-risk profile for a given weekday.

    Blends predicted CIS intensity (60%) with violation-volume intensity (40%)
    for that day-of-week, scaled to a 0–100 risk_score, with a rush-hour bump.
    Returns 24 objects: {hour, h, risk_score, risk_level}.
    """
    if df is None or df.empty:
        return [{"hour": f"{h:02d}:00", "h": h, "risk_score": 0.0, "risk_level": "Low"} for h in range(24)]

    dow = _DOW.get(str(day_name).strip().lower(), 0)
    sub = df[df["created_datetime"].dt.dayofweek == dow]
    if sub.empty:
        sub = df  # fall back to all days if that weekday has no records

    by_hour = sub.groupby(sub["created_datetime"].dt.hour)
    cis_h = by_hour["cis"].mean()
    cnt_h = by_hour.size()
    cmax = float(cis_h.max()) or 1.0
    nmax = float(cnt_h.max()) or 1.0

    out = []
    for h in range(24):
        cis = float(cis_h.get(h, 0.0))
        cnt = int(cnt_h.get(h, 0))
        score = 100.0 * (0.6 * (cis / cmax) + 0.4 * (cnt / nmax))
        if h in (8, 9, 10, 17, 18, 19):  # rush-hour amplification
            score = min(100.0, score * 1.12)
        level = "Critical" if score >= 75 else "High" if score >= 50 else "Medium" if score >= 25 else "Low"
        out.append({"hour": f"{h:02d}:00", "h": h, "risk_score": round(score, 1), "risk_level": level})
    return out


def _engine() -> str:
    try:
        import prophet  # noqa: F401

        return "prophet"
    except Exception:  # noqa: BLE001
        pass
    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except Exception:  # noqa: BLE001
        return "seasonal"


def predict(df: pd.DataFrame, days: int = 7, station: Optional[str] = None) -> dict:
    """Return {engine, station, points:[{ts, hour, day, yhat, lower, upper}]}."""
    work = df if station in (None, "", "all") else df[df["police_station"] == station]
    if work is None or work.empty:
        return {"engine": "none", "station": station, "points": []}

    horizon = int(days) * 24
    last_ts = work["created_datetime"].max().floor("h")
    future = [last_ts + pd.Timedelta(hours=i + 1) for i in range(horizon)]
    engine = _engine()

    # ── Prophet ──
    if engine == "prophet":
        try:
            from prophet import Prophet

            hs = _hourly_series(work).rename(columns={"ts": "ds", "cis": "y"})
            m = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=False)
            m.fit(hs)
            fut = pd.DataFrame({"ds": future})
            fc = m.predict(fut)
            pts = [
                {
                    "ts": ts.isoformat(),
                    "hour": int(ts.hour),
                    "day": ts.strftime("%a %d"),
                    "yhat": round(float(max(0, y)), 1),
                    "lower": round(float(max(0, lo)), 1),
                    "upper": round(float(up), 1),
                }
                for ts, y, lo, up in zip(future, fc["yhat"], fc["yhat_lower"], fc["yhat_upper"])
            ]
            return {"engine": "prophet", "station": station or "all", "points": pts}
        except Exception:  # noqa: BLE001
            engine = "seasonal"

    # ── XGBoost ──
    if engine == "xgboost":
        try:
            from xgboost import XGBRegressor

            hs = _hourly_series(work)
            X = pd.DataFrame(
                {
                    "hour": hs["ts"].dt.hour,
                    "dow": hs["ts"].dt.dayofweek,
                    "month": hs["ts"].dt.month,
                    "is_weekend": (hs["ts"].dt.dayofweek >= 5).astype(int),
                }
            )
            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, verbosity=0)
            model.fit(X, hs["cis"])
            fdf = pd.DataFrame(
                {
                    "hour": [t.hour for t in future],
                    "dow": [t.dayofweek for t in future],
                    "month": [t.month for t in future],
                    "is_weekend": [1 if t.dayofweek >= 5 else 0 for t in future],
                }
            )
            yhat = model.predict(fdf)
            resid = float(np.std(hs["cis"] - model.predict(X))) or 5.0
            pts = [
                {
                    "ts": ts.isoformat(),
                    "hour": int(ts.hour),
                    "day": ts.strftime("%a %d"),
                    "yhat": round(float(max(0, y)), 1),
                    "lower": round(float(max(0, y - resid)), 1),
                    "upper": round(float(y + resid), 1),
                }
                for ts, y in zip(future, yhat)
            ]
            return {"engine": "xgboost", "station": station or "all", "points": pts}
        except Exception:  # noqa: BLE001
            engine = "seasonal"

    # ── Seasonal-naive fallback ──
    profile = _seasonal_profile(work)
    overall = float(work["cis"].mean())
    band = float(work.groupby(work["created_datetime"].dt.hour)["cis"].std().mean()) or 6.0
    pts = []
    for ts in future:
        base = profile.get((ts.dayofweek, ts.hour), overall)
        pts.append(
            {
                "ts": ts.isoformat(),
                "hour": int(ts.hour),
                "day": ts.strftime("%a %d"),
                "yhat": round(max(0, base), 1),
                "lower": round(max(0, base - band * 0.6), 1),
                "upper": round(base + band * 0.6, 1),
            }
        )
    return {"engine": "seasonal", "station": station or "all", "points": pts}
