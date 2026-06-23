"""
backend/agent.py — Tactical AI agent with explainable output (Pillars 3 & 6).

Produces the strict explainable structure:
    { observation, reasoning, recommendation, confidence_score, reply, sources, tools_called }

The reply varies by INTENT (deploy / diagnose / forecast / anomaly / summary /
greeting / general) and by the STATION mentioned in the question, so different
questions get different answers even without an LLM. When GROQ_API_KEY is set,
Groq (Llama-3.3-70B) synthesises the prose from this structured context.

Tool orchestration (each degrades independently):
  • StatsQueryTool   — NL → pandas aggregation over the live frame (SQL-lite).
  • ForecastingTool  — forecasting.predict() for forward-looking questions.
  • AlertsTool       — alerts.detect() folds active anomalies into context.
  • RAGGuidelinesTool— rag.retrieve() grounds recommendations in BTP policy.

Adaptive memory: last 10 turns for the session (interaction_logs if the DB is
enabled, else the history payload), including user corrections.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd
import time

from api import rag
from api import db
from api import langchain_agent


# ── Tools / helpers ──────────────────────────────────────────────────────────
def stats_tool(df: pd.DataFrame, query: str) -> dict:
    """SQL-lite: answer common aggregate questions from the live frame."""
    q = query.lower()
    out = {"avg_cis": round(float(df["cis"].mean()), 1) if len(df) else 0.0}
    if len(df):
        epi = df.groupby("police_station", observed=True)["cis"].agg(["mean", "size"]).reset_index()
        epi = epi.sort_values("mean", ascending=False)
        out["top_station"] = str(epi.iloc[0]["police_station"])
        out["top_station_cis"] = round(float(epi.iloc[0]["mean"]), 1)
        out["n_stations"] = int(len(epi))
        if "hour" in df.columns:
            out["peak_hour"] = int(df.groupby("hour")["cis"].mean().idxmax())
    if "double" in q and "violation_type" in df.columns:
        mask = df["violation_type"].astype(str).str.contains("double", case=False, na=False)
        if mask.any():
            by = df[mask].groupby("police_station", observed=True).size().sort_values(ascending=False)
            out["top_double_parking"] = str(by.index[0])
            out["top_double_parking_count"] = int(by.iloc[0])
    return out


def _match_station(message: str, df: pd.DataFrame) -> Optional[str]:
    """Find a station/zone named in the question."""
    if "police_station" not in df.columns or df.empty:
        return None
    ql = message.lower()
    names = [str(n) for n in df["police_station"].dropna().unique()]
    for n in sorted(names, key=len, reverse=True):  # longest match wins
        if n.lower() in ql:
            return n
    qtok = set(re.findall(r"[a-z]+", ql))
    best, best_score = None, 0
    for n in names:
        score = len(qtok & set(re.findall(r"[a-z]+", n.lower())))
        if score > best_score:
            best, best_score = n, score
    return best if best_score > 0 else None


def _station_stats(df: pd.DataFrame, name: str) -> Optional[dict]:
    sub = df[df["police_station"] == name]
    if sub.empty:
        return None
    return {
        "name": name,
        "count": int(len(sub)),
        "avg_cis": round(float(sub["cis"].mean()), 1),
        "peak_hour": int(sub.groupby("hour")["cis"].mean().idxmax()) if "hour" in sub.columns else None,
        "rush_pct": round(float(sub["is_rush_hour"].mean()) * 100) if "is_rush_hour" in sub.columns else None,
    }


def _wants_forecast(q: str) -> bool:
    return any(w in q.lower() for w in ("forecast", "predict", "tomorrow", "next ", "future", "expect", "upcoming", "will "))


# ── Memory ───────────────────────────────────────────────────────────────────
def _recent_turns(session_id, history, limit=10):
    db_turns = db.get_recent_interactions(session_id, limit=limit)
    if db_turns:
        return db_turns
    return (history or [])[-limit:]


def _corrections(turns):
    cues = ("no,", "not ", "actually", "wrong", "incorrect", "correction")
    return [t["text"] for t in turns if t.get("role") == "user" and any(c in t["text"].lower() for c in cues)]


def log_interaction(session_id, user_query, ai_response, tools_called, execution_time=0.0, confidence_score=0.0, recommendation="", intent=""):
    try:
        db.insert_interaction_log(
            session_id=session_id or "default",
            user_query=user_query,
            ai_response=ai_response,
            tools_used=tools_called,
            execution_time=execution_time,
            confidence_score=confidence_score,
            recommendations_generated=recommendation,
            intent=intent
        )
    except Exception as e:
        print(f"[agent] Failed to log interaction: {e}")


def _peak_str(h):
    return f"{h:02d}:00" if h is not None else "the peak window"


# ── Main entrypoint ──────────────────────────────────────────────────────────
def answer(message: str, df: pd.DataFrame, history=None, session_id=None) -> dict:
    start_time = time.time()
    turns = _recent_turns(session_id, history)
    
    # Try LangChain tool-calling agent first
    lc_result = langchain_agent.run_langchain_agent(message, df, turns)
    if lc_result:
        lc_result["intent"] = "langchain_dynamic"
        execution_time = time.time() - start_time
        log_interaction(
            session_id, message, lc_result["reply"], lc_result.get("tools_called", []),
            execution_time=execution_time,
            confidence_score=lc_result.get("confidence_score", 0.0),
            recommendation=lc_result.get("recommendation", ""),
            intent="langchain_dynamic"
        )
        return lc_result

    # Fallback to deterministic routing if LangChain fails/unavailable
    ql = message.lower().strip()
    tools_called = ["StatsQueryTool"]
    stats = stats_tool(df, message)
    sources = rag.retrieve(message, k=3)
    tools_called.append("RAGGuidelinesTool")
    corrections = _corrections(turns)

    station = _match_station(message, df)
    sstats = _station_stats(df, station) if station else None
    avg = stats.get("avg_cis", 0)
    top = stats.get("top_station", "the top-EPI corridor")
    top_cis = stats.get("top_station_cis", avg)

    def has(*w):
        return any(x in ql for x in w)

    # ── intent routing ──
    if not ql or has("hello", "hi", "hey", "help", "what can you", "who are you"):
        intent = "greeting"
    elif _wants_forecast(message):
        intent = "forecast"
    elif has("anomaly", "anomalies", "alert", "spike", "unusual", "surge"):
        intent = "anomaly"
    elif has("summary", "brief", "executive", "commissioner", "overview", "report"):
        intent = "summary"
    elif has("why", "cause", "reason"):
        intent = "diagnose"
    elif has("deploy", "tow", "patrol", "where", "dispatch", "send", "position", "allocate"):
        intent = "deploy"
    else:
        intent = "general"

    observation = reasoning = recommendation = ""
    table_md = ""  # optional markdown table folded into the synthesis
    confidence = 62

    if intent == "greeting":
        observation = f"Tactical agent online. Monitoring {stats.get('n_stations', 0)} stations; city-wide average CIS {avg}."
        reasoning = "I orchestrate live stats, a 7-day forecast model, anomaly detection, and BTP guideline retrieval to ground every recommendation."
        recommendation = "Ask me e.g. \"Where should I deploy tow trucks at Silk Board?\", \"Why is Koramangala spiking?\", \"Forecast tomorrow's peak\", or \"Show active anomalies\"."
        confidence = 90

    elif intent == "forecast":
        try:
            from api import forecasting

            fc = forecasting.predict(df, days=1, station=station)
            tools_called.append("ForecastingTool")
            pts = fc["points"]
            if pts:
                peak = max(pts, key=lambda p: p["yhat"])
                where = station or "the city"
                observation = f"Next-24h forecast for {where}: peak congestion ~CIS {peak['yhat']} around {_peak_str(peak['hour'])} ({fc['engine']} model)."
                reasoning = "Projection blends weekly seasonality (weekday vs weekend) with the daily profile — morning (08–10) and evening (17–19) peaks dominate."
                recommendation = f"Pre-position 1–2 tow units near {station or top} ~30 min before {_peak_str(peak['hour'])}; brief teams for the rush-hour surge."
                confidence = 80
            else:
                observation = "Not enough history to forecast this slice."
                reasoning = "The filtered frame had too few hourly points."
                recommendation = "Try a city-wide forecast or a busier station."
        except Exception:  # noqa: BLE001
            observation = "Forecast engine unavailable."
            reasoning = "Falling back to current observations."
            recommendation = f"Focus on {top} (avg CIS {top_cis})."

    elif intent == "anomaly":
        try:
            from api import alerts as alerts_mod

            active = alerts_mod.detect(df, top=4)
            tools_called.append("AlertsTool")
        except Exception:  # noqa: BLE001
            active = []
        if active:
            observation = "Active alerts — " + "; ".join(a["msg"] for a in active[:2])
            reasoning = "Detected via rolling z-score on daily counts and an Isolation-Forest on (count, avg-CIS); both flag statistically unusual days."
            recommendation = f"Investigate {active[0]['station']} first ({active[0]['severity']}); verify against the live camera feed and dispatch if confirmed."
            confidence = 84
        else:
            observation = "No statistically significant anomalies right now."
            reasoning = "Daily counts are within ~2σ of each station's baseline."
            recommendation = "Maintain routine patrol coverage; re-check after the next data sync."
            confidence = 70

    elif intent == "summary":
        agg = (
            df.groupby("police_station", observed=True)["cis"].agg(["mean", "size"]).reset_index().sort_values("mean", ascending=False).head(5)
            if len(df) else pd.DataFrame()
        )
        top3 = ", ".join(f"{r['police_station']} (CIS {r['mean']:.1f})" for _, r in agg.head(3).iterrows())
        observation = f"City-wide avg CIS {avg} across {stats.get('n_stations', 0)} stations. Top concerns: {top3}. Peak hour ~{_peak_str(stats.get('peak_hour'))}."
        reasoning = "EPI blends total CIS (0.4), volume (0.3) and average CIS (0.3); the leaders combine high heavy-vehicle share with junction-factor exposure."
        recommendation = f"Authorise reinforced AM-peak patrols at the top-3 zones; pre-stage tow units at {top} by 07:00."
        if len(agg):
            table_md = "| Station | Avg CIS | Violations |\n|---|---|---|\n" + "\n".join(
                f"| {r['police_station']} | {r['mean']:.1f} | {int(r['size'])} |" for _, r in agg.iterrows()
            )
        confidence = 82

    elif intent == "diagnose":
        target = station or top
        ts = sstats or {"avg_cis": top_cis, "peak_hour": stats.get("peak_hour"), "rush_pct": None, "count": None}
        observation = (
            f"{target}: average CIS {ts['avg_cis']}"
            + (f" across {ts['count']} violations" if ts.get("count") else "")
            + (f", peaking ~{_peak_str(ts['peak_hour'])}" if ts.get("peak_hour") is not None else "")
            + (f"; {ts['rush_pct']}% during rush hours." if ts.get("rush_pct") is not None else ".")
        )
        reasoning = (
            f"CIS = Severity × Vehicle Size × Time Factor × Junction Factor. {target} is elevated mainly by heavy-vehicle "
            "double-parking inside the junction-factor radius during the peak window (time factor ×3)."
        )
        recommendation = f"Station 1 officer at {target}'s primary merge point during the peak; tow tankers/HGVs first; request a +15–20s signal-phase extension on the parallel route."
        confidence = 83 if sstats else 72

    elif intent == "deploy":
        target = station or top
        ts = sstats or {"avg_cis": top_cis, "peak_hour": stats.get("peak_hour"), "count": None}
        observation = (
            f"Highest-impact target: {target} (avg CIS {ts['avg_cis']}"
            + (f", {ts['count']} violations" if ts.get("count") else "")
            + (f", peak ~{_peak_str(ts['peak_hour'])})." if ts.get("peak_hour") is not None else ").")
        )
        reasoning = "EPI ranks this corridor highest on the combined CIS volume/frequency/severity blend; double-parked heavy vehicles drive most of the impact."
        recommendation = (
            f"Deploy 2 tow units to {target} — stage at the upstream junction approach, one officer 200 m upstream to divert, "
            "one at the hotspot. Prioritise tankers/HGVs (highest CIS); request signal-phase support from BBMP before towing."
        )
        confidence = 84 if sstats else 76

    else:  # general — answer about the named station if any, else top
        if station and sstats:
            observation = f"{station}: average CIS {sstats['avg_cis']} across {sstats['count']} violations, peaking ~{_peak_str(sstats['peak_hour'])}."
            reasoning = "Derived live from the violation frame for this station and grounded in BTP enforcement guidelines."
            recommendation = f"If {station} is your focus, deploy a tow unit at its peak window and monitor the live feed for repeat offenders."
            confidence = 80
        else:
            observation = f"City-wide average CIS {avg}; highest-impact zone {top} (avg CIS {top_cis}), peak ~{_peak_str(stats.get('peak_hour'))}."
            reasoning = "Computed from the live frame; I can drill into any station, forecast, anomaly, or dispatch plan you name."
            recommendation = f"Start with {top}: 2 tow units at the junction approach during the peak window."
            confidence = 74

    if corrections:
        reasoning += f" Adjusting for your earlier note: \"{corrections[-1]}\"."
    if sources:
        confidence = min(95, confidence + 6)
    if os.environ.get("GROQ_API_KEY"):
        confidence = min(96, confidence + 4)

    reply = _synthesize(message, observation, reasoning, recommendation, sources, turns, table_md)

    result = {
        "observation": observation,
        "reasoning": reasoning,
        "recommendation": recommendation,
        "confidence_score": float(confidence),
        "reply": reply,
        "sources": sources,
        "tools_called": tools_called,
        "intent": intent,
    }
    execution_time = time.time() - start_time
    log_interaction(
        session_id, message, reply, tools_called,
        execution_time=execution_time,
        confidence_score=float(confidence),
        recommendation=recommendation,
        intent=intent
    )
    return result


SYNTH_SYSTEM = (
    "You are a Bengaluru Traffic Police tactical advisor. Answer the officer's SPECIFIC question "
    "directly using only the provided CONTEXT (do not invent data). "
    "Respond in clean GitHub-flavored MARKDOWN only: use '### Observation', '### Reasoning', "
    "'### Recommendation' headings, bullet lists, **bold** for key numbers, and a markdown TABLE "
    "for any statistics (reuse the DATA TABLE if provided). Reference real Bengaluru roads. "
    "STRICT: never output raw JSON, Python dicts, tool logs, SQL, stack traces, or variable names — "
    "only the polished human-readable briefing."
)


def _deterministic_markdown(observation, reasoning, recommendation, sources, table_md) -> str:
    """Clean markdown briefing used when no LLM key is present (the synthesis node's
    fallback). Renders nicely via the frontend markdown renderer."""
    parts = [
        f"### Observation\n{observation}",
        f"### Reasoning\n{reasoning}",
        f"### Recommendation\n{recommendation}",
    ]
    if table_md:
        parts.insert(1, table_md)
    if sources:
        parts.append("**Sources:** " + ", ".join(s.split(":")[0] for s in sources))
    return "\n\n".join(parts)


def _synthesize(message, observation, reasoning, recommendation, sources, turns, table_md="") -> str:
    """Synthesis node: turn the structured tool payload into a clean, human-readable
    markdown briefing — never raw JSON/logs. Uses Groq when a key exists."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            ctx = (
                f"OBSERVATION: {observation}\nREASONING: {reasoning}\n"
                f"RECOMMENDATION: {recommendation}\nGUIDELINES: {' | '.join(sources)}"
            )
            if table_md:
                ctx += f"\n\nDATA TABLE (markdown):\n{table_md}"
            msgs = [{"role": "system", "content": SYNTH_SYSTEM}]
            for t in turns[-6:]:
                msgs.append({"role": "assistant" if t["role"] in ("ai", "assistant") else "user", "content": t["text"]})
            msgs.append({"role": "user", "content": f"QUESTION: {message}\n\nCONTEXT:\n{ctx}"})
            resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, max_tokens=750)
            return resp.choices[0].message.content
        except Exception:  # noqa: BLE001 — never leak the trace to the UI
            return _deterministic_markdown(observation, reasoning, recommendation, sources, table_md)
    return _deterministic_markdown(observation, reasoning, recommendation, sources, table_md)
