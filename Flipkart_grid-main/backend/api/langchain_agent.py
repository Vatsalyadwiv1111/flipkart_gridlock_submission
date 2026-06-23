import os
from typing import List, Dict, Any
import pandas as pd

try:
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    pass

from api import forecasting
from api import alerts as alerts_mod
from api import rag
from api import ml

# Globals to hold the dataframe in the module scope for tools
_DF: pd.DataFrame = None

def set_df(df: pd.DataFrame):
    global _DF
    _DF = df


@tool
def sql_stats_query_tool(query: str) -> str:
    """Useful to get live statistics from the current dataframe (average CIS, top stations, peak hours). Provide a description of what you want to aggregate."""
    if _DF is None or _DF.empty:
        return "No data available."
    
    # Very simple SQL-like aggregation
    q = query.lower()
    res = []
    res.append(f"Total rows: {len(_DF)}")
    res.append(f"City-wide Average CIS: {round(float(_DF['cis'].mean()), 1)}")
    
    epi = _DF.groupby("police_station", observed=True)["cis"].agg(["mean", "size"]).reset_index()
    epi = epi.sort_values("mean", ascending=False)
    res.append(f"Top station by CIS: {epi.iloc[0]['police_station']} ({round(float(epi.iloc[0]['mean']), 1)})")
    
    if "hour" in _DF.columns:
        peak = int(_DF.groupby("hour")["cis"].mean().idxmax())
        res.append(f"Peak congestion hour: {peak:02d}:00")
        
    return " | ".join(res)

@tool
def forecasting_tool(days: int = 1, station: str = None) -> str:
    """Useful to forecast congestion or peak hours for the next few days. Specify 'days' and optionally a 'station' name."""
    if _DF is None:
        return "No data available."
    try:
        fc = forecasting.predict(_DF, days=max(1, days), station=station)
        pts = fc.get("points", [])
        if not pts:
            return "Not enough history to forecast."
        peak = max(pts, key=lambda p: p["yhat"])
        return f"Forecast ({fc['engine']}): Peak congestion ~CIS {peak['yhat']} around {peak['hour']:02d}:00."
    except Exception as e:
        return f"Forecasting failed: {e}"

@tool
def anomaly_detection_tool() -> str:
    """Useful to detect active anomalies, surges, or unusual spikes in the current data."""
    if _DF is None:
        return "No data available."
    try:
        active = alerts_mod.detect(_DF, top=4)
        if active:
            return "Active alerts: " + "; ".join(a["msg"] + f" at {a['station']}" for a in active[:3])
        return "No statistically significant anomalies detected right now."
    except Exception as e:
        return f"Anomaly detection failed: {e}"

@tool
def btp_guidelines_tool(query: str) -> str:
    """Useful to search Bengaluru Traffic Police (BTP) guidelines for enforcement rules and historic advisories."""
    sources = rag.retrieve(query, k=3)
    if sources:
        return " | ".join(sources)
    return "No specific guidelines found."

@tool
def trend_mining_tool() -> str:
    """Useful to analyze long-term trends, economic loss, or emerging hotspots."""
    if _DF is None:
        return "No data available."
    try:
        hotspots = ml.emerging_hotspots(_DF)
        res = "Emerging hotspots: " + ", ".join([h["station"] for h in hotspots[:3]])
        return res
    except Exception as e:
        return f"Trend mining failed: {e}"


def get_agent_executor():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
        
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.1)
    tools = [sql_stats_query_tool, forecasting_tool, anomaly_detection_tool, btp_guidelines_tool, trend_mining_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Tactical AI Commander for the Bengaluru Traffic Police.
You MUST format your final answer strictly using the following markdown structure:

### Observation
[What the data says, using your tools]

### Reasoning
[Why this is happening, root cause analysis]

### Recommendation
[Actionable dispatch or patrol recommendations]

**Confidence Score:** [Number from 0 to 100]%

Use your tools to gather data before answering. Do not invent data.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def run_langchain_agent(message: str, df: pd.DataFrame, history: List[dict]):
    set_df(df)
    executor = get_agent_executor()
    if not executor:
        return None
        
    chat_history = []
    for h in history:
        if h.get("role") == "user":
            chat_history.append(HumanMessage(content=h.get("text", "")))
        else:
            chat_history.append(AIMessage(content=h.get("response") or h.get("text", "")))
            
    try:
        response = executor.invoke({
            "input": message,
            "chat_history": chat_history
        })
        
        reply = response["output"]
        
        # Parse output heuristically to extract observation, reasoning, recommendation, confidence
        observation = ""
        reasoning = ""
        recommendation = ""
        confidence = 85.0
        
        import re
        obs_match = re.search(r"### Observation\n(.*?)(?=### Reasoning|$)", reply, re.DOTALL)
        if obs_match: observation = obs_match.group(1).strip()
        
        reas_match = re.search(r"### Reasoning\n(.*?)(?=### Recommendation|$)", reply, re.DOTALL)
        if reas_match: reasoning = reas_match.group(1).strip()
        
        rec_match = re.search(r"### Recommendation\n(.*?)(?=\*\*Confidence Score|\Z)", reply, re.DOTALL)
        if rec_match: recommendation = rec_match.group(1).strip()
        
        conf_match = re.search(r"\*\*Confidence Score:\*\*\s*(\d+)", reply)
        if conf_match: confidence = float(conf_match.group(1))
        
        return {
            "observation": observation or "See reply.",
            "reasoning": reasoning or "Generated via LangChain agent.",
            "recommendation": recommendation or "Follow AI advice.",
            "confidence_score": confidence,
            "reply": reply,
            "sources": [],
            "tools_called": ["LangChainAgent"],
            "intent": "langchain_dynamic"
        }
    except Exception as e:
        print(f"[langchain_agent] Error: {e}")
        return None
