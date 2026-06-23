from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

# Use DATABASE_URL from environment or fallback to local SQLite database
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./parkwatch.db")

# In Supabase/Render, the DATABASE_URL might be postgres:// instead of postgresql://
# SQLAlchemy 1.4+ requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_query = Column(String, nullable=False)
    ai_response = Column(String, nullable=False)
    tools_used = Column(JSON, nullable=True)
    execution_time = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    recommendations_generated = Column(String, nullable=True)
    intent = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)


class DatasetRegistry(Base):
    __tablename__ = "dataset_registry"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    row_count = Column(Integer)
    schema_json = Column(JSON, nullable=True)
    is_active = Column(Integer, default=0)


# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def insert_interaction_log(
    session_id: str,
    user_query: str,
    ai_response: str,
    tools_used: List[str],
    execution_time: float,
    confidence_score: float,
    recommendations_generated: str,
    intent: str,
    metadata_json: Dict[str, Any] = None
) -> dict:
    """Helper to insert an interaction log synchronously."""
    db = SessionLocal()
    try:
        log = InteractionLog(
            session_id=session_id,
            user_query=user_query,
            ai_response=ai_response,
            tools_used=tools_used,
            execution_time=execution_time,
            confidence_score=confidence_score,
            recommendations_generated=recommendations_generated,
            intent=intent,
            metadata_json=metadata_json or {}
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return {
            "id": log.id,
            "session_id": log.session_id,
            "timestamp": log.timestamp.isoformat() + "Z",
            "user_query": log.user_query,
            "ai_response": log.ai_response,
            "tools_used": log.tools_used,
            "confidence_score": log.confidence_score,
            "intent": log.intent
        }
    except Exception as e:
        db.rollback()
        print(f"[db] Error inserting interaction log: {e}")
        return None
    finally:
        db.close()


def get_recent_interactions(session_id: str = None, limit: int = 10) -> List[dict]:
    """Helper to fetch recent interactions for memory layer."""
    db = SessionLocal()
    try:
        query = db.query(InteractionLog)
        if session_id:
            query = query.filter(InteractionLog.session_id == session_id)
        logs = query.order_by(InteractionLog.timestamp.desc()).limit(limit).all()
        # Return in ascending order for context injection
        return [{"role": "user", "text": log.user_query, "response": log.ai_response, "intent": log.intent, "timestamp": log.timestamp.isoformat()} for log in reversed(logs)]
    except Exception as e:
        print(f"[db] Error fetching recent interactions: {e}")
        return []
    finally:
        db.close()
