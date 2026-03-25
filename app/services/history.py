from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from config import POSTGRES_URL
from sqlalchemy.engine import URL
Base = declarative_base()


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    question    = Column(Text, nullable=False)
    answer      = Column(Text, nullable=False)
    search_type = Column(String(20), default="vector")  # vector или hybrid
    created_at  = Column(DateTime, default=datetime.utcnow)


connection_url = URL.create(
    drivername="postgresql+psycopg",
    username="raguser",
    password="ragpassword",
    host="localhost",
    port=5432,
    database="ragdb"
)

engine = create_engine(connection_url)
#engine = create_engine(POSTGRES_URL)
#engine = create_engine(
#    "postgresql+psycopg2://raguser:ragpassword@localhost:5432/ragdb"
#)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    # Создаём таблицы если не существуют
    Base.metadata.create_all(engine)


def save_to_history(question: str, answer: str, search_type: str = "vector"):
    session = SessionLocal()
    try:
        record = ChatHistory(
            question=question,
            answer=answer,
            search_type=search_type
        )
        session.add(record)
        session.commit()
    finally:
        session.close()


def get_history(limit: int = 10):
    session = SessionLocal()
    try:
        records = (
            session.query(ChatHistory)
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id":          r.id,
                "question":    r.question,
                "answer":      r.answer,
                "search_type": r.search_type,
                "created_at":  r.created_at.isoformat()
            }
            for r in records
        ]
    finally:
        session.close()