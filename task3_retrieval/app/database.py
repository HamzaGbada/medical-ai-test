"""
Database setup for PGVector with SQLAlchemy.

Manages engine creation, session factory, and pgvector extension initialization.
"""

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from task3_retrieval.app.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

engine = create_engine(settings.database_url, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Initialize the database: create pgvector extension and tables."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("PGVector extension enabled")

    # Import models so they are registered with Base
    from task3_retrieval.app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


def get_db():
    """Dependency: yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
