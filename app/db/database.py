import os
import logging
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create data directory if it doesn't exist
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# SQLite database file path
SQLITE_DB_FILE = data_dir / "counter.db"
SQLITE_URL = f"sqlite:///{SQLITE_DB_FILE}"

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Function to set up SQLite connection
def setup_sqlite_engine():
    logger.info(f"Setting up SQLite database at {SQLITE_DB_FILE}")
    sqlite_engine = create_engine(
        SQLITE_URL, 
        connect_args={"check_same_thread": False}
    )
    
    # Configure SQLite for better concurrency
    @event.listens_for(sqlite_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    return sqlite_engine

# Determine which database to use
use_sqlite = False

if not DATABASE_URL:
    logger.warning("No DATABASE_URL found. Using SQLite database.")
    use_sqlite = True
elif "placeholder" in DATABASE_URL:
    logger.warning(f"DATABASE_URL contains placeholder values. Using SQLite database instead.")
    use_sqlite = True
else:
    # Convert postgres:// to postgresql:// for SQLAlchemy
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")
    logger.info(f"Using database connection: {DATABASE_URL.split('@')[0]}@****")

# Try to connect to the primary database, fall back to SQLite if it fails
try:
    if use_sqlite:
        engine = setup_sqlite_engine()
    else:
        # Create SQLAlchemy engine for primary database
        engine = create_engine(DATABASE_URL)
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
    logger.info("Database engine created successfully and connection verified")
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database session factory created successfully")
except Exception as e:
    logger.error(f"Error creating primary database connection: {str(e)}", exc_info=True)
    
    # Fall back to SQLite
    logger.warning("Falling back to SQLite database due to connection error")
    engine = setup_sqlite_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("SQLite fallback database configured successfully")

# Create base class for models
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()
