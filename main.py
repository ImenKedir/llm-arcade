from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our database modules
from app.db.database import Base, engine

# Import our API routers
from app.api.index import router as main_router
from app.api.blackjack import router as blackjack_router
from app.api.taxi import router as taxi_router
from app.api.cliff_walking import router as cliff_walking_router
from app.api.frozen_lake import router as frozen_lake_router

# Create FastAPI app
app = FastAPI(
    title="LLM Arcade",
    description="A collection of reinforcement learning environments",
    version="0.1.0"
)

# Ensure static directory exists
static_dir = Path("./static")
static_dir.mkdir(exist_ok=True)

# Set up static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(main_router)  # This includes the index route
app.include_router(blackjack_router)
app.include_router(taxi_router)
app.include_router(cliff_walking_router)
app.include_router(frozen_lake_router)

# Startup event to initialize the database
@app.on_event("startup")
async def startup_event():
    try:
        # Database is already initialized when the database module is imported
        from app.db.database import Base, engine
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created if they didn't exist.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        logger.warning("Application will continue startup but database functionality may be limited")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
