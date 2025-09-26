#services/api/main.py

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from sqlalchemy import text
from sqlalchemy.orm import Session

from di import get_sync_session
from routers.icesat import router as icesat_router
from routers.compare import router as compare_router
from routers.dem import router as dem_router

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("geoapi")

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting...")
    yield
    logger.info("API stopping...")

app = FastAPI(
    title="GeoHydroAI API",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    # docs_url="/docs",  # Remove /api
    redoc_url="/redoc", # Remove /api
    # openapi_url="/openapi.json", # Remove /api
)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(icesat_router)
app.include_router(compare_router)
app.include_router(dem_router)

# Health
@app.get("/healthz", status_code=status.HTTP_200_OK)
def health():
    return {"status": "ok"}

@app.get("/dbhealth", status_code=status.HTTP_200_OK)
def dbhealth(db: Session = Depends(get_sync_session)):
    db.execute(text("SELECT 1"))
    return {"db": "ok"}