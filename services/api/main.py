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

from services.api.di import get_sync_session
from services.api.routers.icesat import router as icesat_router
from services.api.routers.compare import router as compare_router
from services.api.routers.dem import router as dem_router

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
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
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
@app.get("/health", status_code=status.HTTP_200_OK)
def health(db: Session = Depends(get_sync_session)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}
