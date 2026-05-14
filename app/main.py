from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.segments import router as segments_router, events_router, membership_router

app = FastAPI(
    title="Audience Segmentation Service",
    description=(
        "ML-powered audience segmentation for programmatic advertising. "
        "Build rule-based and lookalike segments, ingest user events, "
        "and evaluate real-time segment membership for targeting."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(segments_router, prefix="/v1")
app.include_router(events_router, prefix="/v1")
app.include_router(membership_router, prefix="/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
