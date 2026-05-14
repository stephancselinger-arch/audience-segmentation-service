from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.models.segment import (
    SegmentCreate, SegmentResponse, SegmentStatus, SegmentType,
    UserEvent, SegmentMembershipResponse,
)
from app.services import segmentation

router = APIRouter(prefix="/segments", tags=["Segments"])
events_router = APIRouter(prefix="/events", tags=["Events"])
membership_router = APIRouter(prefix="/membership", tags=["Membership"])


# ── Segments ──────────────────────────────────────────────────────────────────

@router.post("/", response_model=SegmentResponse, status_code=201)
def create_segment(payload: SegmentCreate):
    return segmentation.create_segment(payload)


@router.get("/", response_model=list[SegmentResponse])
def list_segments(
    advertiser_id: Optional[str] = Query(None),
    status: Optional[SegmentStatus] = Query(None),
    segment_type: Optional[SegmentType] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
):
    return segmentation.list_segments(
        advertiser_id=advertiser_id,
        status=status,
        segment_type=segment_type,
        limit=limit,
        offset=offset,
    )


@router.get("/{segment_id}", response_model=SegmentResponse)
def get_segment(segment_id: str):
    seg = segmentation.get_segment(segment_id)
    if not seg:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    return seg


@router.patch("/{segment_id}/status", response_model=SegmentResponse)
def update_status(segment_id: str, status: SegmentStatus):
    result = segmentation.update_segment_status(segment_id, status)
    if not result:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    return result


@router.get("/{segment_id}/users")
def get_segment_users(segment_id: str) -> dict:
    if not segmentation.get_segment(segment_id):
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    users = segmentation.get_segment_users(segment_id)
    return {"segment_id": segment_id, "user_count": len(users), "user_ids": users}


@router.delete("/{segment_id}", status_code=204)
def delete_segment(segment_id: str):
    if not segmentation.delete_segment(segment_id):
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")


class LookalikeRequest(BaseModel):
    seed_user_ids: list[str]
    expansion_factor: float = 3.0


@router.post("/{segment_id}/lookalike")
def build_lookalike(segment_id: str, body: LookalikeRequest) -> dict:
    seg = segmentation.get_segment(segment_id)
    if not seg:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    if seg.segment_type != SegmentType.LOOKALIKE:
        raise HTTPException(status_code=422, detail="Segment must be of type 'lookalike'")

    count = segmentation.build_lookalike_segment(
        segment_id,
        seed_user_ids=body.seed_user_ids,
        expansion_factor=body.expansion_factor,
    )
    return {"segment_id": segment_id, "user_count": count}


# ── Events ────────────────────────────────────────────────────────────────────

@events_router.post("/ingest")
def ingest_events(events: list[UserEvent]) -> dict:
    count = segmentation.ingest_events(events)
    return {"ingested": count}


# ── Membership ────────────────────────────────────────────────────────────────

@membership_router.get("/{user_id}", response_model=SegmentMembershipResponse)
def evaluate_user(user_id: str):
    return segmentation.evaluate_user(user_id)
