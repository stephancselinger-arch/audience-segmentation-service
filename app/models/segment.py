from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, field_validator
from datetime import datetime
import uuid


class SegmentStatus(str, Enum):
    DRAFT = "draft"
    BUILDING = "building"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class SegmentType(str, Enum):
    BEHAVIORAL = "behavioral"      # based on browsing/engagement signals
    CONTEXTUAL = "contextual"      # based on content categories
    DEMOGRAPHIC = "demographic"    # age, gender, geo
    LOOKALIKE = "lookalike"        # ML expansion from seed audience
    CUSTOM = "custom"              # rule-based / uploaded list


class RuleOperator(str, Enum):
    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


class SegmentRule(BaseModel):
    field: str          # e.g. "age", "category", "event_type"
    operator: RuleOperator
    value: Any


class SegmentDefinition(BaseModel):
    rules: list[SegmentRule] = []
    match_all: bool = True    # True = AND logic, False = OR logic
    lookback_days: int = 30
    min_events: int = 1


class SegmentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    segment_type: SegmentType
    advertiser_id: str
    definition: SegmentDefinition = SegmentDefinition()
    ttl_days: int = 90

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Segment name cannot be empty")
        return v.strip()


class SegmentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    segment_type: SegmentType
    advertiser_id: str
    status: SegmentStatus
    definition: SegmentDefinition
    user_count: int
    ttl_days: int
    created_at: datetime
    updated_at: datetime


class UserEvent(BaseModel):
    user_id: str
    event_type: str       # e.g. "page_view", "add_to_cart", "purchase", "video_play"
    category: Optional[str] = None     # IAB content category
    url: Optional[str] = None
    value: Optional[float] = None      # e.g. purchase value
    age_bucket: Optional[str] = None   # "18-24", "25-34", etc.
    gender: Optional[str] = None
    geo_country: Optional[str] = None
    geo_region: Optional[str] = None
    device_type: Optional[str] = None  # "desktop", "mobile", "tablet", "ctv"
    timestamp: datetime


class SegmentMembershipResponse(BaseModel):
    user_id: str
    segment_ids: list[str]
    evaluated_at: datetime


def new_segment_id() -> str:
    return f"seg_{uuid.uuid4().hex[:16]}"
