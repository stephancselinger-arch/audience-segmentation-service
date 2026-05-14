"""
Core segmentation engine.

Supports:
  - Rule-based matching (behavioral/demographic/contextual segments)
  - K-means clustering for lookalike/behavioral ML segments
  - User event ingestion and membership evaluation
"""

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional
import random
import math

from app.models.segment import (
    SegmentCreate, SegmentResponse, SegmentStatus, SegmentType,
    SegmentRule, RuleOperator, UserEvent, SegmentMembershipResponse,
    new_segment_id,
)
from app.services.feature_engineering import build_feature_vector, normalize_features, FeatureVector


# In-memory stores — replace with PostgreSQL + Redis in production
_segments: dict[str, dict] = {}
_events: dict[str, list[UserEvent]] = defaultdict(list)  # user_id -> events
_memberships: dict[str, set[str]] = defaultdict(set)     # user_id -> segment_ids


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _build_response(record: dict) -> SegmentResponse:
    return SegmentResponse(**{k: v for k, v in record.items() if k != "raw"})


# ── Segment CRUD ──────────────────────────────────────────────────────────────

def create_segment(payload: SegmentCreate) -> SegmentResponse:
    seg_id = new_segment_id()
    now = _now()
    record = {
        "id": seg_id,
        "name": payload.name,
        "description": payload.description,
        "segment_type": payload.segment_type,
        "advertiser_id": payload.advertiser_id,
        "status": SegmentStatus.DRAFT,
        "definition": payload.definition,
        "user_count": 0,
        "ttl_days": payload.ttl_days,
        "created_at": now,
        "updated_at": now,
    }
    _segments[seg_id] = record
    return _build_response(record)


def get_segment(segment_id: str) -> Optional[SegmentResponse]:
    record = _segments.get(segment_id)
    return _build_response(record) if record else None


def list_segments(
    advertiser_id: Optional[str] = None,
    status: Optional[SegmentStatus] = None,
    segment_type: Optional[SegmentType] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[SegmentResponse]:
    results = list(_segments.values())
    if advertiser_id:
        results = [r for r in results if r["advertiser_id"] == advertiser_id]
    if status:
        results = [r for r in results if r["status"] == status]
    if segment_type:
        results = [r for r in results if r["segment_type"] == segment_type]
    return [_build_response(r) for r in results[offset : offset + limit]]


def update_segment_status(segment_id: str, status: SegmentStatus) -> Optional[SegmentResponse]:
    record = _segments.get(segment_id)
    if not record:
        return None
    record["status"] = status
    record["updated_at"] = _now()
    return _build_response(record)


def delete_segment(segment_id: str) -> bool:
    if segment_id not in _segments:
        return False
    _segments.pop(segment_id)
    for uid in list(_memberships.keys()):
        _memberships[uid].discard(segment_id)
    return True


# ── Event Ingestion ───────────────────────────────────────────────────────────

def ingest_events(events: list[UserEvent]) -> int:
    for ev in events:
        _events[ev.user_id].append(ev)
    return len(events)


def get_user_events(user_id: str, lookback_days: int = 30) -> list[UserEvent]:
    cutoff = _now() - timedelta(days=lookback_days)
    return [
        ev for ev in _events.get(user_id, [])
        if ev.timestamp.replace(tzinfo=timezone.utc) >= cutoff
        if ev.timestamp.tzinfo else ev.timestamp >= cutoff.replace(tzinfo=None)
    ]


# ── Rule Evaluation ───────────────────────────────────────────────────────────

def _eval_rule(rule: SegmentRule, events: list[UserEvent]) -> bool:
    values: list = []
    for ev in events:
        raw = getattr(ev, rule.field, None)
        if raw is not None:
            values.append(raw)

    if not values:
        return False

    op = rule.operator
    target = rule.value

    if op == RuleOperator.EQUALS:
        return any(v == target for v in values)
    if op == RuleOperator.NOT_EQUALS:
        return all(v != target for v in values)
    if op == RuleOperator.GREATER_THAN:
        return any(v > target for v in values)
    if op == RuleOperator.LESS_THAN:
        return any(v < target for v in values)
    if op == RuleOperator.IN:
        return any(v in target for v in values)
    if op == RuleOperator.NOT_IN:
        return all(v not in target for v in values)
    if op == RuleOperator.CONTAINS:
        return any(target in str(v) for v in values)
    return False


def _matches_segment(user_id: str, record: dict) -> bool:
    defn = record["definition"]
    events = get_user_events(user_id, lookback_days=defn.lookback_days)

    if len(events) < defn.min_events:
        return False
    if not defn.rules:
        return True

    results = [_eval_rule(rule, events) for rule in defn.rules]
    return all(results) if defn.match_all else any(results)


# ── ML Segmentation (K-Means) ─────────────────────────────────────────────────

def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _kmeans(vectors: list[FeatureVector], k: int, max_iter: int = 50) -> list[int]:
    """Minimal k-means; returns cluster assignment per vector."""
    if len(vectors) <= k:
        return list(range(len(vectors)))

    centroids = [list(v.features) for v in random.sample(vectors, k)]

    assignments = [0] * len(vectors)
    for _ in range(max_iter):
        new_assignments = [
            min(range(k), key=lambda ci: _euclidean(v.features, centroids[ci]))
            for v in vectors
        ]
        if new_assignments == assignments:
            break
        assignments = new_assignments

        for ci in range(k):
            members = [vectors[i].features for i, a in enumerate(assignments) if a == ci]
            if members:
                centroids[ci] = [
                    sum(col) / len(col) for col in zip(*members)
                ]

    return assignments


def build_lookalike_segment(segment_id: str, seed_user_ids: list[str], expansion_factor: float = 3.0) -> int:
    """Expand a seed audience using k-means clustering to find similar users."""
    all_user_ids = list(_events.keys())
    if not all_user_ids:
        return 0

    feature_vecs = [build_feature_vector(uid, _events[uid]) for uid in all_user_ids]
    normalized = normalize_features(feature_vecs)

    seed_set = set(seed_user_ids)
    target_size = int(len(seed_set) * expansion_factor)
    k = max(2, min(10, len(all_user_ids) // 5))

    assignments = _kmeans(normalized, k=k)

    uid_to_cluster = {fv.user_id: assignments[i] for i, fv in enumerate(normalized)}
    seed_clusters: set[int] = set()
    for uid in seed_user_ids:
        if uid in uid_to_cluster:
            seed_clusters.add(uid_to_cluster[uid])

    candidates = [
        uid for uid in all_user_ids
        if uid not in seed_set and uid_to_cluster.get(uid) in seed_clusters
    ][:target_size]

    expanded = list(seed_set) + candidates
    for uid in expanded:
        _memberships[uid].add(segment_id)

    record = _segments.get(segment_id)
    if record:
        record["user_count"] = len(expanded)
        record["status"] = SegmentStatus.ACTIVE
        record["updated_at"] = _now()

    return len(expanded)


# ── Membership Evaluation ─────────────────────────────────────────────────────

def evaluate_user(user_id: str) -> SegmentMembershipResponse:
    matched: set[str] = set()
    for seg_id, record in _segments.items():
        if record["status"] != SegmentStatus.ACTIVE:
            continue
        if record["segment_type"] == SegmentType.LOOKALIKE:
            # Lookalike membership is pre-computed via build_lookalike_segment
            if seg_id in _memberships.get(user_id, set()):
                matched.add(seg_id)
        else:
            if _matches_segment(user_id, record):
                matched.add(seg_id)
                _memberships[user_id].add(seg_id)

    return SegmentMembershipResponse(
        user_id=user_id,
        segment_ids=sorted(matched),
        evaluated_at=_now(),
    )


def get_segment_users(segment_id: str) -> list[str]:
    return [uid for uid, segs in _memberships.items() if segment_id in segs]
