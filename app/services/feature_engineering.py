"""
Converts raw user events into numeric feature vectors for ML segmentation.

Feature space:
  - Event frequency by type (page_view, add_to_cart, purchase, video_play)
  - Category affinity scores (normalized count per IAB category)
  - Recency score (exponential decay, lambda=0.1 per day)
  - Device type one-hot (desktop, mobile, tablet, ctv)
  - Purchase value (sum, capped at 99th percentile)
"""

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import NamedTuple

from app.models.segment import UserEvent


EVENT_TYPES = ["page_view", "add_to_cart", "purchase", "video_play", "search"]
DEVICE_TYPES = ["desktop", "mobile", "tablet", "ctv"]
RECENCY_LAMBDA = 0.1   # decay rate per day — half-life ~7 days


class FeatureVector(NamedTuple):
    user_id: str
    features: list[float]
    feature_names: list[str]


def _days_ago(ts: datetime) -> float:
    now = datetime.now(timezone.utc)
    ts_utc = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    delta = now - ts_utc
    return max(0.0, delta.total_seconds() / 86400)


def build_feature_vector(user_id: str, events: list[UserEvent]) -> FeatureVector:
    if not events:
        n_features = len(EVENT_TYPES) + len(DEVICE_TYPES) + 3  # +recency, purchase_value, category_entropy
        return FeatureVector(user_id=user_id, features=[0.0] * n_features, feature_names=_feature_names())

    event_counts: dict[str, int] = defaultdict(int)
    device_counts: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)
    total_purchase_value = 0.0
    recency_score = 0.0

    for ev in events:
        event_counts[ev.event_type] += 1
        if ev.device_type:
            device_counts[ev.device_type.lower()] += 1
        if ev.category:
            category_counts[ev.category] += 1
        if ev.event_type == "purchase" and ev.value:
            total_purchase_value += ev.value

        days = _days_ago(ev.timestamp)
        recency_score += math.exp(-RECENCY_LAMBDA * days)

    event_features = [float(event_counts.get(et, 0)) for et in EVENT_TYPES]

    device_features = [float(device_counts.get(dt, 0)) for dt in DEVICE_TYPES]

    total_categories = sum(category_counts.values()) or 1
    proportions = [c / total_categories for c in category_counts.values()]
    category_entropy = -sum(p * math.log2(p) for p in proportions if p > 0)

    features = (
        event_features
        + device_features
        + [recency_score, total_purchase_value, category_entropy]
    )
    return FeatureVector(
        user_id=user_id,
        features=features,
        feature_names=_feature_names(),
    )


def _feature_names() -> list[str]:
    return (
        [f"event_{et}" for et in EVENT_TYPES]
        + [f"device_{dt}" for dt in DEVICE_TYPES]
        + ["recency_score", "purchase_value_sum", "category_entropy"]
    )


def normalize_features(vectors: list[FeatureVector]) -> list[FeatureVector]:
    """Min-max normalize each feature column across all users."""
    if not vectors:
        return vectors

    n = len(vectors[0].features)
    mins = [min(v.features[i] for v in vectors) for i in range(n)]
    maxs = [max(v.features[i] for v in vectors) for i in range(n)]

    normalized = []
    for fv in vectors:
        norm = [
            (fv.features[i] - mins[i]) / (maxs[i] - mins[i])
            if maxs[i] > mins[i] else 0.0
            for i in range(n)
        ]
        normalized.append(FeatureVector(
            user_id=fv.user_id,
            features=norm,
            feature_names=fv.feature_names,
        ))
    return normalized
