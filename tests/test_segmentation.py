import pytest
from datetime import datetime, timezone, timedelta

from app.models.segment import (
    SegmentCreate, SegmentDefinition, SegmentRule, SegmentType,
    SegmentStatus, RuleOperator, UserEvent,
)
from app.services import segmentation
from app.services.feature_engineering import build_feature_vector, normalize_features


def _make_event(user_id: str, event_type: str, days_ago: int = 1, **kwargs) -> UserEvent:
    return UserEvent(
        user_id=user_id,
        event_type=event_type,
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
        **kwargs,
    )


def test_create_and_get_segment():
    seg = segmentation.create_segment(SegmentCreate(
        name="High Value Shoppers",
        segment_type=SegmentType.BEHAVIORAL,
        advertiser_id="adv_test",
    ))
    assert seg.id.startswith("seg_")
    assert seg.status == SegmentStatus.DRAFT

    fetched = segmentation.get_segment(seg.id)
    assert fetched is not None
    assert fetched.name == "High Value Shoppers"


def test_rule_based_segment_matching():
    seg = segmentation.create_segment(SegmentCreate(
        name="Cart Abandoners",
        segment_type=SegmentType.BEHAVIORAL,
        advertiser_id="adv_test",
        definition=SegmentDefinition(
            rules=[SegmentRule(field="event_type", operator=RuleOperator.EQUALS, value="add_to_cart")],
            min_events=1,
        ),
    ))
    segmentation.update_segment_status(seg.id, SegmentStatus.ACTIVE)

    segmentation.ingest_events([
        _make_event("user_001", "add_to_cart"),
        _make_event("user_002", "page_view"),
    ])

    result_001 = segmentation.evaluate_user("user_001")
    result_002 = segmentation.evaluate_user("user_002")

    assert seg.id in result_001.segment_ids
    assert seg.id not in result_002.segment_ids


def test_event_ingestion_count():
    events = [_make_event(f"user_{i}", "page_view") for i in range(10)]
    count = segmentation.ingest_events(events)
    assert count == 10


def test_feature_vector_shape():
    events = [
        _make_event("u1", "page_view", category="IAB1"),
        _make_event("u1", "purchase", value=99.99),
        _make_event("u1", "video_play", device_type="mobile"),
    ]
    fv = build_feature_vector("u1", events)
    assert fv.user_id == "u1"
    assert len(fv.features) == len(fv.feature_names)
    assert fv.features[fv.feature_names.index("event_page_view")] == 1.0
    assert fv.features[fv.feature_names.index("event_purchase")] == 1.0
    assert fv.features[fv.feature_names.index("purchase_value_sum")] == pytest.approx(99.99)


def test_normalize_features_range():
    events_a = [_make_event("u_a", "purchase", value=100.0)]
    events_b = [_make_event("u_b", "purchase", value=0.0)]
    fvs = [build_feature_vector("u_a", events_a), build_feature_vector("u_b", events_b)]
    normalized = normalize_features(fvs)
    purchase_idx = normalized[0].feature_names.index("purchase_value_sum")
    assert normalized[0].features[purchase_idx] == pytest.approx(1.0)
    assert normalized[1].features[purchase_idx] == pytest.approx(0.0)


def test_delete_segment():
    seg = segmentation.create_segment(SegmentCreate(
        name="Temp Segment",
        segment_type=SegmentType.CUSTOM,
        advertiser_id="adv_test",
    ))
    assert segmentation.delete_segment(seg.id)
    assert segmentation.get_segment(seg.id) is None
