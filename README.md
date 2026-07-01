# Audience Segmentation Service

ML-powered audience segmentation microservice for programmatic advertising. Build rule-based behavioral segments, ingest user events, evaluate real-time membership, and expand audiences with lookalike modeling using K-means clustering.

## Features

- **Rule-Based Segments** — AND/OR logic across behavioral, demographic, and contextual signals
- **Lookalike Modeling** — K-means clustering to expand a seed audience to similar users
- **Event Ingestion** — ingest raw user events (page views, add-to-cart, purchases, video plays)
- **Real-Time Membership** — evaluate which segments a user belongs to on-demand
- **Feature Engineering** — recency decay, event frequency, device affinity, category entropy
- **Segment Lifecycle** — `draft → building → active → paused → archived`

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8001 --reload
```

API docs: http://localhost:8001/docs

## Docker

```bash
docker compose up
```

## API Reference

### Segments

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/segments/` | Create a new segment |
| `GET` | `/v1/segments/` | List segments (filter by advertiser, status, type) |
| `GET` | `/v1/segments/{id}` | Get segment details |
| `PATCH` | `/v1/segments/{id}/status` | Update segment status |
| `GET` | `/v1/segments/{id}/users` | Get users in a segment |
| `POST` | `/v1/segments/{id}/lookalike` | Build lookalike expansion from seed users |
| `DELETE` | `/v1/segments/{id}` | Delete a segment |

### Events

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/events/ingest` | Batch ingest user events |

### Membership

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/membership/{user_id}` | Evaluate segment membership for a user |

## Example: Create a Behavioral Segment

```json
POST /v1/segments/
{
  "name": "High-Intent Shoppers",
  "segment_type": "behavioral",
  "advertiser_id": "adv_abc123",
  "definition": {
    "rules": [
      {"field": "event_type", "operator": "eq", "value": "add_to_cart"}
    ],
    "match_all": true,
    "lookback_days": 14,
    "min_events": 2
  },
  "ttl_days": 30
}
```

## Example: Ingest User Events

```json
POST /v1/events/ingest
[
  {
    "user_id": "usr_xyz",
    "event_type": "add_to_cart",
    "category": "IAB19",
    "value": 49.99,
    "device_type": "mobile",
    "timestamp": "2026-05-14T10:00:00Z"
  }
]
```

## Example: Lookalike Expansion

```json
POST /v1/segments/{id}/lookalike
{
  "seed_user_ids": ["usr_001", "usr_002", "usr_003"],
  "expansion_factor": 5.0
}
```

## Feature Engineering

Each user is represented by a feature vector:

| Feature | Description |
|---------|-------------|
| `event_{type}` | Count of each event type in lookback window |
| `device_{type}` | Count of events by device |
| `recency_score` | Exponential decay score (λ=0.1/day) |
| `purchase_value_sum` | Total purchase value |
| `category_entropy` | Content diversity score |

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **FastAPI** — async REST framework
- **Pydantic v2** — data validation
- **K-Means** — pure-Python lookalike modeling (drop-in scikit-learn for production)
- Python 3.12+
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01
<!-- Last updated: 2026-07-01 -->
