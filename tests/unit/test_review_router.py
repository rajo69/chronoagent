"""Unit tests for POST /api/v1/review and GET /api/v1/review/{review_id}."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from chronoagent.agents.summarizer import ReviewReport
from chronoagent.config import Settings
from chronoagent.main import create_app
from chronoagent.pipeline.graph import ReviewPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_PAYLOAD: dict[str, Any] = {
    "pr_id": "pr_test_001",
    "title": "Add user authentication",
    "description": "Implements JWT-based auth for all protected routes.",
    "diff": "+ import jwt\n+ def login(user, pwd): ...",
    "files_changed": ["auth.py", "routes.py"],
}


@pytest.fixture()
def client() -> TestClient:
    """TestClient with MockBackend pipeline pre-wired on app.state."""
    settings = Settings(env="test", llm_backend="mock")
    app = create_app(settings=settings)

    # Override lifespan state so the test never hits Together.ai or real Chroma.
    # create() uses MockBackend + EphemeralClient.
    with TestClient(app) as c:
        # Lifespan has run; pipeline + store are already set by create_app lifespan.
        yield c


# ---------------------------------------------------------------------------
# POST /api/v1/review
# ---------------------------------------------------------------------------


class TestSubmitReview:
    """POST /api/v1/review — happy path and validation."""

    def test_returns_201(self, client: TestClient) -> None:
        """Successful submission returns HTTP 201."""
        response = client.post("/api/v1/review", json=_SAMPLE_PAYLOAD)
        assert response.status_code == 201

    def test_response_pr_id_matches(self, client: TestClient) -> None:
        """Response pr_id echoes the submitted pr_id."""
        response = client.post("/api/v1/review", json=_SAMPLE_PAYLOAD)
        assert response.json()["pr_id"] == _SAMPLE_PAYLOAD["pr_id"]

    def test_response_title_matches(self, client: TestClient) -> None:
        """Response title echoes the submitted title."""
        response = client.post("/api/v1/review", json=_SAMPLE_PAYLOAD)
        assert response.json()["title"] == _SAMPLE_PAYLOAD["title"]

    def test_overall_risk_is_valid(self, client: TestClient) -> None:
        """overall_risk is one of the expected severity strings."""
        valid = {"none", "low", "medium", "high", "critical"}
        response = client.post("/api/v1/review", json=_SAMPLE_PAYLOAD)
        assert response.json()["overall_risk"] in valid

    def test_security_findings_is_list(self, client: TestClient) -> None:
        """security_findings is a list."""
        response = client.post("/api/v1/review", json=_SAMPLE_PAYLOAD)
        assert isinstance(response.json()["security_findings"], list)

    def test_style_findings_is_list(self, client: TestClient) -> None:
        """style_findings is a list."""
        response = client.post("/api/v1/review", json=_SAMPLE_PAYLOAD)
        assert isinstance(response.json()["style_findings"], list)

    def test_markdown_is_nonempty_string(self, client: TestClient) -> None:
        """markdown field is a non-empty string."""
        response = client.post("/api/v1/review", json=_SAMPLE_PAYLOAD)
        assert isinstance(response.json()["markdown"], str)
        assert len(response.json()["markdown"]) > 0

    def test_files_changed_defaults_to_empty(self, client: TestClient) -> None:
        """Omitting files_changed is accepted (defaults to [])."""
        payload = {k: v for k, v in _SAMPLE_PAYLOAD.items() if k != "files_changed"}
        payload["pr_id"] = "pr_no_files"
        response = client.post("/api/v1/review", json=payload)
        assert response.status_code == 201

    def test_missing_required_field_returns_422(self, client: TestClient) -> None:
        """Missing required field (title) returns HTTP 422."""
        payload = {k: v for k, v in _SAMPLE_PAYLOAD.items() if k != "title"}
        payload["pr_id"] = "pr_bad"
        response = client.post("/api/v1/review", json=payload)
        assert response.status_code == 422

    def test_content_type_json(self, client: TestClient) -> None:
        """Response content-type is application/json."""
        response = client.post("/api/v1/review", json=_SAMPLE_PAYLOAD)
        assert "application/json" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# GET /api/v1/review/{review_id}
# ---------------------------------------------------------------------------


class TestGetReview:
    """GET /api/v1/review/{review_id} — retrieval and 404 handling."""

    def test_returns_200_after_submit(self, client: TestClient) -> None:
        """GET returns HTTP 200 after the review has been submitted."""
        payload = dict(_SAMPLE_PAYLOAD)
        payload["pr_id"] = "pr_get_001"
        client.post("/api/v1/review", json=payload)
        response = client.get(f"/api/v1/review/{payload['pr_id']}")
        assert response.status_code == 200

    def test_retrieved_pr_id_matches(self, client: TestClient) -> None:
        """Retrieved report pr_id matches the requested id."""
        payload = dict(_SAMPLE_PAYLOAD)
        payload["pr_id"] = "pr_get_002"
        client.post("/api/v1/review", json=payload)
        response = client.get(f"/api/v1/review/{payload['pr_id']}")
        assert response.json()["pr_id"] == payload["pr_id"]

    def test_retrieved_overall_risk_is_valid(self, client: TestClient) -> None:
        """Retrieved overall_risk is a valid severity level."""
        payload = dict(_SAMPLE_PAYLOAD)
        payload["pr_id"] = "pr_get_003"
        client.post("/api/v1/review", json=payload)
        response = client.get(f"/api/v1/review/{payload['pr_id']}")
        valid = {"none", "low", "medium", "high", "critical"}
        assert response.json()["overall_risk"] in valid

    def test_missing_review_returns_404(self, client: TestClient) -> None:
        """GET for an unknown id returns HTTP 404."""
        response = client.get("/api/v1/review/does_not_exist")
        assert response.status_code == 404

    def test_404_detail_mentions_id(self, client: TestClient) -> None:
        """404 error detail contains the requested review id."""
        response = client.get("/api/v1/review/ghost_pr")
        assert "ghost_pr" in response.json()["detail"]

    def test_get_returns_same_report_as_post(self, client: TestClient) -> None:
        """GET returns the identical report that POST computed."""
        payload = dict(_SAMPLE_PAYLOAD)
        payload["pr_id"] = "pr_idempotent"
        post_body = client.post("/api/v1/review", json=payload).json()
        get_body = client.get(f"/api/v1/review/{payload['pr_id']}").json()
        assert post_body["overall_risk"] == get_body["overall_risk"]
        assert post_body["markdown"] == get_body["markdown"]
        assert len(post_body["security_findings"]) == len(get_body["security_findings"])
        assert len(post_body["style_findings"]) == len(get_body["style_findings"])

    def test_multiple_reviews_stored_independently(self, client: TestClient) -> None:
        """Different pr_ids produce independently stored reports."""
        for i in range(3):
            payload = dict(_SAMPLE_PAYLOAD)
            payload["pr_id"] = f"pr_multi_{i}"
            client.post("/api/v1/review", json=payload)

        for i in range(3):
            response = client.get(f"/api/v1/review/pr_multi_{i}")
            assert response.status_code == 200
            assert response.json()["pr_id"] == f"pr_multi_{i}"
