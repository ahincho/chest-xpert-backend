"""Integration tests for the health endpoint."""


class TestHealthEndpoint:
    """Tests for GET /health — requires test_client (model loaded)."""

    def test_health_returns_200(self, test_client):
        """GET /health returns 200 with status ok."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_response_format(self, test_client):
        """Response contains exactly the expected keys."""
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_method_not_allowed(self, test_client):
        """POST to /health returns 405 Method Not Allowed."""
        response = test_client.post("/health")
        assert response.status_code == 405
