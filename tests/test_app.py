from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_returns_200_and_healthy_status():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["application"] == "student-ml-api"


def test_health_reports_correct_version():
    """Guards against the version string drifting out of sync across the app."""
    response = client.get("/health")
    assert response.json()["version"] == "1.0.0"


def test_predict_success_returns_expected_shape_and_value():
    """Happy path: correct math AND correct response shape (both keys present)."""
    response = client.post("/predict", json={"value": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["input"] == 10
    assert body["prediction"] == 20


def test_predict_missing_input_returns_422():
    """Missing required field must be rejected by validation, not crash the server (500)."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_type_returns_422():
    """Wrong type (string instead of number) must be rejected — proves Pydantic validation is active."""
    response = client.post("/predict", json={"value": "not-a-number"})
    assert response.status_code == 422


def test_predict_handles_negative_and_zero_values():
    """Boundary/edge-case test: negative and zero inputs shouldn't be treated as 'missing' or error."""
    response_zero = client.post("/predict", json={"value": 0})
    assert response_zero.status_code == 200
    assert response_zero.json()["prediction"] == 0

    response_negative = client.post("/predict", json={"value": -5})
    assert response_negative.status_code == 200
    assert response_negative.json()["prediction"] == -10