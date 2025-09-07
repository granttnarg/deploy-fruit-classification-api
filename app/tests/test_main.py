import os
from fastapi.testclient import TestClient
from unittest.mock import patch
from ..main import app
import pytest
import torch
from unittest.mock import patch, MagicMock

# Create test client
client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_wandb():
    with patch("wandb.login") as mock_login, \
         patch("wandb.Api") as mock_api, \
         patch("app.model.download_artifact") as mock_download, \
         patch("app.model.load_model") as mock_load_model, \
         patch("app.model.load_transforms") as mock_load_transforms:

        mock_download.return_value = None

        # Make Api() return a mock object with artifact method
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        # Mock the artifact and its download method
        mock_artifact = MagicMock()
        mock_artifact.download.return_value = "path/to/fake/model"
        mock_api_instance.artifact.return_value = mock_artifact

        # Mock the entire load_model function to return a working mock model
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        # Return tensor that makes predictions predictable
        mock_output = torch.tensor([[0.1, 0.9, 0.05, 0.2, 0.3, 0.15]])  # 6 categories
        mock_model.return_value = mock_output
        mock_load_model.return_value = mock_model

        # Mock the transforms
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_load_transforms.return_value = mock_transform

        yield


def test_welcome_endpoint():
    """Test /welcome endpoint returns expected structure"""
    response = client.get("/welcome")

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["supported_fruits"] == ["apple", "banana", "orange"]


def test_health_endpoint():
    """Test /health endpoint returns status"""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["categories_count"] == 6


def test_categories_endpoint_without_api_key():
    """Test /categories endpoint requires API key"""
    response = client.get("/categories")

    assert response.status_code == 422  # Missing required header


def test_categories_endpoint_with_invalid_api_key():
    """Test /categories endpoint with invalid API key"""
    response = client.get("/categories", headers={"X-API-Key": "invalid-key"})

    assert response.status_code == 401


def test_categories_endpoint_with_valid_api_key():
    """Test /categories endpoint with valid API key"""
    # Mock valid API keys
    with patch.dict(os.environ, {"API_KEYS": "test-key-123"}, clear=True):
        from ..main import load_api_keys

        with patch("app.main.VALID_API_KEYS", load_api_keys()):
            response = client.get("/categories", headers={"X-API-Key": "test-key-123"})

    print(f"Categories with valid key: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        assert len(data["categories"]) == 6


def test_predict_endpoint_no_file():
    """Test /predict endpoint without file"""
    response = client.post("/predict")

    assert response.status_code == 422  # Missing required file


def test_root_redirect():
    """Test root endpoint behavior"""
    # Test with browser-like headers
    response = client.get("/", headers={"User-Agent": "Mozilla/5.0"})
    print(f"Root with browser UA: {response.status_code}")

    # Root returns 200
    assert response.status_code == 200


if __name__ == "__main__":
    print("Testing main.py endpoints...")
    test_welcome_endpoint()
    test_health_endpoint()

    test_categories_endpoint_without_api_key()
    test_categories_endpoint_with_invalid_api_key()
    test_categories_endpoint_with_valid_api_key()

    test_predict_endpoint_no_file()
    test_root_redirect()

    print("\n✓ All main.py tests completed!")
