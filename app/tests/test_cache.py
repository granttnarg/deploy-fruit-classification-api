import pytest
from unittest.mock import patch, MagicMock
from app.cache import ml_models
from app.model import get_model, get_transforms


def test_cache_initialization():
    """Test that cache starts empty"""
    # Clear cache for test
    ml_models.clear()
    assert len(ml_models) == 0


@patch("app.model.load_model")
@patch("app.model.load_transforms")
def test_model_caching_behavior(mock_load_transforms, mock_load_model):
    """Test that models are cached after first load"""
    # Clear cache for test
    ml_models.clear()

    # Setup mocks
    mock_model = MagicMock()
    mock_transform = MagicMock()
    mock_load_model.return_value = mock_model
    mock_load_transforms.return_value = mock_transform

    # First call should load from function
    model1 = get_model()
    assert model1 == mock_model
    assert mock_load_model.call_count == 1
    assert "classifier" in ml_models

    # Second call should use cache
    model2 = get_model()
    assert model2 == mock_model
    assert model1 is model2  # Same object reference
    assert mock_load_model.call_count == 1  # Not called again


@patch("app.model.load_model")
@patch("app.model.load_transforms")
def test_transforms_caching_behavior(mock_load_transforms, mock_load_model):
    """Test that transforms are cached after first load"""
    # Clear cache for test
    ml_models.clear()

    # Setup mocks
    mock_transform = MagicMock()
    mock_load_transforms.return_value = mock_transform

    # First call should load from function
    transforms1 = get_transforms()
    assert transforms1 == mock_transform
    assert mock_load_transforms.call_count == 1
    assert "transforms" in ml_models

    # Second call should use cache
    transforms2 = get_transforms()
    assert transforms2 == mock_transform
    assert transforms1 is transforms2  # Same object reference
    assert mock_load_transforms.call_count == 1  # Not called again


@patch("app.model.load_model")
@patch("app.model.load_transforms")
def test_cache_independence(mock_load_transforms, mock_load_model):
    """Test that model and transforms cache independently"""
    # Clear cache for test
    ml_models.clear()

    # Setup mocks
    mock_model = MagicMock()
    mock_transform = MagicMock()
    mock_load_model.return_value = mock_model
    mock_load_transforms.return_value = mock_transform

    # Load only model first
    get_model()
    assert "classifier" in ml_models
    assert "transforms" not in ml_models
    assert mock_load_model.call_count == 1
    assert mock_load_transforms.call_count == 0

    # Load transforms
    get_transforms()
    assert "classifier" in ml_models
    assert "transforms" in ml_models
    assert mock_load_model.call_count == 1
    assert mock_load_transforms.call_count == 1


def test_cache_manual_clear():
    """Test that cache can be manually cleared"""
    # Populate cache
    ml_models["test"] = "value"
    assert len(ml_models) > 0

    # Clear and verify
    ml_models.clear()
    assert len(ml_models) == 0


@patch("app.model.load_model")
def test_cache_behavior_after_clear(mock_load_model):
    """Test that cache works correctly after being cleared"""
    # Setup
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    # Load model, then clear cache
    get_model()
    assert mock_load_model.call_count == 1
    ml_models.clear()

    # Should load again after clear
    get_model()
    assert mock_load_model.call_count == 2
