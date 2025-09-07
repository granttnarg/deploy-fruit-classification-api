import torch
from torchvision.models import ResNet
from torchvision.transforms import v2 as transforms
from ..model import get_raw_model, load_transforms

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_wandb():
    with patch("wandb.login") as mock_login, patch("wandb.Api") as mock_api:

        # Make Api() return a mock object with artifact method
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        # Mock the artifact and its download method
        mock_artifact = MagicMock()
        mock_artifact.download.return_value = "path/to/fake/model"
        mock_api_instance.artifact.return_value = mock_artifact

        yield


def test_get_raw_model():
    """Test that raw model returns correct architecture"""
    model = get_raw_model()

    # Check it's a ResNet
    assert isinstance(model, ResNet)

    # Check the final layer is modified for 6 classes
    fc_layer = model.fc

    # Should be a Sequential with Linear -> ReLU -> Linear(512->6)
    assert isinstance(fc_layer, torch.nn.Sequential)
    assert len(fc_layer) == 3  # Linear, ReLU, Linear

    # Last layer should output 6 classes
    last_layer = fc_layer[-1]
    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.out_features == 6
    print("✓ Raw model architecture is correct")


def test_load_transforms():
    """Test that transforms are loaded correctly"""
    transform = load_transforms()
    # Should be a Compose object
    assert isinstance(transform, transforms.Compose)

    # Check number of transforms
    transforms_list = transform.transforms

    # Should have 5 transforms: Resize, CenterCrop, ToImage, ToDtype, Normalize
    assert len(transforms_list) == 5

    # Check specific transforms
    transform_names = [type(t).__name__ for t in transforms_list]
    expected = ["Resize", "CenterCrop", "ToImage", "ToDtype", "Normalize"]

    for expected_transform in expected:
        assert expected_transform in transform_names
    print("✓ Transforms loaded correctly")


def test_model_forward_pass():
    """Test that model can do forward pass with correct input shape"""
    model = get_raw_model()
    # Create fake input (batch_size=1, channels=3, height=224, width=224)
    fake_input = torch.randn(1, 3, 224, 224)
    # Put model in eval mode

    model.eval()
    with torch.no_grad():
        output = model(fake_input)

    # Output should be (1, 6) for 6 classes
    assert output.shape == (1, 6)

    # Output should be logits (not probabilities)
    # Apply softmax to check it sums to 1
    probs = torch.softmax(output, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-6)
    print("✓ Model forward pass works correctly")


def test_transforms_output_shape():
    """Test that transforms produce correct tensor shape"""
    from PIL import Image
    import numpy as np

    # Create fake RGB image
    fake_image = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )
    transform = load_transforms()
    transformed = transform(fake_image)

    # Should be (3, 224, 224) with float32
    assert transformed.shape == (3, 224, 224)
    assert transformed.dtype == torch.float32

    # Values should be normalized (roughly between -2 and 2 for ImageNet normalization)
    assert transformed.min() >= -3.0
    assert transformed.max() <= 3.0

    print("✓ Transforms produce correct output shape and dtype")


if __name__ == "__main__":
    print("Testing model.py (without wandb calls)...")
    test_get_raw_model()
    test_load_transforms()
    test_model_forward_pass()
    test_transforms_output_shape()

    print("\n✓ All model tests completed!")
