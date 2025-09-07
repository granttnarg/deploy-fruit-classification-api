import os
import wandb
import torch
from torchvision.models import resnet18, ResNet, ResNet18_Weights
from torchvision.transforms import v2 as transforms
from torch import nn
from pathlib import Path
from .cache import ml_models

MODELS_DIR = "../models"
MODEL_FILENAME = "best_model.pth"
BASE_MODEL_FILENAME = "resnet_basemodel.pth"

os.makedirs(MODELS_DIR, exist_ok=True)
wandb_api_key = os.environ.get("WANDB_API_KEY")


def download_artifact(is_baseline=False):
    """Download model artifacts from WandB
    
    Args:
        is_baseline (bool): If True, downloads baseline model using BASE_* env vars
    """
    assert (
        "WANDB_API_KEY" in os.environ
    ), "WANDB_API_KEY not found in environment variables"
    wandb.login(key=wandb_api_key)
    api = wandb.Api()

    if is_baseline:
        # Use BASE_ prefixed environment variables for baseline
        wandb_org = os.environ.get("BASE_WANDB_ORG", "")
        wandb_project = os.environ.get("BASE_WANDB_PROJECT", "")
        wandb_model_name = os.environ.get("BASE_WANDB_MODEL_NAME", "baseline-resnet18")
        wandb_model_version = os.environ.get("BASE_WANDB_MODEL_VERSION", "latest")
        
        if not wandb_project:
            raise ValueError("BASE_WANDB_PROJECT environment variable is required")
        
        model_type = "baseline"
    else:
        # Use regular environment variables for main model
        wandb_org = os.environ.get("WANDB_ORG", "")
        wandb_project = os.environ.get("WANDB_PROJECT", "")
        wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
        wandb_model_version = os.environ.get("WANDB_MODEL_VERSION", "latest")
        
        if not wandb_model_name:
            raise ValueError("WANDB_MODEL_NAME environment variable is required")
        
        model_type = "main"

    artifact_path = (
        f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"
    )
    print(f"{model_type.capitalize()} artifact path: {artifact_path}")

    print(f"Downloading {model_type} artifact {artifact_path} to {MODELS_DIR}")
    artifact = wandb.Api().artifact(artifact_path, type="model")
    artifact.download(root=MODELS_DIR)


def download_baseline_artifact():
    """Download the baseline model from WandB artifacts"""
    download_artifact(is_baseline=True)


def get_raw_model() -> ResNet:
    """Get tthe architexture ofr the model (random weights) - this must much the architexture during training"""
    # Uses random weights to initialize
    architecture = resnet18(weights=None)

    # Change the model architecture to the one that we are actually using
    architecture.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 6))

    return architecture


def load_model() -> ResNet:
    """Loads the model  with its wandb trained weights weights"""
    # download weights
    download_artifact()
    model = get_raw_model()
    # Get the trained model weights
    model_state_dict_path = Path(MODELS_DIR) / MODEL_FILENAME

    # this loads the weights from the file into a state_dictionary in memory
    model_state_dict = torch.load(model_state_dict_path, map_location="cpu", weights_only=False)

    # This merges the weights into the model arch.
    model.load_state_dict(model_state_dict, strict=True)

    # Turn off BatchNorm and Dropout, uses the stats from training instead of stats from the inference set.
    model.eval()
    print("Model loaded and set to Eval Mode.")
    return model


def load_transforms() -> transforms.Compose:
    weights = ResNet18_Weights.IMAGENET1K_V1
    # We could use this as as transform, but lets define it below instead.
    transform = weights.transforms()
    # Here we can see the transforms from the original ImageNet Training, however becareful as they are not printed in the order they need to be run in.
    print(transform)

    # Lets manually redefine the transforms we need
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def load_basemodel():
    """Load baseline ResNet18 model with ImageNet weights from WandB"""
    try:
        baseline_path = Path(MODELS_DIR) / BASE_MODEL_FILENAME
        
        # Download only if file doesn't exist
        if not baseline_path.exists():
            print("📥 Downloading baseline artifact from WandB...")
            download_baseline_artifact()
        else:
            print(f"📂 Using existing baseline model: {baseline_path}")
        
        # Load from file - could be full model or state dict
        loaded_data = torch.load(baseline_path, map_location="cpu", weights_only=False)
        
        # Check if it's a full model or just state dict
        if isinstance(loaded_data, torch.nn.Module):
            # It's a full model
            model = loaded_data
            print("📦 Loaded full baseline model")
        else:
            # It's a state dict
            model = resnet18(weights=None)  # Initialize without weights
            model.load_state_dict(loaded_data, strict=True)
            print("📊 Loaded baseline model from state dict")
            
        model.eval()
        print("✅ Baseline model set to Eval Mode.")
        return model

    except Exception as e:
        print(f"⚠️ Baseline model loading failed: {e}")
        print("📝 Baseline comparison will be disabled")
        print("💡 Make sure BASE_WANDB_PROJECT is set and baseline artifact exists")
        return None


# Load Model & Transforms from Cache if available
def get_model() -> ResNet:
    if "classifier" not in ml_models:
        print("Model not in cache, loading now...")
        ml_models["classifier"] = load_model()
    return ml_models["classifier"]


def get_transforms() -> transforms.Compose:
    if "transforms" not in ml_models:
        print("Transforms not in cache, loading now...")
        ml_models["transforms"] = load_transforms()
    return ml_models["transforms"]

def get_base_model():
    """Get baseline model from cache, returns None if not available"""
    return ml_models.get("base_model", None)

