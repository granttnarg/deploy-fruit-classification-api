import os
import wandb
import torch
from loadotenv import load_env
from torchvision.models import resnet18, ResNet
from torch import nn
from pathlib import Path

MODELS_DIR = '../models'
MODEL_FILENAME = 'best_model.pth'

os.makedirs(MODELS_DIR, exist_ok=True)
wandb_api_key = os.environ.get("WANDB_API_KEY")

load_env()

def download_artifact():
    """Download the weights from our model from weights and biases """
    assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY not found in environment Variables"
    wandb.login(key=wandb_api_key)
    api = wandb.Api()

    wandb_org = os.environ.get("WANDB_ORG", "")
    wandb_project = os.environ.get("WANDB_PROJECT", "")
    wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
    wandb_model_version = os.environ.get("WANDB_MODEL_VERSION", "latest")

    if not wandb_model_name:
        raise ValueError("WANDB_MODEL_NAME environment variable is required")

    artifact_path = f'{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}'
    print(f"Artifact path: {artifact_path}")


    print(f"Downloading artifact {artifact_path} to {MODELS_DIR}")
    artifact = wandb.Api().artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)

def get_raw_model() -> ResNet:

    """Get tthe architexture ofr the model (random weights) - this must much the architexture during training"""
    # Uses random weights to initialize 
    architecture = resnet18(weights=None)

    # Change the model architecture to the one that we are actually using
    architecture.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 6)
    )

    return architecture

def load_model() -> ResNet:
    """Loads the model  with its wandb trained weights weights"""
    # download weights
    download_artifact()
    model = get_raw_model()
    # Get the trained model weights
    model_state_dict_path = Path(MODELS_DIR) / MODEL_FILENAME

    # this loads the weights from the file into a state_dictionary in memory
    model_state_dict = torch.load(model_state_dict_path, map_location='cpu')

    # This merges the weights into the model arch.
    model.load_state_dict(model_state_dict, strict=True)

    # Turn off BatchNorm and Dropout, uses the stats from training instead of stats from the inference set.
    model.eval()
    print('model eval mode')
    return model
    
resnet = load_model()
print(resnet)

