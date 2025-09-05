import os
import wandb
from loadotenv import load_env

MODELS_DIR = 'models'
MODEL_FILE_NAME = 'best_model.pth'

os.makedirs(MODELS_DIR, exist_ok=True)
wandb_api_key = os.environ.get("WANDB_API_KEY")

load_env()

def download_artifact():
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

download_artifact()




