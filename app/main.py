import torch
import io
import os
import datetime
import requests
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Depends, Header, Request, File, UploadFile
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import RedirectResponse
from torchvision.transforms import v2 as transforms
from torchvision.models import ResNet
from PIL import Image
import torch.nn.functional as F
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from app.logger import log_prediction
from .cache import ml_models
from .model import (
    get_model,
    get_transforms,
    load_model,
    load_transforms,
    load_basemodel,
    get_base_model,
)


# This is data model that describes the ouput of the API
class Result(BaseModel):
    category: str
    confidence: float
    inference_time_seconds: float


# Setup caching for our model
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and transforsm on startup
    try:
        # Our finetuned model is added to cache
        ml_models["classifier"] = load_model()
        ml_models["transforms"] = load_transforms()

        # Load baseline model (optional)
        baseline_model = load_basemodel()
        if baseline_model:
            ml_models["base_model"] = baseline_model
            print("Models, Transforms, and Baseline loaded successfully")
        else:
            ml_models["base_model"] = None
            print("Models and Transforms loaded successfully (baseline disabled)")
    except Exception as e:
        print(f"Error loading models: {e}")
        # You might want to exit here if models are critical
        raise
    yield
    # Clean up on shutdown
    print("Shutting down, clearing models...")
    ml_models.clear()


# Create FastAPI instance with lifespan setup for cache
app = FastAPI(
    title="Fruit Classifier API",
    description="A machine learning API for classifying fresh and rotten fruits using PyTorch ResNet",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Load env file in development only
if os.getenv("ENVIRONMENT") != "production":
    try:
        load_dotenv()
    except ImportError:
        pass

CATEGORIES = [
    "freshapple",
    "freshbanana",
    "freshorange",
    "rottenapple",
    "rottenbanana",
    "rottenorange",
]


def load_imagenet_classes():
    """Load ImageNet class names from torchvision or online source"""
    try:
        from torchvision.models import ResNet18_Weights

        return ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    except:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        response = requests.get(url)
        return response.json()


# Load ImageNet categories at module level
IMAGENET_CATEGORIES = load_imagenet_classes()


def perform_inference(
    image_tensor: torch.Tensor, main_model: ResNet, base_model: ResNet
):
    """
    Perform inference on both main and base models

    Args:
        image_tensor: Preprocessed image tensor (1, 3, 224, 224)
        main_model: Fine-tuned fruit classification model
        base_model: Baseline ImageNet model

    Returns:
        dict: Contains results for both models and timing info
    """
    main_model.eval()
    base_model.eval()

    inference_start = datetime.datetime.now()

    with torch.inference_mode():
        # Run inference on both models
        main_outputs = main_model(image_tensor)
        base_outputs = base_model(image_tensor)

        # Main model results
        main_confidence = F.softmax(main_outputs, dim=1).max().item()
        main_category = CATEGORIES[main_outputs.argmax()]

        # Base model results
        base_confidence = F.softmax(base_outputs, dim=1).max().item()
        base_category = IMAGENET_CATEGORIES[base_outputs.argmax()]

    inference_end = datetime.datetime.now()
    inference_time = (inference_end - inference_start).total_seconds()

    return {
        "main_model": {"category": main_category, "confidence": main_confidence},
        "base_model": {"category": base_category, "confidence": base_confidence},
        "inference_time_seconds": inference_time,
    }


# Add basic rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
RATE_LIMIT = os.getenv("RATELIMIT_PER_MIN", "2")


@app.get(
    "/",
    summary="Root Redirect",
    description="Redirects requests from '/' to the interactive Swagger UI. "
    "Useful for browser access, not for API clients.",
)
def root_endpoint(request: Request):
    """
    Smart endpoint that redirects browsers to /docs but returns JSON for API clients
    """
    user_agent = request.headers.get("user-agent", "").lower()
    accept_header = request.headers.get("accept", "").lower()

    # Check if it's likely a browser request
    is_browser = (
        "mozilla" in user_agent
        or "chrome" in user_agent
        or "safari" in user_agent
        or "firefox" in user_agent
        or "edge" in user_agent
        or "text/html" in accept_header
    )

    if is_browser:
        # Redirect browsers to the API documentation
        return RedirectResponse(url="/docs", status_code=302)


@app.get(
    "/welcome",
    summary="API Welcome",
    description="Returns API info for programmatic access.",
)
def welcome():
    return {
        "message": "Welcome to the Fruit Classifier API",
        "description": "Upload fruit images to /predict endpoint for ML classification",
        "version": "1.0.0",
        "docs": "/docs",
        "categories_endpoint": "/categories",
        "health_endpoint": "/health",
        "predict_endpoinrt": "/predict",
        "supported_fruits": ["apple", "banana", "orange"],
        "supported_conditions": ["fresh", "rotten"],
    }


@app.post(
    "/predict",
    summary="ML Classification Prediction",
    description="Upload an image of fruit to get a classification and confidence level",
    response_model=Result,
)
@limiter.limit(f"{RATE_LIMIT}/minute")
@log_prediction()
async def predict(
    request: Request,
    input_image: UploadFile = File(...),
    model: ResNet = Depends(get_model),
    base_model: ResNet = Depends(get_base_model),
    transforms: transforms.Compose = Depends(get_transforms),
) -> Result:
    # Handle image upload and preprocessing
    image = Image.open(io.BytesIO(await input_image.read()))

    # Convert to RGB (remove alpha channel if present)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply transforms and add batch dimension
    image_tensor = transforms(image).reshape(1, 3, 224, 224)

    # Perform inference on both models
    results = perform_inference(image_tensor, model, base_model)

    # Store base model info in request state for logging decorator
    request.state.base_model_info = results["base_model"]

    return Result(
        category=results["main_model"]["category"],
        confidence=results["main_model"]["confidence"],
        inference_time_seconds=results["inference_time_seconds"],
    )


def load_api_keys():
    """Load API keys from environment variables"""
    keys = set()
    api_keys_str = os.getenv("API_KEYS", "")
    if api_keys_str:
        return set(key.strip() for key in api_keys_str.split(",") if key.strip())

    return keys


VALID_API_KEYS = load_api_keys()


async def verify_api_key(x_api_key: str = Header(..., description="API Key")):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


@app.get("/categories")
async def protected_route(api_key: str = Depends(verify_api_key)):
    """Get all available data categories for our prediction model"""
    return {"categories": CATEGORIES, "total": len(CATEGORIES)}


def check_endpoint_health():
    """Check health of critical system components"""
    health_status = {
        "models_loaded": False,
        "transforms_loaded": False,
        "baseline_model_loaded": False,
        "api_keys_configured": False,
        "categories_available": False,
        "imagenet_categories_loaded": False,
    }

    try:
        # Check if models are in cache
        health_status["models_loaded"] = (
            "classifier" in ml_models and ml_models["classifier"] is not None
        )
        health_status["transforms_loaded"] = (
            "transforms" in ml_models and ml_models["transforms"] is not None
        )
        health_status["baseline_model_loaded"] = (
            "base_model" in ml_models and ml_models["base_model"] is not None
        )

        # Check API configuration
        health_status["api_keys_configured"] = len(VALID_API_KEYS) > 0
        health_status["categories_available"] = len(CATEGORIES) > 0
        health_status["imagenet_categories_loaded"] = len(IMAGENET_CATEGORIES) > 0

    except Exception as e:
        print(f"Health check error: {e}")

    return health_status


@app.get(
    "/health",
    summary="Returns Metrics to show overall health status of the API",
)
def health_check():
    endpoint_health = check_endpoint_health()

    # Determine overall status
    critical_checks = ["models_loaded", "transforms_loaded", "categories_available"]
    all_critical_healthy = all(endpoint_health[check] for check in critical_checks)

    status = "healthy" if all_critical_healthy else "degraded"

    return {
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(),
        "components": endpoint_health,
        "endpoints": {
            "predict": all_critical_healthy,
            "categories": endpoint_health["api_keys_configured"],
            "welcome": True,
            "docs": True,
        },
    }
