import torch
import io
import os
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
import datetime
from dotenv import load_dotenv
from app.logger import log_prediction
from .cache import ml_models

from contextlib import asynccontextmanager

from .model import get_model, get_transforms, load_model, load_transforms


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
        ml_models["classifier"] = load_model()
        ml_models["transforms"] = load_transforms()
        print("Models and Transforms loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        # You might want to exit here if models are critical
        raise
    yield
    # Clean up on shutdown
    print("Shutting down, clearing models...")
    ml_models.clear()


# Create FastAPI instance
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

# Add basic rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Debug message, to check that app is running


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
    description="Returns API info for programmatic access, redirects browsers to docs",
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


@app.post("/predict", response_model=Result)
@limiter.limit("2/minute")
@log_prediction()
async def predict(
    request: Request,
    input_image: UploadFile = File(...),
    model: ResNet = Depends(get_model),
    transforms: transforms.Compose = Depends(get_transforms),
) -> Result:
    image = Image.open(io.BytesIO(await input_image.read()))

    # Here we delete the alpha channel, the model doesn't use it
    # and will complain if the input has it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Here we add a batch dimension of 1
    image = transforms(image).reshape(1, 3, 224, 224)

    model.eval()  # We turn off dropout and uses batch stats from training

    # This is inference mode, we don't need gradient tracking
    inference_start = datetime.datetime.now()
    with torch.inference_mode():  # The newer version of no_grad
        outputs = model(image)
        confidence = (
            F.softmax(outputs, dim=1).max().item()
        )  # dim -> batch, output_categories
        category = CATEGORIES[outputs.argmax()]
    inference_end = datetime.datetime.now()

    inference_time = (inference_end - inference_start).total_seconds()

    return Result(
        category=category, confidence=confidence, inference_time_seconds=inference_time
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


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "api_keys_loaded": len(VALID_API_KEYS) > 0,
        "categories_count": len(CATEGORIES),
    }
