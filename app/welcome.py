from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
import os

CATEGORIES = ["freshapple", "freshbanana", "freshorange",
              "rottenapple", "rottenbanana", "rottenorange"]

load_dotenv()
app = FastAPI(
    title="Fruit Classifier API",
    description="A machine learning API for classifying fresh and rotten fruits using PyTorch ResNet",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

def load_api_keys():
    """Load API keys from environment variables"""
    api_keys_str = os.getenv("API_KEYS", "")
    if api_keys_str:
        return set(key.strip() for key in api_keys_str.split(",") if key.strip())

    return keys

VALID_API_KEYS = load_api_keys()

async def verify_api_key(x_api_key: str = Header(..., description="API Key")):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.get("/",
         summary="API Redirect",
         description="Redirects '/' to our swagger docs when using Browser")
def root_endpoint(request: Request):
    """
    Smart endpoint that redirects browsers to /docs but returns JSON for API clients
    """
    user_agent = request.headers.get("user-agent", "").lower()
    accept_header = request.headers.get("accept", "").lower()

    # Check if it's likely a browser request
    is_browser = (
        "mozilla" in user_agent or
        "chrome" in user_agent or
        "safari" in user_agent or
        "firefox" in user_agent or
        "edge" in user_agent or
        "text/html" in accept_header
    )

    if is_browser:
        # Redirect browsers to the API documentation
        return RedirectResponse(url="/docs", status_code=302)

@app.get("/welcome",
         summary="API Welcome",
         description="Returns API info for programmatic access, redirects browsers to docs")
def welcome():
    return {
        'message': 'Welcome to the Fruit Classifier API',
        'description': 'Upload fruit images to /predict endpoint for ML classification',
        'version': '1.0.0',
        'docs': '/docs',
        'categories_endpoint': '/categories',
        'health_endpoint': '/health',
        'supported_fruits': ['apple', 'banana', 'orange'],
        'supported_conditions': ['fresh', 'rotten']
    }

@app.get("/categories")
async def protected_route(api_key: str = Depends(verify_api_key)):
    """Get all available data categories for our prediction model"""
    return {
        "categories": CATEGORIES,
        "total": len(CATEGORIES)
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "api_keys_loaded": len(VALID_API_KEYS) > 0,
        "categories_count": len(CATEGORIES)
    }

