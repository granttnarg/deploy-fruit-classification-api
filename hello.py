from fastapi import FastAPI, HTTPException, Depends, Header
from dotenv import load_dotenv
import os

CATEGORIES = ["freshapple", "freshbanana", "freshorange", 
              "rottenapple", "rottenbanana", "rottenorange"]

load_dotenv()
app = FastAPI()

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

@app.get("/")
def welcome():
    return { 'info': 'Welcome to the fruit classifer, use /predict endpoint to upload an image for a ML prediction on its category' }

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

