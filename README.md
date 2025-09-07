# FastAPI Fruit Classifier - ResNet 18 ML

A production-ready ML API that classifies fresh vs rotten fruits (apples, bananas, oranges) using a fine-tuned ResNet-18 model. The API is built with FastAPI and includes rate limiting, authentication, and comprehensive documentation.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` file:

```env
  WANDB_API_KEY=your-wandb-api-key
  WANDB_ORG=your-org
  WANDB_PROJECT=your-project
  WANDB_MODEL_NAME=your-model-name
  API_KEYS=key1,key2,key3
  ENVIRONMENT=development
  
  # Optional: Baseline model for comparison
  BASE_WANDB_ORG=your-org
  BASE_WANDB_PROJECT=your-baseline-project
  BASE_WANDB_MODEL_NAME=baseline-resnet18
```

3. Run the app:

```bash
fastapi run app/main.py --port 8080 --reload
```

## Features

- **Machine Learning**: ResNet-18 based fruit classification model
- **Model Caching**: Smart in-memory caching to avoid reloading models between requests
- **Baseline Comparison**: Optional baseline model support for performance comparison
- **Rate Limiting**: 2 requests per minute per IP for prediction endpoint
- **Authentication**: API key-based authentication for protected endpoints
- **Model Management**: Automatic model download from Weights & Biases
- **Docker Support**: Containerized deployment ready
- **Interactive Documentation**: Auto-generated Swagger UI at `/docs`

## API Endpoints

### Public Endpoints

- `GET /` - Smart redirect to `/docs` for browsers, JSON response for API clients
- `GET /welcome` - API information and available endpoints
- `GET /health` - Health check with system status
- `POST /predict` - Upload image for ML fruit classification (rate limited: 2/minute)

### Protected Endpoints (Require API Key)

- `GET /categories` - List all available fruit categories

## Authentication

Protected endpoints require an API key in the `x-api-key` header. Configure your API keys in the `.env` file.

## Model Information

The API uses a fine-tuned ResNet-18 model trained to classify:

### Categories

- **Fresh Fruits**: `freshapple`, `freshbanana`, `freshorange`
- **Rotten Fruits**: `rottenapple`, `rottenbanana`, `rottenorange`

### Model Architecture

- Base: ResNet-18 (ImageNet pre-trained)
- Custom classifier: 512 → 512 → 6 classes
- Input size: 224x224 RGB images
- Preprocessing: Resize(256) → CenterCrop(224) → Normalize(ImageNet stats)

### Training Data & Fine-tuning

The model was fine-tuned using the [Apples, Bananas, and Oranges dataset](https://www.kaggle.com/datasets/sriramr/apples-bananas-oranges) from Kaggle. This dataset contains images of fresh and rotten fruits across three categories:

- **Dataset**: Kaggle - Apples, Bananas, and Oranges
- **Classes**: 6 total (fresh/rotten × apple/banana/orange)
- **Training approach**: Transfer learning from ImageNet-pretrained ResNet-18
- **Custom head**: Added fully connected layers (512 → ReLU → 512 → 6) for fruit classification
- **Model storage**: Trained weights stored and versioned in Weights & Biases

### Response Format

```json
{
	"category": "freshapple",
	"confidence": 0.95
}
```

## Docker Deployment

### Build Image

```bash
docker build -t fruit-classifier-api .
```

### Run Container

```bash
docker run -p 8080:8080 --env-file .env fruit-classifier-api
```

### Environment Variables in Docker

Make sure your `.env` file contains all required variables before running the container.

## Testing

Run the test suite with pytest:

```bash
pytest app/tests/ -v
```

## Logging

The API includes comprehensive prediction logging that adapts to the deployment environment:

- **Local Development**: Logs to both `logs/predictions.log` file and console with timestamps
- **Google Cloud Run**: Logs only to stdout for Cloud Logging integration (auto-detected via `K_SERVICE` env var)
- **Log Content**: Prediction results, confidence scores, timing data, image metadata, and quality flags for low-confidence predictions

## Development

### Project Structure

```
app/
├── __init__.py
├── main.py         # FastAPI application and endpoints
├── model.py        # ML model loading and preprocessing
├── cache.py        # In-memory model caching system
├── logger.py       # Prediction logging with cloud adaptation
└── tests/          # Test suite
    ├── test_main.py     # API endpoint tests
    ├── test_model.py    # Model architecture tests
    ├── test_cache.py    # Caching functionality tests
    └── test_logging.py  # Logging functionality tests

requirements.txt    # Python dependencies
Dockerfile         # Container configuration
.env              # Environment variables (not tracked)
logs/             # Local log files (gitignored)
```

### Rate Limiting

- Prediction endpoint: 2 requests per minute per IP address
- Based on client IP address
- Returns HTTP 429 when exceeded

## API Documentation

Once running, visit:

- **Swagger UI**: http://localhost:8000/docs

## Troubleshooting

### Common Issues

1. **WANDB_API_KEY not found**: Ensure all required environment variables are set in `.env`
2. **Model download fails**: Check WANDB credentials and model path configuration
3. **Rate limit exceeded**: Wait before making additional requests to `/predict`
4. **Invalid API key**: Verify your API key is included in the `API_KEYS` environment variable

### Health Check Response

```json
{
	"status": "healthy",
	"api_keys_loaded": true,
	"categories_count": 6
}
```
