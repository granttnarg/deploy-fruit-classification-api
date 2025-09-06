# Fast API Fruit Classifier

A simple ML API that classifies fresh vs rotten fruits (apples, bananas, oranges).

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
```

3. Run the app:

```bash
fastapi run welcome.py
```

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /categories` - List fruit categories (requires API key)

## Authentication

Include API key in header:

```bash
curl -H "x-api-key: your-key" http://localhost:8000/categories
```

## Categories

- Fresh/Rotten: Apple, Banana, Orange
