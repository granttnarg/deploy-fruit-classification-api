# Fast API Fruit Classifier - Resnet 18 ML

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
  ENVIRONMENT=development
```

3. Run the app:

```bash
fastapi run app/main.py --port 8080 --reload
```

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /categories` - List fruit categories (requires API key)
- `POST /predict` - Upload image for ML fruit classification class

## Authentication

Include API key in header:

```bash
curl -H "x-api-key: your-key" http://localhost:8000/categories
```

## Categories

- Fresh/Rotten: Apple, Banana, Orange

## Docker Build and run

build
`docker build -t myfastapiapp . `

run build
`docker run -p 8080:8080 myfastapiapp`
