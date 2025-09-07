import uuid
import json
import sys
import datetime
import functools
import os
from typing import Callable
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_prediction_logging(log_file: str = "logs/predictions.log"):
    """Setup logging for predictions - adapts to environment (local vs cloud)"""

    prediction_logger = logging.getLogger("predictions")
    prediction_logger.setLevel(logging.INFO)
    prediction_logger.handlers.clear()

    is_cloud_run = (
        os.getenv("K_SERVICE") is not None
    )  # Turn of file logging for Cloud deployment, simple format for Google Cloud.

    if is_cloud_run:
        # Cloud Run: stdout only with simple format for Cloud Logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        prediction_logger.addHandler(console_handler)
    else:
        # Local development: write to file + stdout
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB file max, with backups
            backupCount=5,
        )
        file_handler.setLevel(logging.INFO)

        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add both handlers
        prediction_logger.addHandler(file_handler)
        prediction_logger.addHandler(console_handler)

    # Keep propagate=False to avoid duplicates from root logger
    prediction_logger.propagate = False

    return prediction_logger


def log_prediction(
    confidence_threshold: float = 0.9,
    track_inference_time: bool = False,
):
    """
    More detailed decorator that extracts model outputs for logging.
    This version captures the model outputs directly from the function context.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            prediction_id = str(uuid.uuid4())
            start_time = datetime.datetime.now()

            try:
                result = await func(*args, **kwargs)
                end_time = datetime.datetime.now()
                total_time = (end_time - start_time).total_seconds()

                prediction_log = {
                    "prediction_id": prediction_id,
                    "timestamp": start_time.isoformat(),
                    "total_time_seconds": round(total_time, 4),
                    "prediction": {
                        "category": result.category,
                        "confidence": round(result.confidence, 4),
                    },
                }

                # Add accurate inference timing if available
                if track_inference_time and hasattr(result, "inference_time"):
                    prediction_log["inference_time_seconds"] = round(
                        result.inference_time, 4
                    )
                    prediction_log["preprocessing_time_seconds"] = round(
                        total_time - result.inference_time, 4
                    )

                # Add quality flags
                quality_flags = []
                if result.confidence < confidence_threshold:
                    quality_flags.append("low_confidence")

                if quality_flags:
                    prediction_log["quality_flags"] = quality_flags

                # Extract additional parameters
                request = kwargs.get("request") or (args[0] if args else None)
                input_file = kwargs.get("input_image") or (
                    args[1] if len(args) > 1 else None
                )

                if input_file:
                    prediction_log["image_metadata"] = {
                        "filename": getattr(input_file, "filename", "unknown"),
                        "size": getattr(input_file, "size", 0),
                        "content_type": getattr(input_file, "content_type", "unknown"),
                    }

                if request and hasattr(request, "client"):
                    prediction_log["request_metadata"] = {
                        "client_ip": getattr(request.client, "host", "unknown"),
                        "user_agent": (
                            request.headers.get("user-agent", "unknown")
                            if hasattr(request, "headers")
                            else "unknown"
                        ),
                    }

                # Choose log level based on quality
                log_level = "warning" if quality_flags else "info"
                getattr(prediction_logger, log_level)(
                    f"PREDICTION: {json.dumps(prediction_log)}"
                )
                return result

            except Exception as e:
                error_log = {
                    "prediction_id": prediction_id,
                    "timestamp": start_time.isoformat(),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }

                prediction_logger.error(f"PREDICTION_ERROR: {json.dumps(error_log)}")
                raise

        return wrapper

    return decorator


prediction_logger = setup_prediction_logging("logs/predictions.log")
