import os
import asyncio
from unittest.mock import patch, MagicMock
from ..logger import setup_prediction_logging, log_prediction
import logging
import io

original_k_service = os.environ.pop("K_SERVICE", None)


def test_local_vs_cloud_logging():
    """Test that logger behaves differently in local vs cloud environments"""
    # Test local environment
    with patch.dict(os.environ, {}, clear=True):
        local_logger = setup_prediction_logging()
        print(f"Local handlers: {len(local_logger.handlers)} (should be 2)")

    # Test cloud environment
    with patch.dict(os.environ, {"K_SERVICE": "test-service"}, clear=True):
        cloud_logger = setup_prediction_logging()
        print(f"Cloud handlers: {len(cloud_logger.handlers)} (should be 1)")


def test_prediction_logging_levels():
    """Test that high and low confidence predictions use correct log levels"""
    # Create string stream to capture logs
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    # Set formatter to include log level
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Get the prediction logger and replace its handlers temporarily
    pred_logger = logging.getLogger("predictions")

    # Store original handlers
    original_handlers = pred_logger.handlers[:]

    # Clear existing handlers and add our test handler
    pred_logger.handlers.clear()
    pred_logger.addHandler(handler)
    pred_logger.setLevel(logging.INFO)

    try:
        # Test high confidence (should be INFO level)
        high_confidence_result = MagicMock()
        high_confidence_result.category = "freshapple"
        high_confidence_result.confidence = 0.95

        @log_prediction(confidence_threshold=0.9)
        async def high_confidence_predict():
            return high_confidence_result

        # Test low confidence (should be WARNING level)
        low_confidence_result = MagicMock()
        low_confidence_result.category = "rottenbanana"
        low_confidence_result.confidence = 0.3

        @log_prediction(confidence_threshold=0.9)
        async def low_confidence_predict():
            return low_confidence_result

        # Run tests
        result1 = asyncio.run(high_confidence_predict())
        result2 = asyncio.run(low_confidence_predict())

        # Get captured logs
        log_output = log_capture.getvalue()

        print("=== LOG OUTPUT ===")
        print(log_output)
        print("==================")

        # Check that we have both INFO and WARNING in output
        has_info = "INFO" in log_output
        has_warning = "WARNING" in log_output

        print(f"High confidence result: {result1.confidence}")
        print(f"Low confidence result: {result2.confidence}")
        print(f"Has INFO level: {has_info}")
        print(f"Has WARNING level: {has_warning}")

        assert has_info, "Expected INFO level log for high confidence prediction"
        assert has_warning, "Expected WARNING level log for low confidence prediction"

    finally:
        # Restore original handlers
        pred_logger.handlers.clear()
        pred_logger.handlers.extend(original_handlers)


if __name__ == "__main__":
    print("Testing logging setup...")
    test_local_vs_cloud_logging()
    print("\nTesting prediction log levels...")
    test_prediction_logging_levels()  # Fixed: removed return value assignment
    print(f"\n✓ Test complete")
