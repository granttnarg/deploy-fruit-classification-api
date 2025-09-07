import os
import asyncio
from unittest.mock import patch, MagicMock
from ..logger import setup_prediction_logging, log_prediction


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

    # Capture log output
    import logging
    import io

    # Create string stream to capture logs
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    # Get the prediction logger and add our capture handler
    pred_logger = logging.getLogger("predictions")
    pred_logger.addHandler(handler)
    pred_logger.setLevel(logging.INFO)

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

    return has_info, has_warning


if __name__ == "__main__":
    print("Testing logging setup...")
    test_local_vs_cloud_logging()

    print("\nTesting prediction log levels...")
    has_info, has_warning = test_prediction_logging_levels()

    print(f"\n✓ Test complete - INFO: {has_info}, WARNING: {has_warning}")
