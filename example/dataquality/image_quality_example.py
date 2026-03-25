"""Image quality evaluation example.

Demonstrates how to use QoA4ML's image_quality utility to assess
image properties (size, color mode, channels) from numpy arrays and
byte buffers. Uses synthetic images so no external files are needed.
"""

import io
import json
import os

import numpy as np
from PIL import Image

from qoa4ml.qoa_client import QoaClient
from qoa4ml.reports.ml_reports import MLReport
from qoa4ml.utils.dataquality_utils import image_quality

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "client1.yaml")


def create_synthetic_rgb_image(width: int, height: int) -> np.ndarray:
    """Generate a random RGB image as a numpy array."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def create_synthetic_grayscale_image(width: int, height: int) -> np.ndarray:
    """Generate a random grayscale image as a numpy array."""
    return np.random.randint(0, 256, (height, width), dtype=np.uint8)


def numpy_array_to_bytes(arr: np.ndarray) -> bytes:
    """Convert a numpy image array to PNG bytes via PIL."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    client = QoaClient(report_cls=MLReport, config_path=CONFIG_PATH)

    # ---- Evaluate an RGB image from a numpy array ----
    print("=" * 60)
    print("Evaluating RGB image (numpy array)")
    print("=" * 60)

    rgb_image = create_synthetic_rgb_image(width=640, height=480)
    rgb_quality = image_quality(rgb_image)

    for metric_name, value in rgb_quality.items():
        print(f"  {metric_name}: {value}")
        client.observe_metric(metric_name, value, category=1)

    # ---- Evaluate a grayscale image from a numpy array ----
    print("\n" + "=" * 60)
    print("Evaluating grayscale image (numpy array)")
    print("=" * 60)

    gray_image = create_synthetic_grayscale_image(width=320, height=240)
    gray_quality = image_quality(gray_image)

    for metric_name, value in gray_quality.items():
        print(f"  {metric_name}: {value}")
        client.observe_metric(metric_name, value, category=1)

    # ---- Evaluate an image from bytes (PNG buffer) ----
    print("\n" + "=" * 60)
    print("Evaluating image from PNG bytes")
    print("=" * 60)

    png_bytes = numpy_array_to_bytes(rgb_image)
    bytes_quality = image_quality(png_bytes)

    for metric_name, value in bytes_quality.items():
        print(f"  {metric_name}: {value}")
        client.observe_metric(metric_name, value, category=1)

    # ---- Generate and display the report ----
    print("\n" + "=" * 60)
    print("Quality report")
    print("=" * 60)

    report = client.report(submit=True)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
