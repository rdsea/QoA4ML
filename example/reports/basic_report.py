"""Basic QoA4ML report example.

Demonstrates how to use QoaClient with a debug connector to observe
service, data, and inference metrics, then generate a quality report.
No external services (e.g., RabbitMQ) are required.
"""

import json
import os
import random
import time

from qoa4ml.lang.attributes import (
    DataQualityEnum,
    MLModelQualityEnum,
    ServiceQualityEnum,
)
from qoa4ml.qoa_client import QoaClient
from qoa4ml.reports.ml_reports import MLReport

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "client.yaml")


def main():
    # ---- Step 1: Initialize the client with a debug connector ----
    print("=" * 60)
    print("Step 1: Creating QoaClient with debug connector")
    print("=" * 60)
    client = QoaClient(report_cls=MLReport, config_path=CONFIG_PATH)
    print("Client created successfully.\n")

    # ---- Step 2: Observe service-level metrics (category=0) ----
    print("=" * 60)
    print("Step 2: Observing service metrics")
    print("=" * 60)

    response_time_ms = random.uniform(50, 200)
    client.observe_metric(
        ServiceQualityEnum.RESPONSE_TIME,
        {"startTime": time.time(), "responseTime": response_time_ms},
        category=0,
        description="Simulated response time for a single request",
    )
    print(f"  Recorded response_time = {response_time_ms:.2f} ms")

    reliability = random.uniform(95, 100)
    client.observe_metric(
        ServiceQualityEnum.RELIABILITY,
        reliability,
        category=0,
        description="Service reliability percentage",
    )
    print(f"  Recorded reliability = {reliability:.2f}%")

    availability = random.uniform(99, 100)
    client.observe_metric(
        ServiceQualityEnum.AVAILABILITY,
        availability,
        category=0,
        description="Service availability percentage",
    )
    print(f"  Recorded availability = {availability:.2f}%\n")

    # ---- Step 3: Observe data quality metrics (category=1) ----
    print("=" * 60)
    print("Step 3: Observing data quality metrics")
    print("=" * 60)

    accuracy = random.uniform(90, 100)
    client.observe_metric(
        DataQualityEnum.ACCURACY,
        accuracy,
        category=1,
        description="Data accuracy ratio",
    )
    print(f"  Recorded data accuracy = {accuracy:.2f}%")

    completeness = random.uniform(85, 100)
    client.observe_metric(
        DataQualityEnum.COMPLETENESS,
        completeness,
        category=1,
        description="Data completeness ratio",
    )
    print(f"  Recorded data completeness = {completeness:.2f}%\n")

    # ---- Step 4: Observe inference metrics ----
    print("=" * 60)
    print("Step 4: Observing inference metrics")
    print("=" * 60)

    prediction = {"class": "cat", "score": 0.92}
    client.observe_inference(prediction)
    print(f"  Recorded inference prediction = {prediction}")

    confidence = random.uniform(0.8, 1.0)
    client.observe_inference_metric(MLModelQualityEnum.ACCURACY, confidence)
    print(f"  Recorded inference confidence = {confidence:.4f}\n")

    # ---- Step 5: Generate and display the report ----
    print("=" * 60)
    print("Step 5: Generating the quality report")
    print("=" * 60)

    report = client.report(submit=True)
    print(json.dumps(report, indent=2, default=str))

    print("\nReport generated and submitted via debug connector.")


if __name__ == "__main__":
    main()
