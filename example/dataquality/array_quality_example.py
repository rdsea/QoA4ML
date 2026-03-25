"""Array and scalar data quality evaluation example.

Demonstrates how to use QoA4ML's data quality utilities to evaluate
distributions, missing values, duplicates, and erroneous entries in
tabular and array data, then report the results via QoaClient.
"""

import json
import os

import numpy as np
import pandas as pd

from qoa4ml.qoa_client import QoaClient
from qoa4ml.reports.ml_reports import MLReport
from qoa4ml.utils.dataquality_utils import (
    eva_duplicate,
    eva_erronous,
    eva_missing,
    eva_none,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "client1.yaml")


def main():
    client = QoaClient(report_cls=MLReport, config_path=CONFIG_PATH)

    # ---- Create a sample dataset with known quality issues ----
    print("=" * 60)
    print("Sample dataset")
    print("=" * 60)

    data = pd.DataFrame(
        {
            "temperature": [
                22.1,
                23.4,
                np.nan,
                21.0,
                22.1,
                23.4,
                25.5,
                np.nan,
                22.1,
                30.0,
            ],
            "humidity": [45, 50, 55, np.nan, 45, 50, 60, 65, 45, 70],
            "pressure": [1013, 1012, 1015, 1013, 1013, 1012, 1010, 1013, 1013, 1008],
        }
    )
    print(data.to_string())
    print()

    # ---- Evaluate erroneous entries ----
    print("=" * 60)
    print("Erroneous entry evaluation")
    print("=" * 60)

    error_values = [np.nan]
    erroneous_result = eva_erronous(data, error_values)
    if erroneous_result:
        for metric_name, value in erroneous_result.items():
            print(f"  {metric_name}: {value}")
            client.observe_metric(metric_name, float(value), category=1)
    print()

    # ---- Evaluate duplicate rows ----
    print("=" * 60)
    print("Duplicate entry evaluation")
    print("=" * 60)

    duplicate_result = eva_duplicate(data)
    if duplicate_result:
        for metric_name, value in duplicate_result.items():
            print(f"  {metric_name}: {value}")
            client.observe_metric(metric_name, float(value), category=1)
    print()

    # ---- Evaluate missing values ----
    print("=" * 60)
    print("Missing value evaluation")
    print("=" * 60)

    missing_result = eva_missing(data, null_count=True)
    if missing_result:
        for metric_name, value in missing_result.items():
            print(f"  {metric_name}:")
            print(f"    {value}")
            client.observe_metric(metric_name, str(value), category=1)
    print()

    # ---- Evaluate None/NaN statistics ----
    print("=" * 60)
    print("None/NaN statistics")
    print("=" * 60)

    none_result = eva_none(data)
    if none_result:
        for metric_name, value in none_result.items():
            print(f"  {metric_name}: {value}")
            client.observe_metric(metric_name, float(value), category=1)
    print()

    # ---- Evaluate a plain numpy array ----
    print("=" * 60)
    print("Numpy array evaluation")
    print("=" * 60)

    arr = np.array(
        [
            [1.0, 2.0, np.nan],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
            [1.0, 2.0, np.nan],
        ]
    )
    print(f"  Array shape: {arr.shape}")

    arr_none = eva_none(arr)
    if arr_none:
        for metric_name, value in arr_none.items():
            print(f"  {metric_name}: {value}")
            client.observe_metric(metric_name, float(value), category=1)

    arr_dup = eva_duplicate(arr)
    if arr_dup:
        for metric_name, value in arr_dup.items():
            print(f"  {metric_name}: {value}")
            client.observe_metric(metric_name, float(value), category=1)
    print()

    # ---- Generate and display the report ----
    print("=" * 60)
    print("Quality report")
    print("=" * 60)

    report = client.report(submit=True)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
