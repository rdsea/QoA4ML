import io
import random
import time

import numpy as np
import pandas as pd

from qoa4ml.qoa_client import QoaClient
from qoa4ml.reports.ml_reports import MLReport
from qoa4ml.utils.dataquality_utils import (
    eva_duplicate,
    eva_erronous,
    eva_missing,
    eva_none,
    image_quality,
)

client1 = QoaClient(report_cls=MLReport, config_path="./config/client1.yaml")

img_np = np.random.randint(
    0, 255, (random.randint(100, 600), random.randint(100, 600), 3), dtype=np.uint8
)

if __name__ == "__main__":
    start_time = time.time()

    img_bytes = io.BytesIO()
    image_quality_bytes_result = image_quality(img_np)

    for dataquality, value in image_quality_bytes_result.items():
        client1.observe_metric(dataquality, value, 1)

    received_df = pd.DataFrame(
        {
            "col1": [1, 2, 4, 4, 3, np.nan, 5, 100, 100],
            "col2": [5, 6, 4, 4, np.nan, np.nan, 2, 100, 100],
            "col3": ["a", "b", "f", "f", "c", "c", "d", "s", "s"],
        }
    )
    errors_list = [np.nan, "c"]

    eva_erronous_result = eva_erronous(received_df, errors_list)
    for dataquality, value in eva_erronous_result.items():
        client1.observe_metric(dataquality, value, 1)

    eva_duplicate_result = eva_duplicate(received_df)
    for dataquality, value in eva_duplicate_result.items():
        client1.observe_metric(dataquality, value, 1)

    eva_missing_result = eva_missing(received_df, null_count=True, correlations=True)
    for dataquality, value in eva_missing_result.items():
        client1.observe_metric(dataquality, value, 1)

    eva_none_result = eva_none(received_df)
    for dataquality, value in eva_none_result.items():
        client1.observe_metric(dataquality, value, 1)
    report_1 = client1.report(submit=True)
