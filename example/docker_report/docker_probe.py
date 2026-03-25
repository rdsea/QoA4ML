import random
import signal
import sys
import time

from qoa4ml.qoa_client import QoaClient
from qoa4ml.reports.ml_reports import MLReport

client1 = QoaClient(report_cls=MLReport, config_path="./config/client1.yaml")


def signal_handler(sig, frame):
    client1.stop_all_probes()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
if __name__ == "__main__":
    # client1.start_all_probes()
    # NOTE: here the probe will run in the background so if the script ends fast, the probe will result in error
    for _ in range(5):
        client1.observe_metric("metric1", random.randint(1, 100), 0)
        client1.observe_metric("metric2", random.randint(1, 100), 0)
        client1.report(submit=True)
        time.sleep(1)
