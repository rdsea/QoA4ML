from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any

from qoa4ml.config.configs import ClientInfo, ProbeConfig
from qoa4ml.connector.base_connector import BaseConnector
from qoa4ml.utils.qoa_utils import make_folder
from qoa4ml.utils.repeated_timer import RepeatedTimer


class Probe(ABC):
    def __init__(
        self,
        config: ProbeConfig,
        connector: BaseConnector,
        client_info: ClientInfo | None = None,
    ) -> None:
        self.config = config
        self.client_info = client_info
        self.frequency = self.config.frequency
        self.monitoring_interval = 1.0 / self.frequency
        self.execution_flag = False
        self.report_thread = None
        self.log_latency_flag = self.config.log_latency_flag
        if self.log_latency_flag:
            self.latency_logging_path = config.latency_logging_path
            make_folder(self.latency_logging_path)
        self.max_latency = 0.0
        self.connector = connector

    @abstractmethod
    def create_report(self) -> Any:
        pass

    def reporting(self):
        report = self.create_report()
        self.connector.send_report(report)

    def start_reporting(self, background: bool = True):
        """
        background = False for blocking reporting
        """
        self.execution_flag = True
        current_time = time.time()
        time.sleep(math.ceil(current_time) - current_time)
        self.timer = RepeatedTimer(self.monitoring_interval, self.reporting)
        if not background:
            self.timer.thread.join()

    def stop_reporting(self):
        if not hasattr(self, "timer"):
            raise RuntimeError("Can't stop reporting when the timer is not created yet")
        self.timer.stop()

    def send_report(self, report):
        start = time.time()
        self.connector.send_report(report)
        if self.log_latency_flag and self.latency_logging_path:
            self.write_log(
                (time.time() - start) * 1000,
                self.latency_logging_path + "report_latency.txt",
            )

    def write_log(self, latency, filepath: str):
        with open(filepath, "a", encoding="utf-8") as file:
            file.write(str(latency) + "\n")
