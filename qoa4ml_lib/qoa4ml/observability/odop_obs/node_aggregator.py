import logging
import socket
from datetime import datetime
from threading import Thread
from typing import TYPE_CHECKING

import lazy_import
from fastapi import APIRouter
from flatten_dict import flatten, unflatten

from ...collector.socket_collector import SocketCollector
from ...config.configs import NodeAggregatorConfig
from ...lang.datamodel_enum import EnvironmentEnum
from ...utils.qoa_utils import make_folder
from .common import ODOP_PATH
from .embedded_database import EmbeddedDatabase

logging.basicConfig(
    format="%(asctime)s:%(levelname)s -- %(message)s", level=logging.INFO
)

DEFAULT_DATABASE_FOLDER = ODOP_PATH + "metric_database/"
make_folder(DEFAULT_DATABASE_FOLDER)
METRICS_URL_PATH = "/metrics"

if TYPE_CHECKING:
    from ...reports.resources_report_model import ProcessReport, SystemReport
else:
    ProcessReport = lazy_import.lazy_class(
        "...reports.resources_report_model", "ProcessReport"
    )
    SystemReport = lazy_import.lazy_class(
        "...reports.resources_report_model", "SystemReport"
    )


class NodeAggregator:
    def __init__(self, config: NodeAggregatorConfig):
        self.config = config
        self.unit_conversion = self.config.unit_conversion
        self.node_name = socket.gethostname().split(".")[0]
        self.embedded_database = EmbeddedDatabase(
            DEFAULT_DATABASE_FOLDER + self.node_name + ".csv"
        )
        self.environment = config.environment
        self.collector = SocketCollector(
            config.socket_collector_config, self.process_report
        )
        self.server_thread = Thread(target=self.collector.start_collecting, daemon=True)
        self.router = APIRouter()
        self.router.add_api_route(
            METRICS_URL_PATH,
            self.get_lastest_timestamp,
            methods=[self.config.query_method],
        )

    def process_report(self, report):
        if self.environment == EnvironmentEnum.hpc:
            if report["type"] == "system":
                del report["type"]
                metadata = flatten(
                    {"metadata": report["metadata"]}, self.config.data_separator
                )
                timestamp = report["timestamp"]
                del report["metadata"], report["timestamp"]
                fields = self.convert_unit(flatten(report, self.config.data_separator))
                self.embedded_database.insert(
                    timestamp,
                    {"type": "node", **metadata},
                    fields,
                )
            elif report["type"] == "process":
                del report["type"]
                metadata = flatten(
                    {"metadata": report["metadata"]}, self.config.data_separator
                )
                timestamp = report["timestamp"]
                del report["metadata"], report["timestamp"]
                fields = self.convert_unit(flatten(report, self.config.data_separator))
                self.embedded_database.insert(
                    timestamp,
                    {"type": "process", **metadata},
                    fields,
                )
            else:
                logging.error("Value Error: Unkown report type")
        else:
            if isinstance(report, SystemReport):
                node_name = report.metadata.node_name
                timestamp = report.timestamp
                del report.metadata, report.timestamp
                fields = self.convert_unit(
                    flatten(report.dict(exclude_none=True), self.config.data_separator)
                )
                self.embedded_database.insert(
                    timestamp,
                    {
                        "type": "node",
                        "node_name": node_name,
                    },
                    fields,
                )
            elif isinstance(report, ProcessReport):
                metadata = flatten(
                    {"metadata": report.metadata.dict()}, self.config.data_separator
                )
                timestamp = report.timestamp
                del report.metadata, report.timestamp
                fields = self.convert_unit(
                    flatten(report.dict(exclude_none=True), self.config.data_separator)
                )
                self.embedded_database.insert(
                    timestamp, {"type": "process", **metadata}, fields
                )

    def convert_unit(self, report: dict):
        converted_report = report
        for key, value in report.items():
            # TODO: for instead of hard coded
            if isinstance(value, str):
                if "frequency" in key:
                    converted_report[key] = self.unit_conversion["frequency"][value]
                elif "mem" in key:
                    converted_report[key] = self.unit_conversion["mem"][value]
                elif "cpu" in key:
                    if "usage" in key:
                        converted_report[key] = self.unit_conversion["cpu"]["usage"][
                            value
                        ]
                elif "gpu" in key:
                    if "usage" in key:
                        converted_report[key] = self.unit_conversion["gpu"]["usage"][
                            value
                        ]
        return converted_report

    def revert_unit(self, converted_report: dict):
        original_report = converted_report.copy()
        for key, value in converted_report.items():
            if "unit" in key:
                if "frequency" in key:
                    for original_unit, converted_unit in self.unit_conversion[
                        "frequency"
                    ].items():
                        if converted_unit == value:
                            original_report[key] = original_unit
                            break
                elif "mem" in key:
                    for original_unit, converted_unit in self.unit_conversion[
                        "mem"
                    ].items():
                        if converted_unit == value:
                            original_report[key] = original_unit
                            break
                elif "cpu" in key:
                    if "usage" in key:
                        for original_unit, converted_unit in self.unit_conversion[
                            "cpu"
                        ]["usage"].items():
                            if converted_unit == value:
                                original_report[key] = original_unit
                                break
                elif "gpu" in key:
                    if "usage" in key:
                        for original_unit, converted_unit in self.unit_conversion[
                            "gpu"
                        ]["usage"].items():
                            if converted_unit == value:
                                original_report[key] = original_unit
                                break
        return original_report

    def get_lastest_timestamp(self):
        data = self.embedded_database.get_lastest_timestamp()
        return [
            unflatten(
                self.revert_unit(
                    {
                        "timestamp": datetime.timestamp(datapoint.time),
                        **datapoint.tags,
                        **datapoint.fields,
                    }
                ),
                self.config.data_separator,
            )
            for datapoint in data
        ]

    def start(self):
        self.execution_flag = True
        self.server_thread.start()
        logging.info("node aggregator started")

    def stop(self):
        self.execution_flag = False
        self.server_thread.join()
        logging.info("node aggregator stopped")
