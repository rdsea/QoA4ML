import json
import logging
import os
import socket
from datetime import datetime
from pathlib import Path
from threading import Thread

from fastapi import APIRouter
from pydantic import ValidationError

from qoa4ml.collector.socket_collector import SocketCollector
from qoa4ml.config.configs import NodeAggregatorConfig
from qoa4ml.lang.datamodel_enum import EnvironmentEnum
from qoa4ml.observability.odop_obs.embedded_database import EmbeddedDatabase
from qoa4ml.reports.resources_report_model import ProcessReport, SystemReport
from qoa4ml.utils.qoa_utils import flatten, make_folder, unflatten

# Use a named logger instead of hijacking the root logger via basicConfig;
# the host application controls its own handlers/formatters.
logger = logging.getLogger(__name__)

METRICS_URL_PATH = "/metrics"


class NodeAggregator:
    def __init__(self, config: NodeAggregatorConfig, odop_path: Path):
        self.config = config
        self.unit_conversion = self.config.unit_conversion
        self.node_name = socket.gethostname().split(".")[0]
        self.database_path = os.path.join(odop_path, "metric_database/")
        make_folder(self.database_path)
        self.embedded_database = EmbeddedDatabase(
            Path(self.database_path + self.node_name + ".csv")
        )
        self.environment = config.environment
        self.collector = SocketCollector(
            config.socket_collector_config, self.process_report
        )
        self.server_thread = Thread(target=self.collector.start_collecting, daemon=True)
        self.router = APIRouter()
        self.router.add_api_route(
            METRICS_URL_PATH,
            self.get_latest_timestamp,
            methods=[self.config.query_method],
        )

    def process_report(self, report: str):
        try:
            report_dict = json.loads(report)
        except (json.JSONDecodeError, TypeError) as error:
            # Drop malformed frames instead of letting them crash the thread.
            logger.error(
                f"invalid_payload: dropping socket frame ({type(error).__name__}): {error}"
            )
            return

        if self.environment == EnvironmentEnum.hpc:
            self._process_hpc_report(report_dict)
        else:
            self._process_edge_report(report_dict)

    def _process_hpc_report(self, report_dict: dict) -> None:
        """HPC environment: reports are plain JSON dicts with a ``type`` key."""
        report_type = report_dict.get("type")
        if report_type == "system":
            tag_type = "node"
        elif report_type == "process":
            tag_type = "process"
        else:
            logger.error(f"unknown HPC report type: {report_type!r}")
            return

        if "metadata" not in report_dict or "timestamp" not in report_dict:
            logger.error("HPC report missing metadata or timestamp; dropping frame")
            return

        report_copy = dict(report_dict)
        report_copy.pop("type", None)
        metadata = flatten(
            {"metadata": report_copy.pop("metadata")}, self.config.data_separator
        )
        timestamp = report_copy.pop("timestamp")
        fields = self.convert_unit(flatten(report_copy, self.config.data_separator))
        self.embedded_database.insert(timestamp, {"type": tag_type, **metadata}, fields)

    def _process_edge_report(self, report_dict: dict) -> None:
        """Edge / Cloud environment: dict is validated into a Pydantic model.

        Previously the code did ``isinstance(report_dict, SystemReport)`` on
        the raw ``json.loads`` dict, which could never be True — both
        branches were dead. We now parse explicitly, preferring SystemReport
        first and falling back to ProcessReport.
        """
        system_report: SystemReport | None = None
        process_report: ProcessReport | None = None
        try:
            system_report = SystemReport(**report_dict)
        except ValidationError:
            try:
                process_report = ProcessReport(**report_dict)
            except ValidationError as error:
                logger.error(
                    f"edge report did not match SystemReport or ProcessReport: {error}"
                )
                return

        if system_report is not None:
            node_name = system_report.metadata.node_name
            timestamp = system_report.timestamp
            payload = system_report.model_dump(exclude_none=True)
            # Remove the metadata/timestamp keys so only metric fields are flattened.
            payload.pop("metadata", None)
            payload.pop("timestamp", None)
            fields = self.convert_unit(flatten(payload, self.config.data_separator))
            self.embedded_database.insert(
                timestamp,
                {"type": "node", "node_name": node_name},
                fields,
            )
            return

        assert process_report is not None  # narrowed by control flow above
        metadata = flatten(
            {"metadata": process_report.metadata.model_dump()},
            self.config.data_separator,
        )
        timestamp = process_report.timestamp
        payload = process_report.model_dump(exclude_none=True)
        payload.pop("metadata", None)
        payload.pop("timestamp", None)
        fields = self.convert_unit(flatten(payload, self.config.data_separator))
        self.embedded_database.insert(
            timestamp, {"type": "process", **metadata}, fields
        )

    def convert_unit(self, report: dict):
        """Translate unit strings via ``self.unit_conversion``, tolerating misses.

        Keys that begin with ``metadata`` are skipped so metadata fields
        never get unit-converted even if they happen to contain substrings
        like ``mem``/``cpu``/``gpu``. Unit strings not present in the map
        pass through unchanged rather than raising ``KeyError`` — a
        producer that emits a new unit can't poison the whole frame.
        """
        converted_report = dict(report)
        for key, value in report.items():
            if not isinstance(value, str):
                continue
            # Metadata is user-supplied free-form data; never re-map it.
            if key.startswith("metadata"):
                continue
            if "frequency" in key:
                converted_report[key] = self.unit_conversion.get("frequency", {}).get(
                    value, value
                )
            elif "mem" in key:
                converted_report[key] = self.unit_conversion.get("mem", {}).get(
                    value, value
                )
            elif "cpu" in key and "usage" in key:
                converted_report[key] = (
                    self.unit_conversion.get("cpu", {})
                    .get("usage", {})
                    .get(value, value)
                )
            elif "gpu" in key and "usage" in key:
                converted_report[key] = (
                    self.unit_conversion.get("gpu", {})
                    .get("usage", {})
                    .get(value, value)
                )
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

    def get_latest_timestamp(self):
        data = self.embedded_database.get_latest_timestamp()
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
        self.server_thread.start()
        logger.info("node aggregator started")

    def stop(self):
        # Stop the underlying socket collector first; the server thread loop
        # exits when collector.execution_flag flips, so just joining without
        # signalling the collector would block forever on accept().
        self.collector.stop()
        self.server_thread.join()
        logger.info("node aggregator stopped")
