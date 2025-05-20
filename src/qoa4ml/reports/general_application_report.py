import copy
import time
from typing import Any
from uuid import UUID

from qoa4ml.config.configs import ClientInfo
from qoa4ml.lang.common_models import Metric
from qoa4ml.lang.datamodel_enum import ReportTypeEnum
from qoa4ml.reports.abstract_report import AbstractReport
from qoa4ml.reports.ml_report_model import (
    FlattenMetric,
    GeneralApplicationReportModel,
    MicroserviceInstance,
)


class GeneralApplicationReport(AbstractReport):
    """
    GeneralApplicationReport manages the reporting of application metrics and inference data.

    Parameters
    ----------
    client_config : ClientInfo
        Configuration settings related to the client.

    Attributes
    ----------
    client_config : ClientInfo
        A deep copy of the client configuration.
    init_time : float
        The initialization time of the report.
    report : GeneralApplicationReportModel
        The current state of the report.
    execution_instance : MicroserviceInstance
        An instance representing the current execution context.
    previous_reports : list[MicroserviceInstance]
        A list of previous execution instances.

    Methods
    -------
    reset() -> None
        Reset the report to an initial state.
    process_previous_report(previous_report_dict: dict) -> None
        Process and incorporate a previous report.
    observe_metric(report_type: ReportTypeEnum, stage: str, metric: Metric) -> None
        Observe and record a metric.
    observe_inference(inference_value: Any) -> None
        Observe and record inference data.
    observe_inference_metric(metric: Metric) -> None
        Observe and record an inference-specific metric.
    """

    def __init__(self, client_config: ClientInfo) -> None:
        """
        Initialize an instance of GeneralApplicationReport.

        Parameters
        ----------
        client_config : ClientInfo
            Configuration settings related to the client.
        """
        self.client_config = copy.deepcopy(client_config)
        self.reset()
        self.init_time = time.time()

    def reset(self) -> None:
        """
        Reset the report to an initial state.

        Notes
        -----
        - This method initializes a new report model and sets up the execution instance and previous reports list.
        """
        self.report = GeneralApplicationReportModel()
        self.execution_instance = MicroserviceInstance(
            id=UUID(self.client_config.instance_id),
            name=self.client_config.name,
            functionality=self.client_config.functionality,
            stage=self.client_config.stage_id,
        )
        self.previous_reports: list[MicroserviceInstance] = []

    def process_previous_report(self, previous_report_dict: dict) -> None:
        """
        Process and incorporate a previous report.

        Parameters
        ----------
        previous_report_dict : dict
            Dictionary representation of a previous report.

        Notes
        -----
        - This method assumes the last metric in the previous report was observed by the previous instance.
        - It appends the metrics from the previous report to the current report.
        """
        previous_report = GeneralApplicationReportModel(**previous_report_dict)
        self.previous_reports.append(previous_report.metrics[-1].instance)
        for metric in previous_report.metrics:
            self.report.metrics.append(metric)

    def observe_metric(
        self, report_type: ReportTypeEnum, stage: str, metric: Metric
    ) -> None:
        """
        Observe and record a metric.

        Parameters
        ----------
        report_type : ReportTypeEnum
            The type of report being generated.
        stage : str
            The stage of the process in which the metric is recorded.
        metric : Metric
            The metric to be recorded.
        """
        flatten_metric = FlattenMetric(
            metric_name=metric.metric_name,
            records=metric.records,
            unit=metric.unit,
            description=metric.description,
            stage=stage,
            report_type=report_type,
            instance=self.execution_instance,
            previous_instances=self.previous_reports,
        )
        self.report.metrics.append(flatten_metric)

    def observe_inference(self, inference_value: Any) -> None:
        """
        Observe and record inference data.

        Parameters
        ----------
        inference_value : Any
            The value of the inference to be recorded.

        Notes
        -----
        - This method records inference values as a metric with the name "Inference" and report type ml_specific.
        """
        flatten_metric = FlattenMetric(
            metric_name="Inference",
            records=inference_value,
            stage=self.client_config.stage_id,
            report_type=ReportTypeEnum.ml_specific,
            instance=self.execution_instance,
            previous_instances=self.previous_reports,
        )
        self.report.metrics.append(flatten_metric)

    def observe_inference_metric(self, metric: Metric) -> None:
        """
        Observe and record an inference-specific metric.

        Parameters
        ----------
        metric : Metric
            The inference-specific metric to be recorded.
        """
        flatten_metric = FlattenMetric(
            metric_name=metric.metric_name,
            records=metric.records,
            unit=metric.unit,
            description=metric.description,
            stage=self.client_config.stage_id,
            report_type=ReportTypeEnum.ml_specific,
            instance=self.execution_instance,
            previous_instances=self.previous_reports,
        )
        self.report.metrics.append(flatten_metric)
