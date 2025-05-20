import copy
import time
from typing import Any, Optional
from uuid import UUID, uuid4

from qoa4ml.config.configs import ClientInfo
from qoa4ml.lang.common_models import Metric
from qoa4ml.lang.datamodel_enum import ReportTypeEnum
from qoa4ml.reports.abstract_report import AbstractReport
from qoa4ml.reports.ml_report_model import (
    BaseReport,
    GeneralMlInferenceReport,
    InferenceInstance,
    StageReport,
)


class MLReport(AbstractReport):
    """
    MLReport manages the reporting of machine learning metrics and inference data.

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
    previous_report : list[GeneralMlInferenceReport]
        A list of previously processed reports.
    report : GeneralMlInferenceReport
        The current state of the report.

    Methods
    -------
    reset() -> None
        Reset the report to an initial state.
    combine_stage_report(current_stage_report: dict[str, StageReport], previous_stage_report: dict[str, StageReport]) -> dict[str, StageReport]
        Combine metrics from the current and previous stage reports.
    process_previous_report(previous_report_dict: dict) -> None
        Process and incorporate a previous report.
    observe_metric(report_type: ReportTypeEnum, stage: str, metric: Metric) -> None
        Observe and record a metric.
    observe_inference(inference_value: Any) -> None
        Observe and record inference data.
    observe_inference_metric(metric: Metric) -> None
        Observe and record an inference-specific metric.
    generate_report(reset: bool = True, corr_id: Optional[str] = None) -> BaseReport
        Generate the report and optionally reset the current report state.
    """

    def __init__(self, client_config: ClientInfo) -> None:
        """
        Initialize an instance of MLReport.

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
        - This method initializes a new report model and clears the list of previous reports.
        """
        self.previous_report: list[GeneralMlInferenceReport] = []
        self.report = GeneralMlInferenceReport()

    def combine_stage_report(
        self,
        current_stage_report: dict[str, StageReport],
        previous_stage_report: dict[str, StageReport],
    ) -> dict[str, StageReport]:
        """
        Combine metrics from the current and previous stage reports.

        Parameters
        ----------
        current_stage_report : dict
            The current stage report containing metrics.
        previous_stage_report : dict
            The previous stage report containing metrics.

        Returns
        -------
        dict
            Combined stage report containing metrics from both reports.
        """
        combined_stage_report: dict[str, StageReport] = {}
        for stage_name, stage_report in previous_stage_report.items():
            new_stage_report = StageReport(name=stage_name, metrics={})
            if stage_name not in current_stage_report:
                current_stage_report[stage_name] = StageReport(
                    name=stage_name, metrics={}
                )
            for metric_name, instance_report_dict in stage_report.metrics.items():
                if metric_name not in current_stage_report[stage_name].metrics:
                    current_stage_report[stage_name].metrics[metric_name] = {}
                new_stage_report.metrics[metric_name] = {
                    **current_stage_report[stage_name].metrics[metric_name],
                    **instance_report_dict,
                }
            combined_stage_report[stage_name] = new_stage_report
        return combined_stage_report

    def process_previous_report(self, previous_report_dict: dict) -> None:
        """
        Process and incorporate a previous report.

        Parameters
        ----------
        previous_report_dict : dict
            Dictionary representation of a previous report.

        Notes
        -----
        - Service quality, data quality, and ML inference reports are combined with the current report.
        """
        previous_report = GeneralMlInferenceReport(**previous_report_dict)
        self.previous_report.append(previous_report)

        self.report.service = self.combine_stage_report(
            self.report.service, previous_report.service
        )
        self.report.data = self.combine_stage_report(
            self.report.data, previous_report.data
        )
        self.report.ml_inference |= previous_report.ml_inference

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

        Raises
        ------
        ValueError
            If the stage name is empty or the report type is not handled.
        """
        if stage == "":
            raise ValueError("Stage name can't be empty")

        report_dict = None
        if report_type == ReportTypeEnum.service:
            if stage not in self.report.service:
                self.report.service[stage] = StageReport(name=stage, metrics={})
            report_dict = self.report.service[stage].metrics
        elif report_type == ReportTypeEnum.data:
            if stage not in self.report.data:
                self.report.data[stage] = StageReport(name=stage, metrics={})
            report_dict = self.report.data[stage].metrics
        else:
            raise ValueError(f"Can't handle report type {report_type}")

        if metric.metric_name not in report_dict:
            report_dict[metric.metric_name] = {}

        report_dict[metric.metric_name][UUID(self.client_config.instance_id)] = metric

    def observe_inference(self, inference_value: Any) -> None:
        """
        Observe and record inference data.

        Parameters
        ----------
        inference_value : Any
            The value of the inference to be recorded.

        Notes
        -----
        - Raises a warning if inference data already exists for the current instance.
        """
        instance_id = UUID(self.client_config.instance_id)

        if instance_id in self.report.ml_inference:
            raise RuntimeWarning(
                "Inference existed, will override the existing inference"
            )

        self.report.ml_inference[instance_id] = InferenceInstance(
            inference_id=uuid4(),
            instance_id=instance_id,
            functionality=self.client_config.functionality,
            prediction=inference_value,
        )

    def observe_inference_metric(self, metric: Metric) -> None:
        """
        Observe and record an inference-specific metric.

        Parameters
        ----------
        metric : Metric
            The inference-specific metric to be recorded.
        """
        instance_id = UUID(self.client_config.instance_id)

        if instance_id in self.report.ml_inference:
            self.report.ml_inference[instance_id].metrics.append(metric)
        else:
            self.report.ml_inference[instance_id] = InferenceInstance(
                inference_id=uuid4(),
                instance_id=instance_id,
                functionality=self.client_config.functionality,
                metrics=[metric],
            )

    def generate_report(
        self, reset: bool = True, corr_id: Optional[str] = None
    ) -> BaseReport:
        """
        Generate the report and optionally reset the current report state.

        Parameters
        ----------
        reset : bool, optional
            Whether to reset the report state after generating the report, default is True.
        corr_id : Optional[str], optional
            Correlation ID for the report, default is None.

        Returns
        -------
        BaseReport
            The generated report.

        Notes
        -----
        - Adds metadata such as client configuration, timestamp, and runtime to the report.
        - Deep copies the current state of the report before optionally resetting it.
        """
        self.report.metadata["client_config"] = copy.deepcopy(self.client_config)
        self.report.metadata["timestamp"] = time.time()
        if corr_id is not None:
            self.report.metadata["corr_id"] = corr_id
        self.report.metadata["runtime"] = (
            self.report.metadata["timestamp"] - self.init_time
        )

        report = copy.deepcopy(self.report)
        if reset:
            self.reset()
        return report
