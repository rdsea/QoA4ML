import copy
import time
from typing import Any, Optional
from uuid import UUID, uuid4

from qoa4ml.config.configs import ClientInfo
from qoa4ml.lang.common_models import Metric
from qoa4ml.lang.datamodel_enum import ReportTypeEnum
from qoa4ml.reports.abstract_report import AbstractReport
from qoa4ml.reports.ml_report_model import (
    EnsembleInferenceReport,
    ExecutionGraph,
    InferenceGraph,
    InferenceInstance,
    LinkedInstance,
    MicroserviceInstance,
    RoheReportModel,
    StageReport,
)
from qoa4ml.utils.qoa_utils import load_config


class RoheReport(AbstractReport):
    """
    RoheReport manages the reporting of metrics and inference data for a specific application.

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
    previous_report : list[RoheReportModel]
        A list of previously processed reports.
    inference_report : EnsembleInferenceReport
        The current inference report.
    execution_graph : ExecutionGraph
        The current execution graph.
    report : RoheReportModel
        The current state of the report.
    previous_microservice_instance : list[MicroserviceInstance]
        List of previous microservice instances.
    execution_instance : MicroserviceInstance
        An instance representing the current execution context.

    Methods
    -------
    reset() -> None
        Reset the report to an initial state.
    import_report_from_file(file_path: str) -> None
        Import a report from a specified file path.
    combine_stage_report(current_stage_report: dict[str, StageReport], previous_stage_report: dict[str, StageReport]) -> dict[str, StageReport]
        Combine metrics from the current and previous stage reports.
    process_previous_report(previous_report_dict: dict) -> None
        Process and incorporate a previous report.
    build_execution_graph() -> None
        Build the execution graph for the current report.
    observe_metric(report_type: ReportTypeEnum, stage: str, metric: Metric) -> None
        Observe and record a metric.
    observe_inference(inference_value: Any) -> None
        Observe and record inference data.
    observe_inference_metric(metric: Metric) -> None
        Observe and record an inference-specific metric.
    generate_report(reset: bool = True, corr_id: Optional[str] = None) -> RoheReportModel
        Generate the report and optionally reset the current report state.
    """

    def __init__(self, client_config: ClientInfo) -> None:
        """
        Initialize an instance of RoheReport.

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
        - It also resets the inference report, execution graph, and execution instance.
        """
        self.previous_report: list[RoheReportModel] = []
        self.inference_report = EnsembleInferenceReport()
        self.execution_graph = ExecutionGraph(linked_list={})
        self.report = RoheReportModel()
        self.previous_microservice_instance = []
        self.execution_instance = MicroserviceInstance(
            id=UUID(self.client_config.instance_id),
            name=self.client_config.name,
            functionality=self.client_config.functionality,
            stage=self.client_config.stage_id,
        )

    def import_report_from_file(self, file_path: str) -> None:
        """
        Import a report from a specified file path.

        Parameters
        ----------
        file_path : str
            The path to the file containing the report data.

        Notes
        -----
        - The imported report updates the current inference report and execution graph.
        """
        report = load_config(file_path)
        self.inference_report = EnsembleInferenceReport(**report["inference_report"])
        self.execution_graph = ExecutionGraph(**report["execution_graph"])
        self.report = RoheReportModel(
            inference_report=self.inference_report,
            execution_graph=self.execution_graph,
        )

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
        - Raises a ValueError if the previous report is empty.
        - Service quality, data quality, ML-specific quality reports, and execution graphs are combined with the current report.
        """
        previous_report = RoheReportModel(**previous_report_dict)
        self.previous_report.append(previous_report)
        if not previous_report.inference_report or not previous_report.execution_graph:
            raise ValueError("Can't process empty previous report")

        self.inference_report.service = self.combine_stage_report(
            self.inference_report.service, previous_report.inference_report.service
        )
        self.inference_report.data = self.combine_stage_report(
            self.inference_report.data, previous_report.inference_report.data
        )

        if not self.inference_report.ml_specific:
            if previous_report.inference_report.ml_specific:
                self.inference_report.ml_specific = (
                    previous_report.inference_report.ml_specific
                )
                end_point = InferenceInstance(
                    inference_id=uuid4(),
                    functionality=self.client_config.functionality,
                    instance_id=self.execution_instance.id,
                )
                self.inference_report.ml_specific.end_point = end_point
                self.inference_report.ml_specific.linked_list[end_point.instance_id] = (
                    LinkedInstance(
                        instance=end_point,
                        previous=[
                            previous_report.inference_report.ml_specific.end_point
                        ],
                    )
                )
        else:
            if (
                previous_report.inference_report.ml_specific
                and self.inference_report.ml_specific.end_point
            ):
                self.inference_report.ml_specific.linked_list.update(
                    previous_report.inference_report.ml_specific.linked_list
                )
                current_end_point = self.inference_report.ml_specific.end_point
                previous_end_point = (
                    previous_report.inference_report.ml_specific.end_point
                )
                self.inference_report.ml_specific.linked_list[
                    current_end_point.instance_id
                ].previous.append(previous_end_point)

        if not self.execution_graph:
            self.execution_graph = previous_report.execution_graph
        else:
            self.execution_graph.linked_list.update(
                previous_report.execution_graph.linked_list
            )

        self.previous_microservice_instance.append(
            previous_report.execution_graph.end_point
        )
        self.report = RoheReportModel(
            inference_report=self.inference_report,
            execution_graph=self.execution_graph,
        )

    def build_execution_graph(self) -> None:
        """
        Build the execution graph for the current report.

        Notes
        -----
        - Creates a new endpoint in the execution graph linking to previous microservice instances.
        """
        end_point = LinkedInstance(
            instance=self.execution_instance,
            previous=self.previous_microservice_instance,
        )
        self.execution_graph.linked_list[end_point.instance.id] = end_point
        self.execution_graph.end_point = end_point.instance
        self.report.execution_graph = self.execution_graph

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
            if stage not in self.inference_report.service:
                self.inference_report.service[stage] = StageReport(
                    name=stage, metrics={}
                )
            report_dict = self.inference_report.service[stage].metrics
        elif report_type == ReportTypeEnum.data:
            if stage not in self.inference_report.data:
                self.inference_report.data[stage] = StageReport(name=stage, metrics={})
            report_dict = self.inference_report.data[stage].metrics
        else:
            raise ValueError(f"Can't handle report type {report_type}")

        if metric.metric_name not in report_dict:
            report_dict[metric.metric_name] = {}

        report_dict[metric.metric_name][UUID(self.client_config.instance_id)] = metric
        self.report.inference_report = self.inference_report

    def observe_inference(self, inference_value: Any) -> None:
        """
        Observe and record inference data.

        Parameters
        ----------
        inference_value : Any
            The value of the inference to be recorded.

        Notes
        -----
        - If an inference endpoint already exists, the prediction value is updated.
        - Otherwise, a new inference endpoint is created in the inference report.
        """
        if (
            self.inference_report.ml_specific
            and self.inference_report.ml_specific.end_point
        ):
            self.inference_report.ml_specific.end_point.prediction = inference_value
        else:
            self.inference_report.ml_specific = InferenceGraph()
            end_point = InferenceInstance(
                inference_id=uuid4(),
                instance_id=self.execution_instance.id,
                functionality=self.client_config.functionality,
                prediction=inference_value,
            )
            self.inference_report.ml_specific.end_point = end_point
            self.inference_report.ml_specific.linked_list[end_point.instance_id] = (
                LinkedInstance(
                    instance=end_point,
                    previous=[],
                )
            )

    def observe_inference_metric(self, metric: Metric) -> None:
        """
        Observe and record an inference-specific metric.

        Parameters
        ----------
        metric : Metric
            The inference-specific metric to be recorded.

        Notes
        -----
        - If an inference endpoint already exists, the metric is appended to the existing metrics.
        - Otherwise, a new inference endpoint is created and the metric is added to it.
        """
        if (
            self.inference_report.ml_specific
            and self.inference_report.ml_specific.end_point
        ):
            self.inference_report.ml_specific.end_point.metrics.append(metric)
        else:
            if not self.inference_report.ml_specific:
                self.inference_report.ml_specific = InferenceGraph()
            if not self.inference_report.ml_specific.end_point:
                end_point = InferenceInstance(
                    inference_id=uuid4(),
                    instance_id=self.execution_instance.id,
                    functionality=self.client_config.functionality,
                )
                self.inference_report.ml_specific.end_point = end_point
                self.inference_report.ml_specific.linked_list[end_point.instance_id] = (
                    LinkedInstance(
                        instance=end_point,
                    )
                )
            self.inference_report.ml_specific.end_point.metrics.append(metric)

    def generate_report(
        self, reset: bool = True, corr_id: Optional[str] = None
    ) -> RoheReportModel:
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
        RoheReportModel
            The generated report.

        Notes
        -----
        - Adds metadata such as client configuration, timestamp, and runtime to the report.
        - Builds the execution graph and deep copies the current state of the report before optionally resetting it.
        """
        self.build_execution_graph()
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
