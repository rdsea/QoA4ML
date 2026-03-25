from uuid import UUID

import pytest

from qoa4ml.config.configs import ClientInfo
from qoa4ml.lang.common_models import Metric
from qoa4ml.lang.datamodel_enum import ReportTypeEnum
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
from qoa4ml.reports.rohe_reports import RoheReport

INSTANCE_ID = "b6f83293-cf67-44dd-a7b5-77229d384012"


@pytest.fixture
def client_info():
    return ClientInfo(
        name="test_client",
        instance_id=INSTANCE_ID,
        stage_id="gateway",
        functionality="REST",
        user_id="user1",
        application_name="test_app",
    )


@pytest.fixture
def rohe_report(client_info):
    return RoheReport(client_config=client_info)


@pytest.fixture
def accuracy_metric():
    return Metric(metric_name="accuracy", records=[0.95])


@pytest.fixture
def latency_metric():
    return Metric(metric_name="response_time", records=[120.5])


class TestRoheReportInit:
    def test_init_sets_client_config(self, rohe_report, client_info):
        assert rohe_report.client_config.name == client_info.name
        assert rohe_report.client_config.instance_id == client_info.instance_id

    def test_init_deep_copies_config(self, rohe_report, client_info):
        assert rohe_report.client_config is not client_info
        client_info.name = "modified"
        assert rohe_report.client_config.name == "test_client"

    def test_init_sets_init_time(self, rohe_report):
        assert isinstance(rohe_report.init_time, float)
        assert rohe_report.init_time > 0

    def test_init_creates_empty_report(self, rohe_report):
        assert isinstance(rohe_report.report, RoheReportModel)
        assert isinstance(rohe_report.inference_report, EnsembleInferenceReport)
        assert isinstance(rohe_report.execution_graph, ExecutionGraph)

    def test_init_creates_execution_instance(self, rohe_report):
        assert isinstance(rohe_report.execution_instance, MicroserviceInstance)
        assert rohe_report.execution_instance.id == UUID(INSTANCE_ID)
        assert rohe_report.execution_instance.name == "test_client"
        assert rohe_report.execution_instance.functionality == "REST"
        assert rohe_report.execution_instance.stage == "gateway"


class TestReset:
    def test_reset_clears_previous_reports(self, rohe_report):
        rohe_report.previous_report.append(RoheReportModel())
        rohe_report.reset()
        assert rohe_report.previous_report == []

    def test_reset_creates_fresh_inference_report(self, rohe_report, accuracy_metric):
        rohe_report.observe_metric(ReportTypeEnum.service, "stage1", accuracy_metric)
        rohe_report.reset()
        assert rohe_report.inference_report.service == {}
        assert rohe_report.inference_report.data == {}
        assert rohe_report.inference_report.ml_specific is None

    def test_reset_clears_execution_graph(self, rohe_report):
        rohe_report.reset()
        assert rohe_report.execution_graph.linked_list == {}
        assert rohe_report.execution_graph.end_point is None

    def test_reset_clears_previous_microservice_instances(self, rohe_report):
        rohe_report.previous_microservice_instance.append(
            MicroserviceInstance(id=UUID(INSTANCE_ID), name="old")
        )
        rohe_report.reset()
        assert rohe_report.previous_microservice_instance == []

    def test_reset_preserves_execution_instance_identity(self, rohe_report):
        rohe_report.reset()
        assert rohe_report.execution_instance.id == UUID(INSTANCE_ID)
        assert rohe_report.execution_instance.name == "test_client"


class TestObserveMetric:
    def test_observe_service_metric(self, rohe_report, accuracy_metric):
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", accuracy_metric)
        stage_report = rohe_report.inference_report.service["gateway"]
        assert "accuracy" in stage_report.metrics
        assert UUID(INSTANCE_ID) in stage_report.metrics["accuracy"]
        assert stage_report.metrics["accuracy"][UUID(INSTANCE_ID)] == accuracy_metric

    def test_observe_data_metric(self, rohe_report, accuracy_metric):
        rohe_report.observe_metric(ReportTypeEnum.data, "ingestion", accuracy_metric)
        stage_report = rohe_report.inference_report.data["ingestion"]
        assert "accuracy" in stage_report.metrics
        assert UUID(INSTANCE_ID) in stage_report.metrics["accuracy"]

    def test_observe_metric_creates_stage_if_missing(
        self, rohe_report, accuracy_metric
    ):
        rohe_report.observe_metric(ReportTypeEnum.service, "new_stage", accuracy_metric)
        assert "new_stage" in rohe_report.inference_report.service
        assert rohe_report.inference_report.service["new_stage"].name == "new_stage"

    def test_observe_metric_empty_stage_raises(self, rohe_report, accuracy_metric):
        with pytest.raises(ValueError, match="Stage name can't be empty"):
            rohe_report.observe_metric(ReportTypeEnum.service, "", accuracy_metric)

    def test_observe_metric_invalid_report_type_raises(
        self, rohe_report, accuracy_metric
    ):
        with pytest.raises(ValueError, match="Can't handle report type"):
            rohe_report.observe_metric(
                ReportTypeEnum.ml_specific, "stage1", accuracy_metric
            )

    def test_observe_multiple_metrics_same_stage(
        self, rohe_report, accuracy_metric, latency_metric
    ):
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", accuracy_metric)
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", latency_metric)
        stage_metrics = rohe_report.inference_report.service["gateway"].metrics
        assert "accuracy" in stage_metrics
        assert "response_time" in stage_metrics

    def test_observe_metric_updates_report(self, rohe_report, accuracy_metric):
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", accuracy_metric)
        assert rohe_report.report.inference_report is rohe_report.inference_report


class TestObserveInference:
    def test_observe_inference_creates_ml_specific(self, rohe_report):
        rohe_report.observe_inference({"class": "cat", "confidence": 0.95})
        ml_specific = rohe_report.inference_report.ml_specific
        assert ml_specific is not None
        assert isinstance(ml_specific, InferenceGraph)
        assert ml_specific.end_point is not None
        assert ml_specific.end_point.prediction == {
            "class": "cat",
            "confidence": 0.95,
        }

    def test_observe_inference_sets_instance_id(self, rohe_report):
        rohe_report.observe_inference(0.42)
        end_point = rohe_report.inference_report.ml_specific.end_point
        assert end_point.instance_id == UUID(INSTANCE_ID)
        assert end_point.functionality == "REST"

    def test_observe_inference_creates_linked_list_entry(self, rohe_report):
        rohe_report.observe_inference(0.42)
        ml_specific = rohe_report.inference_report.ml_specific
        assert ml_specific.end_point.instance_id in ml_specific.linked_list
        linked = ml_specific.linked_list[ml_specific.end_point.instance_id]
        assert linked.previous == []

    def test_observe_inference_updates_existing_prediction(self, rohe_report):
        rohe_report.observe_inference(0.42)
        rohe_report.observe_inference(0.99)
        assert rohe_report.inference_report.ml_specific.end_point.prediction == 0.99

    def test_observe_inference_with_float(self, rohe_report):
        rohe_report.observe_inference(3.14)
        assert rohe_report.inference_report.ml_specific.end_point.prediction == 3.14

    def test_observe_inference_with_none(self, rohe_report):
        rohe_report.observe_inference(None)
        assert rohe_report.inference_report.ml_specific.end_point.prediction is None


class TestObserveInferenceMetric:
    def test_observe_inference_metric_creates_endpoint(
        self, rohe_report, accuracy_metric
    ):
        rohe_report.observe_inference_metric(accuracy_metric)
        ml_specific = rohe_report.inference_report.ml_specific
        assert ml_specific is not None
        assert ml_specific.end_point is not None
        assert accuracy_metric in ml_specific.end_point.metrics

    def test_observe_inference_metric_appends_to_existing(
        self, rohe_report, accuracy_metric, latency_metric
    ):
        rohe_report.observe_inference(0.42)
        rohe_report.observe_inference_metric(accuracy_metric)
        rohe_report.observe_inference_metric(latency_metric)
        metrics = rohe_report.inference_report.ml_specific.end_point.metrics
        assert len(metrics) == 2
        assert accuracy_metric in metrics
        assert latency_metric in metrics

    def test_observe_inference_metric_without_prior_inference(
        self, rohe_report, accuracy_metric
    ):
        rohe_report.observe_inference_metric(accuracy_metric)
        end_point = rohe_report.inference_report.ml_specific.end_point
        assert end_point.instance_id == UUID(INSTANCE_ID)
        assert end_point.functionality == "REST"
        assert end_point.prediction is None

    def test_observe_inference_metric_linked_list_entry(
        self, rohe_report, accuracy_metric
    ):
        rohe_report.observe_inference_metric(accuracy_metric)
        ml_specific = rohe_report.inference_report.ml_specific
        assert ml_specific.end_point.instance_id in ml_specific.linked_list


class TestBuildExecutionGraph:
    def test_build_execution_graph_sets_endpoint(self, rohe_report):
        rohe_report.build_execution_graph()
        assert rohe_report.execution_graph.end_point is not None
        assert rohe_report.execution_graph.end_point.id == UUID(INSTANCE_ID)
        assert rohe_report.execution_graph.end_point.name == "test_client"

    def test_build_execution_graph_adds_linked_instance(self, rohe_report):
        rohe_report.build_execution_graph()
        linked_list = rohe_report.execution_graph.linked_list
        assert UUID(INSTANCE_ID) in linked_list
        linked_instance = linked_list[UUID(INSTANCE_ID)]
        assert linked_instance.instance == rohe_report.execution_instance
        assert linked_instance.previous == []

    def test_build_execution_graph_with_previous_instances(self, rohe_report):
        prev_instance = MicroserviceInstance(
            id=UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
            name="previous_service",
        )
        rohe_report.previous_microservice_instance.append(prev_instance)
        rohe_report.build_execution_graph()
        linked = rohe_report.execution_graph.linked_list[UUID(INSTANCE_ID)]
        assert prev_instance in linked.previous

    def test_build_execution_graph_updates_report(self, rohe_report):
        rohe_report.build_execution_graph()
        assert rohe_report.report.execution_graph is rohe_report.execution_graph


class TestGenerateReport:
    def test_generate_report_returns_rohe_report_model(self, rohe_report):
        report = rohe_report.generate_report()
        assert isinstance(report, RoheReportModel)

    def test_generate_report_includes_metadata(self, rohe_report):
        report = rohe_report.generate_report()
        assert "client_config" in report.metadata
        assert "timestamp" in report.metadata
        assert "runtime" in report.metadata

    def test_generate_report_metadata_client_config(self, rohe_report):
        report = rohe_report.generate_report()
        config = report.metadata["client_config"]
        assert config.name == "test_client"
        assert config.instance_id == INSTANCE_ID

    def test_generate_report_runtime_is_positive(self, rohe_report):
        report = rohe_report.generate_report()
        assert report.metadata["runtime"] >= 0

    def test_generate_report_with_corr_id(self, rohe_report):
        report = rohe_report.generate_report(corr_id="test-correlation-123")
        assert report.metadata["corr_id"] == "test-correlation-123"

    def test_generate_report_without_corr_id(self, rohe_report):
        report = rohe_report.generate_report()
        assert "corr_id" not in report.metadata

    def test_generate_report_resets_by_default(self, rohe_report, accuracy_metric):
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", accuracy_metric)
        rohe_report.generate_report(reset=True)
        assert rohe_report.inference_report.service == {}
        assert rohe_report.previous_report == []

    def test_generate_report_no_reset(self, rohe_report, accuracy_metric):
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", accuracy_metric)
        rohe_report.generate_report(reset=False)
        assert "gateway" in rohe_report.inference_report.service

    def test_generate_report_returns_deep_copy(self, rohe_report, accuracy_metric):
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", accuracy_metric)
        report = rohe_report.generate_report(reset=False)
        assert report is not rohe_report.report

    def test_generate_report_builds_execution_graph(self, rohe_report):
        report = rohe_report.generate_report()
        assert report.execution_graph is not None
        assert report.execution_graph.end_point is not None
        assert report.execution_graph.end_point.id == UUID(INSTANCE_ID)

    def test_generate_report_includes_observed_metrics(
        self, rohe_report, accuracy_metric
    ):
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", accuracy_metric)
        report = rohe_report.generate_report()
        stage = report.inference_report.service["gateway"]
        assert UUID(INSTANCE_ID) in stage.metrics["accuracy"]


class TestProcessPreviousReport:
    @pytest.fixture
    def previous_report_dict(self):
        prev_instance_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        prev_instance = MicroserviceInstance(
            id=prev_instance_id,
            name="prev_service",
            functionality="ML",
            stage="inference",
        )
        execution_graph = ExecutionGraph(
            end_point=prev_instance,
            linked_list={
                prev_instance_id: LinkedInstance[MicroserviceInstance](
                    instance=prev_instance,
                    previous=[],
                )
            },
        )
        service_stage = StageReport(
            name="inference",
            metrics={
                "accuracy": {
                    prev_instance_id: Metric(metric_name="accuracy", records=[0.92])
                }
            },
        )
        inference_report = EnsembleInferenceReport(
            service={"inference": service_stage},
            data={},
        )
        report = RoheReportModel(
            inference_report=inference_report,
            execution_graph=execution_graph,
        )
        return report.model_dump(mode="json")

    def test_process_previous_report_appends_to_list(
        self, rohe_report, previous_report_dict
    ):
        rohe_report.process_previous_report(previous_report_dict)
        assert len(rohe_report.previous_report) == 1

    def test_process_previous_report_merges_service_metrics(
        self, rohe_report, previous_report_dict
    ):
        rohe_report.process_previous_report(previous_report_dict)
        assert "inference" in rohe_report.inference_report.service

    def test_process_previous_report_updates_execution_graph(
        self, rohe_report, previous_report_dict
    ):
        rohe_report.process_previous_report(previous_report_dict)
        prev_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert prev_id in rohe_report.execution_graph.linked_list

    def test_process_previous_report_adds_microservice_instance(
        self, rohe_report, previous_report_dict
    ):
        rohe_report.process_previous_report(previous_report_dict)
        assert len(rohe_report.previous_microservice_instance) == 1
        assert rohe_report.previous_microservice_instance[0].id == UUID(
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        )

    def test_process_empty_previous_report_raises(self, rohe_report):
        empty_report = RoheReportModel().model_dump()
        with pytest.raises(ValueError, match="Can't process empty previous report"):
            rohe_report.process_previous_report(empty_report)

    def test_process_multiple_previous_reports(self, rohe_report, previous_report_dict):
        second_id = UUID("11111111-2222-3333-4444-555555555555")
        second_instance = MicroserviceInstance(
            id=second_id, name="second_service", functionality="DL", stage="preprocess"
        )
        second_graph = ExecutionGraph(
            end_point=second_instance,
            linked_list={
                second_id: LinkedInstance[MicroserviceInstance](
                    instance=second_instance, previous=[]
                )
            },
        )
        second_report = RoheReportModel(
            inference_report=EnsembleInferenceReport(
                service={
                    "preprocess": StageReport(
                        name="preprocess",
                        metrics={
                            "response_time": {
                                second_id: Metric(
                                    metric_name="response_time", records=[50.0]
                                )
                            }
                        },
                    )
                },
                data={},
            ),
            execution_graph=second_graph,
        )

        rohe_report.process_previous_report(previous_report_dict)
        rohe_report.process_previous_report(second_report.model_dump(mode="json"))

        assert len(rohe_report.previous_report) == 2
        assert len(rohe_report.previous_microservice_instance) == 2
        prev_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert prev_id in rohe_report.execution_graph.linked_list
        assert second_id in rohe_report.execution_graph.linked_list

    def test_process_previous_report_with_ml_specific(self, rohe_report):
        prev_instance_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        prev_instance = MicroserviceInstance(id=prev_instance_id, name="prev_service")
        prev_inference_instance = InferenceInstance(
            inference_id=UUID("22222222-3333-4444-5555-666666666666"),
            instance_id=prev_instance_id,
            functionality="ML",
            prediction=0.85,
        )
        inference_graph = InferenceGraph(
            end_point=prev_inference_instance,
            linked_list={
                prev_instance_id: LinkedInstance[InferenceInstance](
                    instance=prev_inference_instance, previous=[]
                )
            },
        )
        report = RoheReportModel(
            inference_report=EnsembleInferenceReport(
                service={}, data={}, ml_specific=inference_graph
            ),
            execution_graph=ExecutionGraph(
                end_point=prev_instance,
                linked_list={
                    prev_instance_id: LinkedInstance[MicroserviceInstance](
                        instance=prev_instance, previous=[]
                    )
                },
            ),
        )
        rohe_report.process_previous_report(report.model_dump(mode="json"))
        ml_specific = rohe_report.inference_report.ml_specific
        assert ml_specific is not None
        assert ml_specific.end_point is not None
        assert ml_specific.end_point.instance_id == UUID(INSTANCE_ID)


class TestCombineStageReport:
    def test_combine_with_empty_current(self, rohe_report):
        instance_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        previous = {
            "stage1": StageReport(
                name="stage1",
                metrics={
                    "accuracy": {
                        instance_id: Metric(metric_name="accuracy", records=[0.9])
                    }
                },
            )
        }
        current = {}
        result = rohe_report.combine_stage_report(current, previous)
        assert "stage1" in result
        assert instance_id in result["stage1"].metrics["accuracy"]

    def test_combine_merges_metrics(self, rohe_report):
        id1 = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        id2 = UUID(INSTANCE_ID)
        previous = {
            "stage1": StageReport(
                name="stage1",
                metrics={
                    "accuracy": {id1: Metric(metric_name="accuracy", records=[0.9])}
                },
            )
        }
        current = {
            "stage1": StageReport(
                name="stage1",
                metrics={
                    "accuracy": {id2: Metric(metric_name="accuracy", records=[0.95])}
                },
            )
        }
        result = rohe_report.combine_stage_report(current, previous)
        assert id1 in result["stage1"].metrics["accuracy"]
        assert id2 in result["stage1"].metrics["accuracy"]

    def test_combine_with_empty_previous(self, rohe_report):
        current = {
            "stage1": StageReport(
                name="stage1",
                metrics={
                    "accuracy": {
                        UUID(INSTANCE_ID): Metric(
                            metric_name="accuracy", records=[0.95]
                        )
                    }
                },
            )
        }
        result = rohe_report.combine_stage_report(current, {})
        assert result == {}

    def test_combine_multiple_stages(self, rohe_report):
        id1 = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        previous = {
            "stage1": StageReport(
                name="stage1",
                metrics={
                    "accuracy": {id1: Metric(metric_name="accuracy", records=[0.9])}
                },
            ),
            "stage2": StageReport(
                name="stage2",
                metrics={
                    "response_time": {
                        id1: Metric(metric_name="response_time", records=[100])
                    }
                },
            ),
        }
        result = rohe_report.combine_stage_report({}, previous)
        assert "stage1" in result
        assert "stage2" in result


class TestEdgeCases:
    def test_generate_report_with_no_observations(self, rohe_report):
        report = rohe_report.generate_report()
        assert report.inference_report is None
        assert report.execution_graph is not None
        assert report.execution_graph.end_point is not None

    def test_multiple_generate_report_cycles(self, rohe_report, accuracy_metric):
        rohe_report.observe_metric(ReportTypeEnum.service, "s1", accuracy_metric)
        report1 = rohe_report.generate_report(reset=True)
        assert "s1" in report1.inference_report.service

        rohe_report.observe_metric(ReportTypeEnum.data, "d1", accuracy_metric)
        report2 = rohe_report.generate_report(reset=True)
        assert "d1" in report2.inference_report.data
        assert report2.inference_report.service == {}

    def test_observe_same_metric_overwrites(self, rohe_report):
        m1 = Metric(metric_name="accuracy", records=[0.9])
        m2 = Metric(metric_name="accuracy", records=[0.99])
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", m1)
        rohe_report.observe_metric(ReportTypeEnum.service, "gateway", m2)
        stored = rohe_report.inference_report.service["gateway"].metrics["accuracy"]
        assert stored[UUID(INSTANCE_ID)].records == [0.99]

    def test_observe_metric_across_different_stages(self, rohe_report, accuracy_metric):
        rohe_report.observe_metric(ReportTypeEnum.service, "stage_a", accuracy_metric)
        rohe_report.observe_metric(ReportTypeEnum.service, "stage_b", accuracy_metric)
        assert "stage_a" in rohe_report.inference_report.service
        assert "stage_b" in rohe_report.inference_report.service
