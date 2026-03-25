from uuid import UUID, uuid4

import pytest

from qoa4ml.config.configs import ClientInfo
from qoa4ml.lang.common_models import Metric
from qoa4ml.lang.datamodel_enum import ReportTypeEnum
from qoa4ml.reports.ml_report_model import (
    GeneralMlInferenceReport,
    InferenceInstance,
    StageReport,
)
from qoa4ml.reports.ml_reports import MLReport


@pytest.fixture()
def instance_id():
    return str(uuid4())


@pytest.fixture()
def client_config(instance_id):
    return ClientInfo(
        name="test-client",
        instance_id=instance_id,
        functionality="classification",
        stage_id="inference",
    )


@pytest.fixture()
def ml_report(client_config):
    return MLReport(client_config)


class TestMLReportCreation:
    def test_init_stores_deep_copy_of_config(self, client_config):
        report = MLReport(client_config)
        assert report.client_config == client_config
        assert report.client_config is not client_config

    def test_init_creates_empty_report(self, ml_report):
        assert isinstance(ml_report.report, GeneralMlInferenceReport)
        assert ml_report.report.service == {}
        assert ml_report.report.data == {}
        assert ml_report.report.ml_inference == {}

    def test_init_sets_init_time(self, ml_report):
        assert ml_report.init_time > 0

    def test_init_creates_empty_previous_reports(self, ml_report):
        assert ml_report.previous_report == []


class TestObserveMetric:
    def test_observe_service_metric(self, ml_report):
        metric = Metric(metric_name="response_time", records=[1.5])
        ml_report.observe_metric(ReportTypeEnum.service, "gateway", metric)

        assert "gateway" in ml_report.report.service
        stage_report = ml_report.report.service["gateway"]
        assert stage_report.name == "gateway"
        assert "response_time" in stage_report.metrics
        instance_uuid = UUID(ml_report.client_config.instance_id)
        assert instance_uuid in stage_report.metrics["response_time"]
        assert stage_report.metrics["response_time"][instance_uuid] == metric

    def test_observe_data_metric(self, ml_report):
        metric = Metric(metric_name="completeness", records=[0.95])
        ml_report.observe_metric(ReportTypeEnum.data, "preprocessing", metric)

        assert "preprocessing" in ml_report.report.data
        stage_report = ml_report.report.data["preprocessing"]
        assert stage_report.name == "preprocessing"
        assert "completeness" in stage_report.metrics

    def test_observe_metric_empty_stage_raises(self, ml_report):
        metric = Metric(metric_name="accuracy", records=[0.9])
        with pytest.raises(ValueError, match="Stage name can't be empty"):
            ml_report.observe_metric(ReportTypeEnum.service, "", metric)

    def test_observe_metric_invalid_type_raises(self, ml_report):
        metric = Metric(metric_name="accuracy", records=[0.9])
        with pytest.raises(ValueError, match="Can't handle report type"):
            ml_report.observe_metric(ReportTypeEnum.ml_specific, "stage1", metric)

    def test_observe_multiple_metrics_same_stage(self, ml_report):
        metric1 = Metric(metric_name="response_time", records=[1.0])
        metric2 = Metric(metric_name="throughput", records=[100])
        ml_report.observe_metric(ReportTypeEnum.service, "gateway", metric1)
        ml_report.observe_metric(ReportTypeEnum.service, "gateway", metric2)

        stage_report = ml_report.report.service["gateway"]
        assert "response_time" in stage_report.metrics
        assert "throughput" in stage_report.metrics

    def test_observe_metrics_different_stages(self, ml_report):
        metric1 = Metric(metric_name="latency", records=[0.5])
        metric2 = Metric(metric_name="accuracy", records=[0.99])
        ml_report.observe_metric(ReportTypeEnum.service, "stage_a", metric1)
        ml_report.observe_metric(ReportTypeEnum.service, "stage_b", metric2)

        assert "stage_a" in ml_report.report.service
        assert "stage_b" in ml_report.report.service


class TestObserveInferenceMetric:
    def test_observe_inference_metric_creates_instance(self, ml_report):
        metric = Metric(metric_name="confidence", records=[0.95])
        ml_report.observe_inference_metric(metric)

        instance_id = UUID(ml_report.client_config.instance_id)
        assert instance_id in ml_report.report.ml_inference
        inf_instance = ml_report.report.ml_inference[instance_id]
        assert isinstance(inf_instance, InferenceInstance)
        assert len(inf_instance.metrics) == 1
        assert inf_instance.metrics[0] == metric

    def test_observe_inference_metric_appends_to_existing(self, ml_report):
        ml_report.observe_inference({"class": "cat", "score": 0.9})

        metric = Metric(metric_name="confidence", records=[0.95])
        ml_report.observe_inference_metric(metric)

        instance_id = UUID(ml_report.client_config.instance_id)
        inf_instance = ml_report.report.ml_inference[instance_id]
        assert len(inf_instance.metrics) == 1
        assert inf_instance.metrics[0] == metric

    def test_observe_multiple_inference_metrics(self, ml_report):
        m1 = Metric(metric_name="confidence", records=[0.95])
        m2 = Metric(metric_name="latency", records=[0.1])
        ml_report.observe_inference_metric(m1)
        ml_report.observe_inference_metric(m2)

        instance_id = UUID(ml_report.client_config.instance_id)
        assert len(ml_report.report.ml_inference[instance_id].metrics) == 2


class TestObserveInference:
    def test_observe_inference_stores_prediction(self, ml_report):
        ml_report.observe_inference({"class": "dog", "score": 0.85})

        instance_id = UUID(ml_report.client_config.instance_id)
        inf = ml_report.report.ml_inference[instance_id]
        assert inf.prediction == {"class": "dog", "score": 0.85}
        assert inf.functionality == "classification"

    def test_observe_inference_override_warns(self, ml_report):
        ml_report.observe_inference({"class": "cat"})
        with pytest.warns(RuntimeWarning, match="Inference existed"):
            ml_report.observe_inference({"class": "dog"})


class TestGenerateReport:
    def test_generate_report_returns_base_report(self, ml_report):
        metric = Metric(metric_name="accuracy", records=[0.9])
        ml_report.observe_metric(ReportTypeEnum.service, "inference", metric)
        report = ml_report.generate_report()

        assert "client_config" in report.metadata
        assert "timestamp" in report.metadata
        assert "runtime" in report.metadata

    def test_generate_report_with_corr_id(self, ml_report):
        report = ml_report.generate_report(corr_id="req-123")
        assert report.metadata["corr_id"] == "req-123"

    def test_generate_report_without_corr_id(self, ml_report):
        report = ml_report.generate_report()
        assert "corr_id" not in report.metadata

    def test_generate_report_resets_by_default(self, ml_report):
        metric = Metric(metric_name="accuracy", records=[0.9])
        ml_report.observe_metric(ReportTypeEnum.service, "inference", metric)
        ml_report.generate_report(reset=True)

        assert ml_report.report.service == {}
        assert ml_report.report.data == {}
        assert ml_report.report.ml_inference == {}
        assert ml_report.previous_report == []

    def test_generate_report_no_reset_preserves_state(self, ml_report):
        metric = Metric(metric_name="accuracy", records=[0.9])
        ml_report.observe_metric(ReportTypeEnum.service, "inference", metric)
        ml_report.generate_report(reset=False)

        assert "inference" in ml_report.report.service

    def test_generate_report_returns_deep_copy(self, ml_report):
        metric = Metric(metric_name="accuracy", records=[0.9])
        ml_report.observe_metric(ReportTypeEnum.service, "inference", metric)
        report = ml_report.generate_report(reset=False)

        assert report.service is not ml_report.report.service

    def test_runtime_is_positive(self, ml_report):
        report = ml_report.generate_report()
        assert report.metadata["runtime"] >= 0


class TestReset:
    def test_reset_clears_report(self, ml_report):
        metric = Metric(metric_name="accuracy", records=[0.9])
        ml_report.observe_metric(ReportTypeEnum.service, "inference", metric)
        ml_report.observe_inference({"class": "cat"})

        ml_report.reset()

        assert ml_report.report.service == {}
        assert ml_report.report.data == {}
        assert ml_report.report.ml_inference == {}
        assert ml_report.previous_report == []


class TestCombineStageReport:
    def test_combine_merges_metrics(self, ml_report):
        uuid1 = uuid4()
        uuid2 = uuid4()
        metric1 = Metric(metric_name="accuracy", records=[0.9])
        metric2 = Metric(metric_name="accuracy", records=[0.85])

        current = {
            "stage1": StageReport(name="stage1", metrics={"accuracy": {uuid1: metric1}})
        }
        previous = {
            "stage1": StageReport(name="stage1", metrics={"accuracy": {uuid2: metric2}})
        }

        combined = ml_report.combine_stage_report(current, previous)
        assert uuid1 in combined["stage1"].metrics["accuracy"]
        assert uuid2 in combined["stage1"].metrics["accuracy"]

    def test_combine_adds_missing_stage_to_current(self, ml_report):
        uuid1 = uuid4()
        metric = Metric(metric_name="latency", records=[0.5])

        current = {}
        previous = {
            "new_stage": StageReport(
                name="new_stage", metrics={"latency": {uuid1: metric}}
            )
        }

        combined = ml_report.combine_stage_report(current, previous)
        assert "new_stage" in combined
        assert uuid1 in combined["new_stage"].metrics["latency"]
