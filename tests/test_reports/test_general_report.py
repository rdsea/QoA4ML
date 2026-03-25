import copy
import time
from uuid import uuid4

import pytest

from qoa4ml.config.configs import ClientInfo
from qoa4ml.lang.common_models import Metric
from qoa4ml.lang.datamodel_enum import ReportTypeEnum
from qoa4ml.reports.general_application_report import GeneralApplicationReport
from qoa4ml.reports.ml_report_model import (
    FlattenMetric,
    GeneralApplicationReportModel,
    MicroserviceInstance,
)


class ConcreteGeneralApplicationReport(GeneralApplicationReport):
    """Concrete subclass providing the missing generate_report implementation."""

    def generate_report(self, reset=True, corr_id=None):
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


@pytest.fixture()
def instance_id():
    return str(uuid4())


@pytest.fixture()
def client_config(instance_id):
    return ClientInfo(
        name="test-service",
        instance_id=instance_id,
        functionality="prediction",
        stage_id="inference",
    )


@pytest.fixture()
def app_report(client_config):
    return ConcreteGeneralApplicationReport(client_config)


class TestGeneralApplicationReportCreation:
    def test_init_stores_deep_copy_of_config(self, client_config):
        report = ConcreteGeneralApplicationReport(client_config)
        assert report.client_config == client_config
        assert report.client_config is not client_config

    def test_init_creates_empty_report(self, app_report):
        assert isinstance(app_report.report, GeneralApplicationReportModel)
        assert app_report.report.metrics == []

    def test_init_creates_execution_instance(self, app_report, instance_id):
        inst = app_report.execution_instance
        assert isinstance(inst, MicroserviceInstance)
        assert str(inst.id) == instance_id
        assert inst.name == "test-service"
        assert inst.functionality == "prediction"
        assert inst.stage == "inference"

    def test_init_empty_previous_reports(self, app_report):
        assert app_report.previous_reports == []

    def test_init_sets_init_time(self, app_report):
        assert app_report.init_time > 0


class TestObserveMetric:
    def test_observe_service_metric(self, app_report):
        metric = Metric(metric_name="response_time", records=[1.2])
        app_report.observe_metric(ReportTypeEnum.service, "gateway", metric)

        assert len(app_report.report.metrics) == 1
        fm = app_report.report.metrics[0]
        assert isinstance(fm, FlattenMetric)
        assert fm.metric_name == "response_time"
        assert fm.stage == "gateway"
        assert fm.report_type == ReportTypeEnum.service
        assert fm.records == [1.2]

    def test_observe_data_metric(self, app_report):
        metric = Metric(metric_name="completeness", records=[0.98])
        app_report.observe_metric(ReportTypeEnum.data, "preprocessing", metric)

        fm = app_report.report.metrics[0]
        assert fm.report_type == ReportTypeEnum.data
        assert fm.stage == "preprocessing"

    def test_observe_multiple_metrics(self, app_report):
        m1 = Metric(metric_name="latency", records=[0.5])
        m2 = Metric(metric_name="throughput", records=[200])
        app_report.observe_metric(ReportTypeEnum.service, "stage1", m1)
        app_report.observe_metric(ReportTypeEnum.service, "stage2", m2)

        assert len(app_report.report.metrics) == 2

    def test_metric_contains_instance_info(self, app_report, instance_id):
        metric = Metric(metric_name="accuracy", records=[0.95])
        app_report.observe_metric(ReportTypeEnum.service, "inference", metric)

        fm = app_report.report.metrics[0]
        assert str(fm.instance.id) == instance_id
        assert fm.instance.name == "test-service"

    def test_metric_has_empty_previous_instances(self, app_report):
        metric = Metric(metric_name="accuracy", records=[0.95])
        app_report.observe_metric(ReportTypeEnum.service, "inference", metric)

        fm = app_report.report.metrics[0]
        assert fm.previous_instances == []


class TestObserveInference:
    def test_observe_inference_adds_metric(self, app_report):
        app_report.observe_inference([{"class": "cat", "score": 0.9}])

        assert len(app_report.report.metrics) == 1
        fm = app_report.report.metrics[0]
        assert fm.metric_name == "Inference"
        assert fm.report_type == ReportTypeEnum.ml_specific
        assert fm.records == [{"class": "cat", "score": 0.9}]

    def test_observe_inference_uses_stage_from_config(self, app_report):
        app_report.observe_inference([0.75])

        fm = app_report.report.metrics[0]
        assert fm.stage == "inference"


class TestObserveInferenceMetric:
    def test_observe_inference_metric(self, app_report):
        metric = Metric(
            metric_name="confidence",
            records=[0.95],
            unit="probability",
            description="Model confidence",
        )
        app_report.observe_inference_metric(metric)

        assert len(app_report.report.metrics) == 1
        fm = app_report.report.metrics[0]
        assert fm.metric_name == "confidence"
        assert fm.report_type == ReportTypeEnum.ml_specific
        assert fm.unit == "probability"
        assert fm.description == "Model confidence"
        assert fm.stage == "inference"


class TestReset:
    def test_reset_clears_metrics(self, app_report):
        metric = Metric(metric_name="latency", records=[1.0])
        app_report.observe_metric(ReportTypeEnum.service, "gateway", metric)

        app_report.reset()

        assert app_report.report.metrics == []
        assert app_report.previous_reports == []

    def test_reset_preserves_execution_instance_identity(self, app_report, instance_id):
        app_report.reset()
        assert str(app_report.execution_instance.id) == instance_id


class TestProcessPreviousReport:
    def test_process_previous_report_appends_metrics(self, app_report, instance_id):
        prev_instance = MicroserviceInstance(
            id=uuid4(), name="prev-service", functionality="encoding"
        )
        prev_metric = FlattenMetric(
            metric_name="latency",
            records=[0.3],
            stage="encoding",
            report_type=ReportTypeEnum.service,
            instance=prev_instance,
            previous_instances=[],
        )
        prev_report = GeneralApplicationReportModel(metrics=[prev_metric])
        prev_dict = prev_report.model_dump()

        app_report.process_previous_report(prev_dict)

        assert len(app_report.report.metrics) == 1
        assert app_report.report.metrics[0].metric_name == "latency"
        assert len(app_report.previous_reports) == 1
