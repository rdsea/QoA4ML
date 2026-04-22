from typing import TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

from qoa4ml.lang.common_models import Metric
from qoa4ml.lang.datamodel_enum import MetricNameEnum, ReportTypeEnum

GENERAL_REPORT_VERSION = "v0.1"
GENERAL_REPORT_NAME = "qoa4ml-report-common-schema"

ML_REPORT_VERSION = "v0.1"
ML_REPORT_NAME = "qoa4ml-report-ml-schema"

ENSEMBLE_REPORT_VERSION = "v0.1"
ENSEMBLE_REPORT_NAME = "qoa4ml-report-eemls-schema"


class MicroserviceInstance(BaseModel):
    id: UUID
    name: str
    functionality: str = ""
    stage: str | None = None


class StageReport(BaseModel):
    name: str
    metrics: dict[MetricNameEnum, dict[UUID, Metric]]


class InferenceInstance(BaseModel):
    inference_id: UUID
    instance_id: UUID
    functionality: str
    metrics: list[Metric] = Field(default_factory=list)
    prediction: dict | float | None = None


InstanceType = TypeVar("InstanceType")


class LinkedInstance[InstanceType](BaseModel):
    previous: list[InstanceType] = Field(default_factory=list)
    instance: InstanceType


class ExecutionGraph(BaseModel):
    end_point: MicroserviceInstance | None = None
    linked_list: dict[UUID, LinkedInstance[MicroserviceInstance]]


class InferenceGraph(BaseModel):
    end_point: InferenceInstance | None = None
    linked_list: dict[UUID, LinkedInstance[InferenceInstance]] = Field(
        default_factory=dict
    )


# NOTE: use dict so that we know which stage to add metric to


class BaseReport(BaseModel):
    metadata: dict = Field(default_factory=dict)


class FlattenMetric(Metric):
    stage: str
    report_type: ReportTypeEnum
    instance: MicroserviceInstance
    previous_instances: list[MicroserviceInstance]


class GeneralApplicationReportModel(BaseReport):
    metrics: list[FlattenMetric] = Field(default_factory=list)


class MlQualityReport(BaseModel):
    service: dict[str, StageReport] = Field(default_factory=dict)
    data: dict[str, StageReport] = Field(default_factory=dict)
    security: dict[str, StageReport] = Field(default_factory=dict)


class GeneralMlInferenceReport(MlQualityReport, BaseReport):
    ml_inference: dict[UUID, InferenceInstance] = Field(default_factory=dict)


class EnsembleInferenceReport(MlQualityReport):
    ml_specific: InferenceGraph | None = None


class RoheReportModel(BaseReport):
    inference_report: EnsembleInferenceReport | None = None
    execution_graph: ExecutionGraph | None = None
