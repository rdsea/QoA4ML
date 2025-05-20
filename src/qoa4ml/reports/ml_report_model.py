from typing import Generic, Optional, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel

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
    stage: Optional[str] = None
    # stage: Optional[StageNameEnum] = None


class StageReport(BaseModel):
    # name: StageNameEnum
    name: str
    metrics: dict[MetricNameEnum, dict[UUID, Metric]]


class InferenceInstance(BaseModel):
    inference_id: UUID
    instance_id: UUID
    functionality: str
    metrics: list[Metric] = []
    prediction: Optional[Union[dict, float]] = None


InstanceType = TypeVar("InstanceType")


class LinkedInstance(BaseModel, Generic[InstanceType]):
    previous: list[InstanceType] = []
    instance: InstanceType


class ExecutionGraph(BaseModel):
    end_point: Optional[MicroserviceInstance] = None
    linked_list: dict[UUID, LinkedInstance[MicroserviceInstance]]


class InferenceGraph(BaseModel):
    end_point: Optional[InferenceInstance] = None
    linked_list: dict[UUID, LinkedInstance[InferenceInstance]] = {}


# NOTE: use dict so that we know which stage to add metric to


class BaseReport(BaseModel):
    metadata: dict = {}


class FlattenMetric(Metric):
    stage: str
    report_type: ReportTypeEnum
    instance: MicroserviceInstance
    previous_instances: list[MicroserviceInstance]


class GeneralApplicationReportModel(BaseReport):
    metrics: list[FlattenMetric] = []


class MlQualityReport(BaseModel):
    service: dict[str, StageReport] = {}
    data: dict[str, StageReport] = {}


class GeneralMlInferenceReport(MlQualityReport, BaseReport):
    ml_inference: dict[str, InferenceInstance] = {}


class EnsembleInferenceReport(MlQualityReport):
    ml_specific: Optional[InferenceGraph] = None


class RoheReportModel(BaseReport):
    inference_report: Optional[EnsembleInferenceReport] = None
    execution_graph: Optional[ExecutionGraph] = None
