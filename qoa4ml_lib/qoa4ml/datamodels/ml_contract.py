from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel
from datamodel_enum import *
import json


class Stakeholder(BaseModel):
    id: str
    name: str
    roles: StakeholderRoleEnum
    provisioning: ResourceEnum


class MicroserviceSpecs(BaseModel):
    id: str
    name: str
    service_api: List[ServiceAPIEnum]
    infrastructure: List[InfrastructureEnum]
    processor_types: List[ProcessorEnum]


class DataSpecs(BaseModel):
    id: str
    name: str
    types: List[DataTypeEnum]
    formats: List[DataFormatEnum]


class MLSpecs(BaseModel):
    id: str
    name: str
    development_environment: List[DevelopmentEnvironmentEnum]
    serving_platform: List[ServingPlatformEnum]
    model_classes: List[ModelCategoryEnum]
    inference_modes: List[InferenceModeEnum]


class ResourceConstraint(BaseModel):
    services_specs: List[MicroserviceSpecs]
    data_specs: DataSpecs
    ml_specs: MLSpecs


class Metric(BaseModel):
    metric_name: MetricNameEnum
    record: dict | float | int
    category: MetricCategoryEnum


class Condition(Enum):
    operator: OperatorEnum
    metric: Metric


# TODO: use BaseConstraint
class CostConstraint(BaseModel):
    operator: OperatorEnum
    unit: CostUnitEnum
    value: float
    # TODO: fix naming
    condition: List[Condition]


class InterpretabilityConstraint(BaseModel):
    explainability: dict


class FairnessConstraint(BaseModel):
    bias: dict


class PrivacyConstraint(BaseModel):
    risks: dict


class SecurityConstraint(BaseModel):
    encryption: dict


class BaseConstraint(BaseModel):
    id: str
    metric: Metric
    operator: OperatorEnum
    condition: List[Condition]


# TODO: can have additional attributed
class MLSpecificConstraint(BaseConstraint):
    pass


# TODO: can have additional attributed
class DataConstraint(BaseConstraint):
    pass


# TODO: can have additional attributed
class ServiceConstraint(BaseConstraint):
    pass


class QualityConstraint(BaseModel):
    service: List[ServiceConstraint]
    data: List[DataConstraint]
    ml_specific: List[MLSpecificConstraint]
    security: List[SecurityConstraint]
    privacy: List[PrivacyConstraint]
    fairness: List[FairnessConstraint]
    interpretability: List[InterpretabilityConstraint]
    cost: List[CostConstraint]


class MLContract(BaseModel):
    stakeholders: List[Stakeholder]
    resources: ResourceConstraint
    quality: QualityConstraint

with open("ml_contract_model.json", "w") as file:
    json.dump(MLContract.model_json_schema(), file

    )

