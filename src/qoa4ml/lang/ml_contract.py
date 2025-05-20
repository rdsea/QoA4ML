from pydantic import BaseModel

from qoa4ml.lang.common_models import BaseConstraint
from qoa4ml.lang.datamodel_enum import (
    DataFormatEnum,
    DataTypeEnum,
    DevelopmentEnvironmentEnum,
    InferenceModeEnum,
    InfrastructureEnum,
    ModelCategoryEnum,
    ProcessorEnum,
    ResourceEnum,
    ServiceAPIEnum,
    ServingPlatformEnum,
    StakeholderRoleEnum,
)

ML_CONTRACT_VERSION = "v0.3"
ML_CONTRACT_NAME = "qoa4ml-contract-schema"


class Stakeholder(BaseModel):
    id: str
    name: str
    roles: list[StakeholderRoleEnum]
    provisioning: list[ResourceEnum]


class MicroserviceSpecs(BaseModel):
    id: str
    name: str
    service_api: list[ServiceAPIEnum]
    infrastructure: list[InfrastructureEnum]
    processor_types: list[ProcessorEnum]


class DataSpecs(BaseModel):
    id: str
    name: str
    types: list[DataTypeEnum]
    formats: list[DataFormatEnum]


class MLSpecs(BaseModel):
    id: str
    name: str
    development_environment: list[DevelopmentEnvironmentEnum]
    serving_platform: list[ServingPlatformEnum]
    model_classes: list[ModelCategoryEnum]
    inference_modes: list[InferenceModeEnum]


class ResourceSpecs(BaseModel):
    services_specs: list[MicroserviceSpecs]
    data_specs: DataSpecs
    ml_specs: MLSpecs


class CostConstraint(BaseConstraint):
    name: str = "cost_constraint"


class InterpretabilityConstraint(BaseModel):
    explainability: dict


class FairnessConstraint(BaseConstraint):
    name: str = "fairness_constraint"


class PrivacyConstraint(BaseModel):
    risks: dict


class SecurityConstraint(BaseModel):
    encryption: dict


class MLSpecificConstraint(BaseConstraint):
    name: str = "ml_specific_constraint"


class DataConstraint(BaseConstraint):
    name: str = "data_constaint"


class ServiceConstraint(BaseConstraint):
    name: str = "service_constraint"


class QualityConstraint(BaseModel):
    service: list[ServiceConstraint]
    data: list[DataConstraint]
    ml_specific: list[MLSpecificConstraint]
    security: list[SecurityConstraint]
    privacy: list[PrivacyConstraint]
    fairness: list[FairnessConstraint]
    interpretability: list[InterpretabilityConstraint]
    cost: list[CostConstraint]


class MLContract(BaseModel):
    stakeholders: list[Stakeholder]
    resources: ResourceSpecs
    quality: QualityConstraint
