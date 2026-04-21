from pydantic import BaseModel, Field

from qoa4ml.lang.datamodel_enum import (
    AggregateFunctionEnum,
    MetricNameEnum,
    OperatorEnum,
)


class Metric(BaseModel):
    metric_name: MetricNameEnum
    records: list[dict | float | int | tuple | str] = Field(default_factory=list)
    unit: str | None = None
    description: str | None = None


class Condition(BaseModel):
    operator: OperatorEnum
    value: dict | float | int


class MetricConstraint(BaseModel):
    metrics: Metric
    condition: Condition
    aggregate_function: AggregateFunctionEnum


class BaseConstraint(BaseModel):
    name: str
    constraint_list: list[MetricConstraint]
