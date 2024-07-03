from typing import List, Optional, Union

from pydantic import BaseModel

from .datamodel_enum import AggregateFunctionEnum, MetricNameEnum, OperatorEnum


class Metric(BaseModel):
    metric_name: MetricNameEnum
    records: List[Union[dict, float, int, tuple, str]] = []
    unit: Optional[str] = None
    description: Optional[str] = None


class Condition(BaseModel):
    operator: OperatorEnum
    value: Union[dict, float, int]


class MetricConstraint(BaseModel):
    metrics: Metric
    condition: Condition
    aggregate_function: AggregateFunctionEnum


class BaseConstraint(BaseModel):
    name: str
    constraint_list: List[MetricConstraint]
