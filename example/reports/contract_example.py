"""QoA4ML contract/constraint example.

Shows how to create quality contracts using Metric, Condition,
MetricConstraint, and BaseConstraint objects. These contracts
express quality requirements that can be evaluated against reports.
"""

from qoa4ml.lang.attributes import DataQualityEnum, ServiceQualityEnum
from qoa4ml.lang.common_models import (
    BaseConstraint,
    Condition,
    Metric,
    MetricConstraint,
)
from qoa4ml.lang.datamodel_enum import AggregateFunctionEnum, OperatorEnum


def main():
    # ---- Constraint 1: Response time must be <= 200 ms on average ----
    print("=" * 60)
    print("Building quality contract: response time constraint")
    print("=" * 60)

    response_time_metric = Metric(
        metric_name=ServiceQualityEnum.RESPONSE_TIME,
        records=[],
        unit="ms",
        description="Service response time",
    )
    response_time_condition = Condition(
        operator=OperatorEnum.leq,
        value=200.0,
    )
    response_time_constraint = MetricConstraint(
        metrics=response_time_metric,
        condition=response_time_condition,
        aggregate_function=AggregateFunctionEnum.AVG,
    )
    print(f"  Metric: {response_time_metric.metric_name}")
    print(
        f"  Condition: {response_time_condition.operator} {response_time_condition.value}"
    )
    print(f"  Aggregate: {response_time_constraint.aggregate_function}\n")

    # ---- Constraint 2: Service reliability must be >= 99% ----
    print("=" * 60)
    print("Building quality contract: reliability constraint")
    print("=" * 60)

    reliability_metric = Metric(
        metric_name=ServiceQualityEnum.RELIABILITY,
        records=[],
        unit="%",
        description="Service reliability percentage",
    )
    reliability_condition = Condition(
        operator=OperatorEnum.geq,
        value=99.0,
    )
    reliability_constraint = MetricConstraint(
        metrics=reliability_metric,
        condition=reliability_condition,
        aggregate_function=AggregateFunctionEnum.MIN,
    )
    print(f"  Metric: {reliability_metric.metric_name}")
    print(
        f"  Condition: {reliability_condition.operator} {reliability_condition.value}"
    )
    print(f"  Aggregate: {reliability_constraint.aggregate_function}\n")

    # ---- Constraint 3: Data accuracy must be >= 95% on average ----
    print("=" * 60)
    print("Building quality contract: data accuracy constraint")
    print("=" * 60)

    accuracy_metric = Metric(
        metric_name=DataQualityEnum.ACCURACY,
        records=[],
        unit="%",
        description="Data accuracy ratio",
    )
    accuracy_condition = Condition(
        operator=OperatorEnum.geq,
        value=95.0,
    )
    accuracy_constraint = MetricConstraint(
        metrics=accuracy_metric,
        condition=accuracy_condition,
        aggregate_function=AggregateFunctionEnum.AVG,
    )
    print(f"  Metric: {accuracy_metric.metric_name}")
    print(f"  Condition: {accuracy_condition.operator} {accuracy_condition.value}")
    print(f"  Aggregate: {accuracy_constraint.aggregate_function}\n")

    # ---- Combine constraints into a contract ----
    print("=" * 60)
    print("Assembling the full quality contract")
    print("=" * 60)

    contract = BaseConstraint(
        name="ml_service_quality_contract",
        constraint_list=[
            response_time_constraint,
            reliability_constraint,
            accuracy_constraint,
        ],
    )

    contract_json = contract.model_dump_json(indent=2)
    print(contract_json)

    # Demonstrate round-trip: parse the JSON back into a BaseConstraint
    parsed = BaseConstraint.model_validate_json(contract_json)
    print(
        f"\nRound-trip OK: parsed contract has {len(parsed.constraint_list)} constraints."
    )
    for i, mc in enumerate(parsed.constraint_list, 1):
        print(
            f"  [{i}] {mc.metrics.metric_name} "
            f"{mc.condition.operator} {mc.condition.value} "
            f"(agg={mc.aggregate_function})"
        )


if __name__ == "__main__":
    main()
