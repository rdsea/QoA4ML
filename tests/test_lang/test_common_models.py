import pytest
from pydantic import ValidationError

from qoa4ml.lang.common_models import (
    BaseConstraint,
    Condition,
    Metric,
    MetricConstraint,
)
from qoa4ml.lang.datamodel_enum import AggregateFunctionEnum, OperatorEnum


class TestMetric:
    def test_minimal(self):
        m = Metric(metric_name="accuracy")
        assert m.metric_name == "accuracy"
        assert m.records == []
        assert m.unit is None
        assert m.description is None

    def test_with_all_fields(self):
        m = Metric(
            metric_name="response_time",
            records=[1.2, 3.4, 5.6],
            unit="seconds",
            description="End-to-end latency",
        )
        assert m.metric_name == "response_time"
        assert m.records == [1.2, 3.4, 5.6]
        assert m.unit == "seconds"
        assert m.description == "End-to-end latency"

    def test_records_accepts_mixed_types(self):
        m = Metric(
            metric_name="test",
            records=[{"a": 1}, 0.5, 42, (1, 2), "info"],
        )
        assert len(m.records) == 5

    def test_with_enum_metric_name(self):
        from qoa4ml.lang.attributes import ServiceQualityEnum

        m = Metric(metric_name=ServiceQualityEnum.RESPONSE_TIME)
        assert m.metric_name == "response_time"

    def test_missing_metric_name_raises(self):
        with pytest.raises(ValidationError):
            Metric()

    def test_serialization_round_trip(self):
        m = Metric(metric_name="accuracy", records=[0.9], unit="percent")
        data = m.model_dump()
        m2 = Metric(**data)
        assert m2.metric_name == m.metric_name
        assert m2.records == m.records
        assert m2.unit == m.unit


class TestCondition:
    def test_with_float_value(self):
        c = Condition(operator=OperatorEnum.geq, value=0.95)
        assert c.operator == OperatorEnum.geq
        assert c.value == 0.95

    def test_with_int_value(self):
        c = Condition(operator=OperatorEnum.eq, value=100)
        assert c.value == 100

    def test_with_dict_value(self):
        c = Condition(operator=OperatorEnum.range, value={"min": 0.8, "max": 1.0})
        assert c.value == {"min": 0.8, "max": 1.0}

    def test_all_operators(self):
        for op in OperatorEnum:
            c = Condition(operator=op, value=1)
            assert c.operator == op

    def test_missing_operator_raises(self):
        with pytest.raises(ValidationError):
            Condition(value=0.5)

    def test_missing_value_raises(self):
        with pytest.raises(ValidationError):
            Condition(operator=OperatorEnum.eq)


class TestMetricConstraint:
    def test_valid(self):
        m = Metric(metric_name="accuracy", records=[0.95])
        c = Condition(operator=OperatorEnum.geq, value=0.9)
        mc = MetricConstraint(
            metrics=m,
            condition=c,
            aggregate_function=AggregateFunctionEnum.AVG,
        )
        assert mc.metrics.metric_name == "accuracy"
        assert mc.condition.operator == OperatorEnum.geq
        assert mc.aggregate_function == AggregateFunctionEnum.AVG

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            MetricConstraint(
                metrics=Metric(metric_name="accuracy"),
                condition=Condition(operator=OperatorEnum.eq, value=1),
            )

    def test_serialization(self):
        mc = MetricConstraint(
            metrics=Metric(metric_name="mse"),
            condition=Condition(operator=OperatorEnum.leq, value=0.1),
            aggregate_function=AggregateFunctionEnum.MIN,
        )
        data = mc.model_dump()
        assert data["aggregate_function"] == "MIN"
        assert data["metrics"]["metric_name"] == "mse"


class TestBaseConstraint:
    def test_valid(self):
        mc = MetricConstraint(
            metrics=Metric(metric_name="accuracy"),
            condition=Condition(operator=OperatorEnum.geq, value=0.9),
            aggregate_function=AggregateFunctionEnum.AVG,
        )
        bc = BaseConstraint(name="accuracy_constraint", constraint_list=[mc])
        assert bc.name == "accuracy_constraint"
        assert len(bc.constraint_list) == 1

    def test_multiple_constraints(self):
        constraints = [
            MetricConstraint(
                metrics=Metric(metric_name="accuracy"),
                condition=Condition(operator=OperatorEnum.geq, value=0.9),
                aggregate_function=AggregateFunctionEnum.AVG,
            ),
            MetricConstraint(
                metrics=Metric(metric_name="response_time"),
                condition=Condition(operator=OperatorEnum.leq, value=500),
                aggregate_function=AggregateFunctionEnum.MAX,
            ),
        ]
        bc = BaseConstraint(name="sla_constraints", constraint_list=constraints)
        assert len(bc.constraint_list) == 2

    def test_empty_constraint_list(self):
        bc = BaseConstraint(name="empty", constraint_list=[])
        assert bc.constraint_list == []

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            BaseConstraint(constraint_list=[])

    def test_missing_constraint_list_raises(self):
        with pytest.raises(ValidationError):
            BaseConstraint(name="test")
