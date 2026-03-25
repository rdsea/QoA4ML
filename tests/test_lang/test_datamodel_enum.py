from enum import StrEnum

from qoa4ml.lang.datamodel_enum import (
    AggregateFunctionEnum,
    CgroupVersionEnum,
    CostUnitEnum,
    DataFormatEnum,
    DataTypeEnum,
    DevelopmentEnvironmentEnum,
    EnvironmentEnum,
    FunctionalityEnum,
    ImageQualityNameEnum,
    InferenceModeEnum,
    InfrastructureEnum,
    MetricCategoryEnum,
    MetricClassEnum,
    MetricNameEnum,
    ModelCategoryEnum,
    OperatorEnum,
    ProcessorEnum,
    ReportTypeEnum,
    ResourceEnum,
    ResourcesUtilizationMetricNameEnum,
    ServiceAPIEnum,
    ServingPlatformEnum,
    StageNameEnum,
    StakeholderRoleEnum,
)

ALL_STRENUM_CLASSES = [
    ResourcesUtilizationMetricNameEnum,
    ImageQualityNameEnum,
    StageNameEnum,
    FunctionalityEnum,
    StakeholderRoleEnum,
    ResourceEnum,
    ServiceAPIEnum,
    InfrastructureEnum,
    ProcessorEnum,
    DataTypeEnum,
    DataFormatEnum,
    DevelopmentEnvironmentEnum,
    ModelCategoryEnum,
    InferenceModeEnum,
    OperatorEnum,
    AggregateFunctionEnum,
    CostUnitEnum,
    MetricCategoryEnum,
    CgroupVersionEnum,
    MetricClassEnum,
    ReportTypeEnum,
    EnvironmentEnum,
]


class TestEnumUniqueness:
    """Every enum class should have unique values (no two members share a value)."""

    def test_all_enums_have_unique_values(self):
        for enum_cls in ALL_STRENUM_CLASSES:
            values = [member.value for member in enum_cls]
            assert len(values) == len(
                set(values)
            ), f"{enum_cls.__name__} has duplicate values"

    def test_all_enums_have_unique_names(self):
        for enum_cls in ALL_STRENUM_CLASSES:
            names = [member.name for member in enum_cls]
            assert len(names) == len(
                set(names)
            ), f"{enum_cls.__name__} has duplicate names"


class TestSpecificEnumValues:
    def test_service_api_enum(self):
        assert ServiceAPIEnum.rest == "REST"
        assert ServiceAPIEnum.mqtt == "MQTT"
        assert ServiceAPIEnum.kafka == "Kafka"
        assert ServiceAPIEnum.amqp == "AMQP"
        assert ServiceAPIEnum.socket == "socket"
        assert ServiceAPIEnum.debug == "Debug"
        assert ServiceAPIEnum.coapp == "coapp"

    def test_environment_enum(self):
        assert EnvironmentEnum.hpc == "HPC"
        assert EnvironmentEnum.edge == "Edge"
        assert EnvironmentEnum.cloud == "Cloud"

    def test_operator_enum(self):
        assert OperatorEnum.range == "range"
        assert OperatorEnum.leq == "less_equal"
        assert OperatorEnum.geq == "greater_equal"
        assert OperatorEnum.lt == "less_than"
        assert OperatorEnum.gt == "greater_than"
        assert OperatorEnum.eq == "equal"

    def test_metric_class_enum(self):
        assert MetricClassEnum.gauge == "Gauge"
        assert MetricClassEnum.counter == "Counter"
        assert MetricClassEnum.summary == "Summary"
        assert MetricClassEnum.histogram == "Histogram"

    def test_aggregate_function_enum(self):
        assert AggregateFunctionEnum.MIN == "MIN"
        assert AggregateFunctionEnum.MAX == "MAX"
        assert AggregateFunctionEnum.AVG == "AVERAGE"
        assert AggregateFunctionEnum.SUM == "SUM"
        assert AggregateFunctionEnum.COUNT == "COUNT"
        assert AggregateFunctionEnum.OR == "OR"
        assert AggregateFunctionEnum.AND == "AND"
        assert AggregateFunctionEnum.PRODUCT == "PRODUCT"

    def test_processor_enum(self):
        assert ProcessorEnum.cpu == "CPU"
        assert ProcessorEnum.gpu == "GPU"
        assert ProcessorEnum.tpu == "TPU"

    def test_data_format_enum(self):
        assert DataFormatEnum.binary == "binary"
        assert DataFormatEnum.csv == "csv"
        assert DataFormatEnum.json == "json"
        assert DataFormatEnum.png == "png"

    def test_report_type_enum(self):
        assert ReportTypeEnum.data == "data_report"
        assert ReportTypeEnum.service == "service_report"
        assert ReportTypeEnum.ml_specific == "ml_specific_report"

    def test_resources_utilization_enum(self):
        assert ResourcesUtilizationMetricNameEnum.cpu == "cpu_usage"
        assert ResourcesUtilizationMetricNameEnum.memory == "memory_usage"

    def test_image_quality_name_enum(self):
        assert ImageQualityNameEnum.image_size == "image_size"
        assert ImageQualityNameEnum.color_mode == "color_mode"


class TestMetricNameEnumTypeAlias:
    """MetricNameEnum is a Union type alias, not an actual Enum class."""

    def test_accepts_service_quality_value(self):
        from qoa4ml.lang.attributes import ServiceQualityEnum

        val: MetricNameEnum = ServiceQualityEnum.RESPONSE_TIME
        assert val == "response_time"

    def test_accepts_ml_model_quality_value(self):
        from qoa4ml.lang.attributes import MLModelQualityEnum

        val: MetricNameEnum = MLModelQualityEnum.ACCURACY
        assert val == "accuracy"

    def test_accepts_data_quality_value(self):
        from qoa4ml.lang.attributes import DataQualityEnum

        val: MetricNameEnum = DataQualityEnum.COMPLETENESS
        assert val == "completeness"

    def test_accepts_resources_utilization_value(self):
        val: MetricNameEnum = ResourcesUtilizationMetricNameEnum.cpu
        assert val == "cpu_usage"

    def test_accepts_plain_string(self):
        val: MetricNameEnum = "custom_metric"
        assert val == "custom_metric"


class TestServingPlatformEnum:
    """ServingPlatformEnum uses base Enum, not StrEnum."""

    def test_is_not_strenum(self):
        assert not issubclass(ServingPlatformEnum, StrEnum)

    def test_values(self):
        assert ServingPlatformEnum.tensorflow.value == "TensorFlow"
        assert ServingPlatformEnum.prediction.value == "prediction"
