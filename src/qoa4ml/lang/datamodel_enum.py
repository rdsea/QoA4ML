from enum import Enum, StrEnum

from qoa4ml.lang.attributes import (
    DataQualityEnum,
    MLModelQualityEnum,
    ServiceQualityEnum,
)


class ResourcesUtilizationMetricNameEnum(StrEnum):
    cpu = "cpu_usage"
    memory = "memory_usage"


class ImageQualityNameEnum(StrEnum):
    image_size = "image_size"
    object_size = "object_size"
    color_mode = "color_mode"
    color_channel = "color_channel"


MetricNameEnum = (
    ServiceQualityEnum
    | MLModelQualityEnum
    | DataQualityEnum
    | ResourcesUtilizationMetricNameEnum
    | ImageQualityNameEnum
    | str
)


class StageNameEnum(StrEnum):
    ml_inference_aggregate = "ml_inference_aggregate"
    ml_inference_ensemble = "ml_inference_ensemble"
    data_processing = "data_processing"
    gateway = "gateway"


class FunctionalityEnum(StrEnum):
    rest = "REST"
    tensorflow = "TensorFlow"
    transformation = "Transformation"
    max_aggregate = "Max Aggregate"


class StakeholderRoleEnum(StrEnum):
    ml_consumer = "ml_consumer"
    ml_provider = "ml_provider"
    ml_infrastructure = "ml_infrastructure"


class ResourceEnum(StrEnum):
    ml_service = "ml_service"
    storage = "storage"
    ml_models = "ml_models"


class ServiceAPIEnum(StrEnum):
    rest = "REST"
    mqtt = "MQTT"
    kafka = "Kafka"
    amqp = "AMQP"
    coapp = "coapp"
    socket = "socket"
    debug = "Debug"


class InfrastructureEnum(StrEnum):
    raspi4 = "Raspberry Pi 4 Model B"
    nvidia_jetson_nano = "NVIDIA Jetson Nano"
    nvidia_jetson_orin_nano = "NVIDIA Jetson Orin Nano"
    nvidia_jetson_agx_xavier = "NVIDIA Jetson AGX Xavier"
    beelink_bt3 = "Beelink BT3"
    rock_pi_n10 = "Rock Pi N10"


class ProcessorEnum(StrEnum):
    cpu = "CPU"
    gpu = "GPU"
    tpu = "TPU"


class DataTypeEnum(StrEnum):
    video = "video"
    image = "image"
    message = "message"


class DataFormatEnum(StrEnum):
    binary = "binary"
    csv = "csv"
    json = "json"
    avro = "avro"
    png = "png"
    jpg = "jpg"
    mp4 = "mp4"


class DevelopmentEnvironmentEnum(StrEnum):
    kerash5 = "kerash5"
    onnx = "onnx"


class ServingPlatformEnum(Enum):
    tensorflow = "TensorFlow"
    prediction = "prediction"


class ModelCategoryEnum(StrEnum):
    svm = "SVM"
    dt = "DT"
    cnn = "CNN"
    lr = "LR"
    kmeans = "KMeans"
    ann = "ANN"


class InferenceModeEnum(StrEnum):
    static = "static"
    dynamic = "dynamic"


class OperatorEnum(StrEnum):
    range = "range"
    leq = "less_equal"
    geq = "greater_equal"
    lt = "less_than"
    gt = "greater_than"
    eq = "equal"


class AggregateFunctionEnum(StrEnum):
    MIN = "MIN"
    MAX = "MAX"
    AVG = "AVERAGE"
    SUM = "SUM"
    COUNT = "COUNT"
    OR = "OR"
    AND = "AND"
    PRODUCT = "PRODUCT"


class CostUnitEnum(StrEnum):
    usd = "USD"
    eur = "EUR"
    other = "other"


class MetricCategoryEnum(StrEnum):
    service = "service"
    data = "data"
    ml_specific = "ml_specific"
    quality = "quality"
    inference = "inference"


class CgroupVersionEnum(StrEnum):
    v1 = "cgroupv1"
    v2 = "cgroupv2"


class MetricClassEnum(StrEnum):
    gauge = "Gauge"
    counter = "Counter"
    summary = "Summary"
    histogram = "Histogram"


class ReportTypeEnum(StrEnum):
    data = "data_report"
    service = "service_report"
    ml_specific = "ml_specific_report"
    security = "security_report"


class EnvironmentEnum(StrEnum):
    hpc = "HPC"
    edge = "Edge"
    cloud = "Cloud"
