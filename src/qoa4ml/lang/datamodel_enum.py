from enum import Enum
from typing import Union

from qoa4ml.lang.attributes import (
    DataQualityEnum,
    MLModelQualityEnum,
    ServiceQualityEnum,
)


class ResourcesUtilizationMetricNameEnum(str, Enum):
    cpu = "cpu_usage"
    memory = "memory_usage"


class ImageQualityNameEnum(str, Enum):
    image_size = "image_size"
    object_size = "object_size"
    color_mode = "color_mode"
    color_channel = "color_channel"


MetricNameEnum = Union[
    ServiceQualityEnum,
    MLModelQualityEnum,
    DataQualityEnum,
    ResourcesUtilizationMetricNameEnum,
    ImageQualityNameEnum,
    str,
]


class StageNameEnum(str, Enum):
    ml_inference_aggregate = "ml_inference_aggregate"
    ml_inference_ensemble = "ml_inference_ensemble"
    data_processing = "data_processing"
    gateway = "gateway"


class FunctionalityEnum(str, Enum):
    rest = "REST"
    tensorflow = "TensorFlow"
    transformation = "Transformation"
    max_aggregate = "Max Aggregate"


class StakeholderRoleEnum(str, Enum):
    ml_consumer = "ml_provider"
    ml_provider = "ml_provider"
    ml_infrastructure = "ml_infrastructure"


class ResourceEnum(str, Enum):
    ml_service = "ml_service"
    storage = "storage"
    ml_models = "ml_models"


class ServiceAPIEnum(str, Enum):
    rest = "REST"
    mqtt = "MQTT"
    kafka = "Kafka"
    amqp = "AMQP"
    coapp = "coapp"
    socket = "socket"
    debug = "Debug"


class InfrastructureEnum(str, Enum):
    raspi4 = "Raspberry Pi 4 Model B"
    nvidia_jetson_nano = "NVIDIA Jetson Nano"
    nvidia_jetson_orin_nano = "NVIDIA Jetson Orin Nano"
    nvidia_jetson_agx_xavier = "NVIDIA Jetson AGX Xavier"
    beelink_bt3 = "Beelink BT3"
    rock_pi_n10 = "Rock Pi N10"


class ProcessorEnum(str, Enum):
    cpu = "CPU"
    gpu = "GPU"
    tpu = "TPU"


class DataTypeEnum(str, Enum):
    video = "video"
    image = "image"
    message = "message"


class DataFormatEnum(str, Enum):
    binary = "binary"
    csv = "csv"
    json = "json"
    avro = "avro"
    png = "png"
    jpg = "jpg"
    mp4 = "mp4"


class DevelopmentEnvironmentEnum(str, Enum):
    kerash5 = "kerash5"
    onnx = "onnx"


class ServingPlatformEnum(Enum):
    tensorflow = "TensorFlow"
    predictio = "predictio"


class ModelCategoryEnum(str, Enum):
    svm = "SVM"
    dt = "DT"
    cnn = "CNN"
    lr = "LR"
    kmeans = "KMeans"
    ann = "ANN"


class InferenceModeEnum(str, Enum):
    static = "static"
    dynamic = "dynamic"


class OperatorEnum(str, Enum):
    range = "range"
    leq = "less_equal"
    geq = "greater_equal"
    lt = "less_than"
    gt = "greater_than"
    eq = "equal"


class AggregateFunctionEnum(str, Enum):
    MIN = "MIN"
    MAX = "MAX"
    AVG = "AVERAGE"
    SUM = "SUM"
    COUNT = "COUNT"
    OR = "OR"
    AND = "AND"
    PRODUCT = "PRODUCT"


class CostUnitEnum(str, Enum):
    usd = "USD"
    eur = "EUR"
    other = "other"


class MetricCategoryEnum(str, Enum):
    service = "service"
    data = "data"
    ml_specific = "ml_specific"
    quality = "quality"
    inference = "inference"


class CgroupVersionEnum(str, Enum):
    v1 = "cgroupv1"
    v2 = "cgroupv2"


class MetricClassEnum(str, Enum):
    gauge = "Gauge"
    counter = "Counter"
    summary = "Summary"
    histogram = "Histogram"


class ReportTypeEnum(str, Enum):
    data = "data_report"
    service = "service_report"
    ml_specific = "ml_specific_report"
    security = "security_report"


class EnvironmentEnum(str, Enum):
    hpc = "HPC"
    edge = "Edge"
    cloud = "Cloud"
