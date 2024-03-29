from enum import Enum
from typing import Union


class ServiceMetricNameEnum(Enum):
    response_time = "response_time"
    reliability = "reliability"
    completeness = "completeness"


class MlSpecificMetricNameEnum(Enum):
    confidence = "confidence"
    acccuracy = "accuracy"
    precision = "precision"
    recall = "recall"
    auc = "auc"
    mse = "mse"


class ResourcesUtilizationMetricNameEnum(Enum):
    cpu = "cpu_usage"
    memory = "memory_usage"


print(ResourcesUtilizationMetricNameEnum.memory == "cpu_usage")


MetricNameEnum = Union[
    ServiceMetricNameEnum,
    MlSpecificMetricNameEnum,
    ResourcesUtilizationMetricNameEnum,
    str,
]


class DataQualityEnum(Enum):
    image_size = "image_size"
    object_size = "object_size"


class StageNameEnum(Enum):
    ml_inference_aggregate = "ml_inference_aggregate"
    ml_inference_ensemble = "ml_inference_ensemble"
    data_processing = "data_processing"
    gate_way = "gateway"


class MethodEnum(Enum):
    rest = "REST"
    tensorflow = "tensorflow"
    transformation = "transformation"
    max_aggregate = "max_aggregate"


class StakeholderRoleEnum(Enum):
    ml_consumer = "ml_provider"
    ml_provider = "ml_provider"
    ml_infrastructure = "ml_infrastructure"


class ResourceEnum(Enum):
    ml_service = "ml_service"
    storage = "storage"
    ml_models = "ml_models"


class ServiceAPIEnum(Enum):
    rest = "REST"
    mqtt = "MQTT"
    kafka = "Kafka"
    amqp = "AMQP"
    coapp = "coapp"


class InfrastructureEnum(Enum):
    raspi4 = "Raspberry Pi 4 Model B"
    nvidia_jetson_nano = "NVIDIA Jetson Nano"
    nvidia_jetson_orin_nano = "NVIDIA Jetson Orin Nano"
    nvidia_jetson_agx_xavier = "NVIDIA Jetson AGX Xavier"
    beelink_bt3 = "Beelink BT3"
    rock_pi_n10 = "Rock Pi N10"


class ProcessorEnum(Enum):
    cpu = "CPU"
    gpu = "GPU"
    tpu = "TPU"


class DataTypeEnum(Enum):
    video = "video"
    image = "image"
    message = "message"


class DataFormatEnum(Enum):
    binary = "binary"
    csv = "csv"
    json = "json"
    avro = "avro"
    png = "png"
    jpg = "jpg"
    mp4 = "mp4"


class DevelopmentEnvironmentEnum(Enum):
    kerash5 = "kerash5"
    onnx = "onnx"


class ServingPlatformEnum(Enum):
    tensorflow = "TensorFlow"
    predictio = "predictio"


class ModelCategoryEnum(Enum):
    svm = "SVM"
    dt = "DT"
    cnn = "CNN"
    lr = "LR"
    kmeans = "KMeans"
    ann = "ANN"


class InferenceModeEnum(Enum):
    static = "static"
    dynamic = "dynamic"


class OperatorEnum(Enum):
    range = "range"
    leq = "less_equal"
    geq = "greater_equal"
    lt = "less_than"
    gt = "greater_than"
    eq = "equal"


class AggregateFunctionEnum(Enum):
    MIN = "MIN"
    MAX = "MAX"
    AVG = "AVERAGE"
    SUM = "SUM"
    COUNT = "COUNT"
    OR = "OR"
    AND = "AND"
    PRODUCT = "PRODUCT"


class CostUnitEnum(Enum):
    usd = "USD"
    eur = "EUR"
    other = "other"


class MetricCategoryEnum(Enum):
    service = "service"
    data = "data"
    ml_specific = "ml_specific"
    quality = "quality"
    inference = "inference"


class CgroupVersionEnum(Enum):
    v1 = "cgroupv1"
    v2 = "cgroupv2"


class MetricClassEnum(Enum):
    gauge = "Gauge"
    coutner = "Counter"
    summary = "Summary"
