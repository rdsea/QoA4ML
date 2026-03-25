import inspect
from enum import EnumMeta, StrEnum

QOA_ATTRIBUTES_VERSION = "v0.3"
QOA_ATTRIBUTES_NAME = "qoa4ml-attributes"


class MetaEnum(EnumMeta):
    def __new__(cls, clsname, bases, classdict):
        cls = super().__new__(cls, clsname, bases, classdict)

        source = inspect.getsource(cls)
        docstrings = source.split('"""')[-2::-2]

        for member_name, doc_str in zip(
            reversed(cls._member_names_), docstrings, strict=False
        ):
            enum_member = getattr(cls, member_name)
            enum_member.__doc__ = doc_str.strip()

        return cls


class QoAttribute(StrEnum, metaclass=MetaEnum):
    pass


class DataQualityEnum(QoAttribute):
    ACCURACY = "accuracy"
    """The ratio between correct and total data the service received (%)"""

    COMPLETENESS = "completeness"
    """The ratio between received and expected number of data attributes sent to the service"""

    TOTAL_ERRORS = "total_errors"
    """Total number of errors as given by the user in the data"""

    ERROR_RATIOS = "error_ratios"
    """Ratio of errors in the data"""

    DUPLICATE_RATIO = "duplicate_ratio"
    """Ratio of duplicate entries in the data"""

    TOTAL_DUPLICATE = "total_duplicate"
    """Total number of duplicate entries in the data"""

    NULL_COUNT = "null_count"
    """Count of null or undefined entries in the data"""

    NULL_CORRELATIONS = "null_correlations"
    """Correlation between null or undefined entries in the data"""

    TOTAL_VALID = "total_valid"
    """Total number of valid entries in the data"""

    TOTAL_NONE = "total_none"
    """Total number of none or empty entries in the data"""

    NONE_RATIO = "none_ratio"
    """Ratio of none or empty entries in the data"""


class MLModelQualityEnum(QoAttribute):
    AUC = "auc"
    """The measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve"""

    ACCURACY = "accuracy"
    """Can be measured in different ways such as confidence score in classification models"""

    MSE = "mse"
    """Mean square error used for regression models"""

    PRECISION = "precision"
    """The fraction of true positive responses over total number of positive responses"""

    RECALL = "recall"
    """The fraction of true positive responses over total number of correct responses"""


class ServiceQualityEnum(QoAttribute):
    AVAILABILITY = "availability"
    """The ratio between up time and down time of the service (%)"""

    RELIABILITY = "reliability"
    """The ratio between correct and total service responses (%)"""

    RESPONSE_TIME = "response_time"
    """The response time of each microservice, measured by the time period between receiving and replying the request (s)"""
