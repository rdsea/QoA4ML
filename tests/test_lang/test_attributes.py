from enum import StrEnum

from qoa4ml.lang.attributes import (
    QOA_ATTRIBUTES_NAME,
    QOA_ATTRIBUTES_VERSION,
    DataQualityEnum,
    MetaEnum,
    MLModelQualityEnum,
    QoAttribute,
    ServiceQualityEnum,
)


class TestConstants:
    def test_version(self):
        assert QOA_ATTRIBUTES_VERSION == "v0.3"

    def test_name(self):
        assert QOA_ATTRIBUTES_NAME == "qoa4ml-attributes"


class TestQoAttribute:
    def test_is_strenum(self):
        assert issubclass(QoAttribute, StrEnum)

    def test_uses_meta_enum(self):
        assert type(QoAttribute) is MetaEnum


class TestDataQualityEnum:
    def test_all_members_exist(self):
        expected = {
            "ACCURACY",
            "COMPLETENESS",
            "TOTAL_ERRORS",
            "ERROR_RATIOS",
            "DUPLICATE_RATIO",
            "TOTAL_DUPLICATE",
            "NULL_COUNT",
            "NULL_CORRELATIONS",
            "TOTAL_VALID",
            "TOTAL_NONE",
            "NONE_RATIO",
        }
        actual = set(DataQualityEnum._member_names_)
        assert actual == expected

    def test_values(self):
        # Regression: `DataQualityEnum.ACCURACY` used to share the value
        # "accuracy" with `MLModelQualityEnum.ACCURACY`, causing silent
        # coercion collisions when Pydantic hydrated a `MetricNameEnum`
        # union from JSON. Values are now namespaced.
        assert DataQualityEnum.ACCURACY == "data_accuracy"
        assert DataQualityEnum.COMPLETENESS == "completeness"
        assert DataQualityEnum.TOTAL_ERRORS == "total_errors"
        assert DataQualityEnum.NULL_COUNT == "null_count"
        assert DataQualityEnum.NONE_RATIO == "none_ratio"

    def test_is_strenum(self):
        assert isinstance(DataQualityEnum.ACCURACY, str)

    def test_docstrings_assigned(self):
        assert DataQualityEnum.ACCURACY.__doc__ is not None
        assert "correct" in DataQualityEnum.ACCURACY.__doc__.lower()

    def test_completeness_docstring(self):
        assert DataQualityEnum.COMPLETENESS.__doc__ is not None
        assert "received" in DataQualityEnum.COMPLETENESS.__doc__.lower()


class TestMLModelQualityEnum:
    def test_all_members_exist(self):
        expected = {"AUC", "ACCURACY", "MSE", "PRECISION", "RECALL"}
        actual = set(MLModelQualityEnum._member_names_)
        assert actual == expected

    def test_values(self):
        assert MLModelQualityEnum.AUC == "auc"
        # Regression: previously "accuracy", which collided with DataQualityEnum.
        assert MLModelQualityEnum.ACCURACY == "model_accuracy"
        assert MLModelQualityEnum.MSE == "mse"
        assert MLModelQualityEnum.PRECISION == "precision"
        assert MLModelQualityEnum.RECALL == "recall"

    def test_accuracy_does_not_collide_with_data_quality(self):
        # The two ACCURACY members must have distinct string values so
        # Pydantic coercion through MetricNameEnum can round-trip cleanly.
        from qoa4ml.lang.attributes import DataQualityEnum

        assert MLModelQualityEnum.ACCURACY.value != DataQualityEnum.ACCURACY.value

    def test_docstrings_assigned(self):
        assert MLModelQualityEnum.AUC.__doc__ is not None
        assert "classifier" in MLModelQualityEnum.AUC.__doc__.lower()

    def test_precision_docstring(self):
        assert MLModelQualityEnum.PRECISION.__doc__ is not None
        assert "positive" in MLModelQualityEnum.PRECISION.__doc__.lower()


class TestServiceQualityEnum:
    def test_all_members_exist(self):
        expected = {"AVAILABILITY", "RELIABILITY", "RESPONSE_TIME"}
        actual = set(ServiceQualityEnum._member_names_)
        assert actual == expected

    def test_values(self):
        assert ServiceQualityEnum.AVAILABILITY == "availability"
        assert ServiceQualityEnum.RELIABILITY == "reliability"
        assert ServiceQualityEnum.RESPONSE_TIME == "response_time"

    def test_docstrings_assigned(self):
        assert ServiceQualityEnum.AVAILABILITY.__doc__ is not None
        assert "up time" in ServiceQualityEnum.AVAILABILITY.__doc__.lower()

    def test_response_time_docstring(self):
        doc = ServiceQualityEnum.RESPONSE_TIME.__doc__
        assert doc is not None
        assert "response time" in doc.lower()


class TestMetaEnumDocstringAssignment:
    """Verify that MetaEnum properly assigns docstrings from source code."""

    def test_all_data_quality_members_have_docs(self):
        for member_name in DataQualityEnum._member_names_:
            member = getattr(DataQualityEnum, member_name)
            assert member.__doc__ is not None, f"{member_name} should have a docstring"
            assert len(member.__doc__.strip()) > 0, (
                f"{member_name} docstring should not be empty"
            )

    def test_all_ml_model_quality_members_have_docs(self):
        for member_name in MLModelQualityEnum._member_names_:
            member = getattr(MLModelQualityEnum, member_name)
            assert member.__doc__ is not None, f"{member_name} should have a docstring"

    def test_all_service_quality_members_have_docs(self):
        for member_name in ServiceQualityEnum._member_names_:
            member = getattr(ServiceQualityEnum, member_name)
            assert member.__doc__ is not None, f"{member_name} should have a docstring"
