"""Regression tests for qoa4ml.utils.dataquality_utils."""

from __future__ import annotations

import warnings

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from qoa4ml.lang.attributes import DataQualityEnum  # noqa: E402
from qoa4ml.utils.dataquality_utils import (  # noqa: E402
    eva_duplicate,
    eva_erronous,
    eva_missing,
    eva_none,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 1.0],
            "b": [np.nan, 2.0, 3.0, np.nan],
        }
    )


class TestEvaMissing:
    def test_predict_flag_emits_warning_without_raising(self, sample_df):
        # Regression: previous `raise RuntimeWarning(...)` was caught by the
        # surrounding `except Exception` and silently returned None. The
        # warning must reach the user, and the rest of the result stays intact.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = eva_missing(sample_df, null_count=True, predict=True)

        assert result is not None
        assert DataQualityEnum.NULL_COUNT in result
        predict_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert predict_warnings, "RuntimeWarning must be emitted when predict=True"

    def test_null_count_returns_series(self, sample_df):
        result = eva_missing(sample_df, null_count=True)
        assert result is not None
        counts = result[DataQualityEnum.NULL_COUNT]
        assert int(counts["a"]) == 1
        assert int(counts["b"]) == 2


class TestEvaErronousSignature:
    def test_errors_list_is_second_positional(self, sample_df):
        # Regression: doc claimed signature `eva_erronous(data, columns)`;
        # real signature is `eva_erronous(data, errors=None)`. This call
        # verifies the real signature accepts a value list.
        result = eva_erronous(sample_df, [1.0])
        assert result is not None
        assert int(result[DataQualityEnum.TOTAL_ERRORS]) == 2


class TestEvaNoneSignature:
    def test_accepts_only_data(self, sample_df):
        # Regression: doc claimed `eva_none(data, columns)`; real signature
        # takes only `data`.
        result = eva_none(sample_df)
        assert result is not None
        assert DataQualityEnum.TOTAL_VALID in result
        assert DataQualityEnum.TOTAL_NONE in result


class TestEvaDuplicate:
    def test_counts_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 2, 1, 3], "b": [1, 2, 1, 3]})
        result = eva_duplicate(df)
        assert result is not None
        assert int(result[DataQualityEnum.TOTAL_DUPLICATE]) == 1


class TestImageQuality:
    def test_numpy_rgb_image(self):
        pytest.importorskip("PIL")

        from qoa4ml.lang.datamodel_enum import ImageQualityNameEnum
        from qoa4ml.utils.dataquality_utils import image_quality

        arr = np.zeros((4, 8, 3), dtype=np.uint8)
        result = image_quality(arr)
        assert result[ImageQualityNameEnum.image_size] == (8, 4)
        assert result[ImageQualityNameEnum.color_mode] == "RGB"
        assert result[ImageQualityNameEnum.color_channel] == 3

    def test_rejects_unsupported_type(self):
        from qoa4ml.utils.dataquality_utils import image_quality

        # Regression: the previous implementation silently left `image`
        # unbound when given an unsupported type, which then surfaced as
        # `UnboundLocalError`. Now the function raises TypeError.
        with pytest.raises(TypeError):
            image_quality("not an image")  # type: ignore[arg-type]
