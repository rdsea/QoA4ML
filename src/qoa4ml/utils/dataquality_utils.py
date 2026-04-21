import io
import warnings
from typing import Any

import numpy as np
import pandas as pd
from fastapi import UploadFile
from PIL import Image

from qoa4ml.lang.attributes import DataQualityEnum
from qoa4ml.lang.datamodel_enum import ImageQualityNameEnum
from qoa4ml.utils.logger import qoa_logger


def eva_input_file_type(input_file: UploadFile, allowed_data_type: list[str]):
    """
    Check if the input file matches any of the allowed data types

    Parameters:
    -----------
    input_file : UploadFile
        The uploaded file object to be checked for data type.
    allowed_data_type : List[str]
        A list of allowed data types to compare against the content type of the input file.

    Returns:
    --------
    bool
        True if the content type of the input file is in the list of allowed data types,
        otherwise False.
    """
    return input_file.content_type in allowed_data_type


def image_quality(input_image: bytes | np.ndarray) -> dict[ImageQualityNameEnum, Any]:
    """
    Assess various quality metrics of an input image.

    Parameters
    ----------
    input_image : bytes or np.ndarray
        The input image in either byte format or as a numpy array.

    Returns
    -------
    dict
        A dictionary keyed by ``ImageQualityNameEnum`` with:
          - ``image_size``: tuple ``(width, height)``.
          - ``color_mode``: PIL color mode (e.g. ``"RGB"``).
          - ``color_channel``: number of color channels.

    Raises
    ------
    TypeError
        If ``input_image`` is neither ``bytes`` nor ``numpy.ndarray``.
    """
    if isinstance(input_image, bytes):
        image: Image.Image = Image.open(io.BytesIO(input_image))
    elif isinstance(input_image, np.ndarray):
        image = Image.fromarray(input_image)
    else:
        raise TypeError(
            f"image_quality expects bytes or numpy.ndarray, got {type(input_image).__name__}"
        )
    return {
        ImageQualityNameEnum.image_size: image.size,
        ImageQualityNameEnum.color_mode: image.mode,
        ImageQualityNameEnum.color_channel: len(image.getbands()),
    }


def eva_erronous(data: np.ndarray | pd.DataFrame, errors: list | None = None):
    """
    Evaluate and return the number or percentage of erroneous data entries.

    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Input data to be evaluated.
    errors : list, optional
        List of items considered as errors. If not provided, NaNs will be considered as errors.

    Returns:
    --------
    dict or None
        A dictionary containing the following keys if successful:
          - DataQualityEnum.total_errors: Total number of errors.
          - DataQualityEnum.error_ratios: Percentage of errors.
        Returns None if the input data type is unsupported or if an exception occurs.
    """
    try:
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        if isinstance(data, pd.DataFrame):
            if errors and isinstance(errors, list):
                error_mask = data.isin(errors)
            else:
                error_mask = data.isna()

            total_errors = error_mask.sum().sum()
            total_count = data.size

            results = {
                DataQualityEnum.TOTAL_ERRORS: total_errors,
                DataQualityEnum.ERROR_RATIOS: 100 * total_errors / total_count,
            }
            return results
        else:
            qoa_logger.warning(f"Unsupported data: {type(data)}")
            return None
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} in eva_erronous")
        return None


def eva_duplicate(data: np.ndarray | pd.DataFrame):
    """
    Evaluate and return the number or percentage of duplicate entries in the data.

    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Input data to be evaluated.

    Returns:
    --------
    dict or None
        A dictionary containing the following keys if successful:
          - DataQualityEnum.duplicate_ratio: Percentage of duplicate data.
          - DataQualityEnum.total_duplicate: Total number of duplicate entries.
        Returns None if the input data type is unsupported or if an exception occurs.
    """
    try:
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        if isinstance(data, pd.DataFrame):
            duplicate_mask = data.duplicated()
            duplicate_data = data[duplicate_mask]

            results = {
                DataQualityEnum.DUPLICATE_RATIO: 100
                * duplicate_data.shape[0]
                / data.shape[0],
                DataQualityEnum.TOTAL_DUPLICATE: duplicate_data.shape[0],
            }
            return results
        else:
            qoa_logger.warning(f"Unsupported data: {type(data)}")
            return None
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} in eva_duplicate")
        return None


def eva_missing(
    data: np.ndarray | pd.DataFrame, null_count=True, correlations=False, predict=False
):
    """
    Evaluate and return statistics about missing data in the dataset.

    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Input data to be evaluated.
    null_count : bool, default=True
        If True, return the count of missing values in each column.
    correlations : bool, default=False
        If True, return the correlation matrix of missing values.
    predict : bool, default=False
        If True, enable missing data prediction (not implemented).

    Returns:
    --------
    dict or None
        A dictionary containing:
          - DataQualityEnum.null_count: Count of missing values (if null_count is True).
          - DataQualityEnum.null_correlations: Correlation matrix of missing values (if correlations is True).
        Returns None if the input data type is unsupported or if an exception occurs.
    """
    try:
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            results = {}
            if null_count:
                count = data.isnull().sum()
                results[DataQualityEnum.NULL_COUNT] = count

            if correlations:
                nulls = data.loc[:, results[DataQualityEnum.NULL_COUNT] > 0]
                results[DataQualityEnum.NULL_CORRELATIONS] = nulls.isnull().corr()

            if predict:
                warnings.warn(
                    "Predict is enabled but not implemented yet",
                    RuntimeWarning,
                    stacklevel=2,
                )

            return results
        else:
            qoa_logger.warning(f"Unsupported data: {type(data)}")
            return None
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} in eva_missing")
        return None


def eva_none(data: np.ndarray | pd.DataFrame):
    """
    Evaluate and return statistics about valid and None (NaN) values in the dataset.

    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Input data to be evaluated.

    Returns:
    --------
    dict or None
        A dictionary containing the following keys if successful:
          - DataQualityEnum.total_valid: Total count of valid (non-NaN) entries.
          - DataQualityEnum.total_none: Total count of None (NaN) entries.
          - DataQualityEnum.none_ratio: Percentage of valid entries.
        Returns None if the input data type is unsupported or if an exception occurs.
    """
    try:
        if isinstance(data, pd.DataFrame):
            data_numeric = data.select_dtypes(include=[np.number])
            data = data_numeric.to_numpy()
        if isinstance(data, np.ndarray):
            valid_count = np.count_nonzero(~np.isnan(data))
            none_count = np.count_nonzero(np.isnan(data))
            results: dict[DataQualityEnum, float] = {}
            results[DataQualityEnum.TOTAL_VALID] = float(valid_count)
            results[DataQualityEnum.TOTAL_NONE] = float(none_count)
            results[DataQualityEnum.NONE_RATIO] = (
                100 * valid_count / (valid_count + none_count)
            )
            return results
        else:
            qoa_logger.warning(f"Unsupported data: {type(data)}")
            return None
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} in eva_none")
        return None
