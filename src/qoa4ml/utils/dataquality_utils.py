import io

import numpy as np
import pandas as pd
from fastapi import UploadFile
from PIL import Image

from qoa4ml.lang.attributes import DataQualityEnum
from qoa4ml.lang.datamodel_enum import ImageQualityNameEnum
from qoa4ml.utils.logger import qoa_logger


# TODO: add citation for each metric implementation
# Abstract class for return value
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


def image_quality(input_image: bytes | np.ndarray):
    """
    Assess various quality metrics of an input image.

    Parameters:
    -----------
    input_image : bytes or np.ndarray
        The input image in either byte format or as a numpy array.

    Returns:
    --------
    dict
        A dictionary containing the following keys:
          - ImageQualityNameEnum.image_size: The size of the image (width, height).
          - ImageQualityNameEnum.color_mode: The color mode of the image (e.g., 'RGB').
          - ImageQualityNameEnum.color_channel: The number of color channels in the image.
    """
    quality = {}
    if isinstance(input_image, bytes):
        image = Image.open(io.BytesIO(input_image))
    elif isinstance(input_image, np.ndarray):
        image = Image.fromarray(input_image)
    quality[ImageQualityNameEnum.image_size] = image.size
    quality[ImageQualityNameEnum.color_mode] = image.mode
    quality[ImageQualityNameEnum.color_channel] = len(image.getbands())
    return quality


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
                raise RuntimeWarning("Predict is enabled but not implemented yet")
                # results["missing_prediction"] = mp.predict_missings()

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
            results = {}
            results[DataQualityEnum.TOTAL_VALID] = valid_count
            results[DataQualityEnum.TOTAL_NONE] = none_count
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


#
#
# class OutlierDetector:
#     def __init__(self, data):
#         self.data = None
#         self.update_data(data)
#
#     def detect_outlier(self, n_data, labels=None, random_state=0, n=10, cluster=False):
#         if labels is None:
#             labels = []
#         if is_numpyarray(n_data):
#             n_data = pd.DataFrame(n_data)
#         if is_pddataframe(n_data):
#             if self.data is not None:
#                 data = None
#                 try:
#                     data = pd.concat([self.data, n_data])
#                 except Exception as e:
#                     qoa_logger.error(
#                         f"Error {type(e)} in concatenating data: {e.__traceback__}"
#                     )
#                     traceback.print_exception(*sys.exc_info())
#                 if data is not None:
#                     results = {}
#                     for label in labels:
#                         try:
#                             if "LabelInspector" not in globals():
#                                 global LabelInspector
#                                 from ydata_quality.labelling import LabelInspector
#                             li = LabelInspector(
#                                 df=data, label=label, random_state=random_state
#                             )
#                             results[label] = li.outlier_detection(
#                                 th=n, use_clusters=cluster
#                             )
#                         except Exception as e:
#                             qoa_logger.error(
#                                 f"Error {type(e)} in LabelInspector: {e.__traceback__}"
#                             )
#                             traceback.print_exception(*sys.exc_info())
#                     return results
#                 else:
#                     return {"Error": "Cannot concatenate data"}
#             else:
#                 return {"Error": "Historical data has not been set"}
#         else:
#             return {"Error": f"Unsupported data: {type(data)}"}
#
#     def update_data(self, data):
#         if is_numpyarray(data):
#             data = pd.DataFrame(data)
#         if is_pddataframe(data):
#             self.data = data
#             return {"Response": "Success"}
#         else:
#             return {"Error": f"Unsupported data: {type(data)}"}
