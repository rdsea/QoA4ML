# This library is built based on ydata_quality: https://github.com/ydataai/ydata-quality

import io
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

from ..lang.datamodel_enum import ImageQualityNameEnum
from ..utils.qoa_utils import is_numpyarray, is_pddataframe, qoaLogger

p_dir = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(p_dir))

# Define metric names, return formats: dictionary {metric name} {sub-element}
# Return error/debugging
################################################ DATA QUALITY ########################################################


def eva_erronous(data, errors=None):
    """
    Return number/percentage of error data
    data: numpy array or pandas data frame
    errors: list of item considered as error
    ratio: return percentage if set to True
    sum: sum the result if set to True, otherwise return errors following the categories in list of 'errors'
    """
    try:
        if "ErroneousDataIdentifier" not in globals():
            global ErroneousDataIdentifier
            from ydata_quality.erroneous_data import ErroneousDataIdentifier
        if is_numpyarray(data):
            data = pd.DataFrame(data)
        if is_pddataframe(data):
            if errors and isinstance(errors, list):
                eva_err = ErroneousDataIdentifier(df=data, ed_extensions=errors)
            else:
                eva_err = ErroneousDataIdentifier(df=data)
            error_df = eva_err.predefined_erroneous_data()

            total_count = data.count().to_numpy().flatten().sum()
            results = {}
            results["totalErrors"] = error_df.to_numpy().flatten().sum()
            results["errorRatio"] = 100 * error_df / total_count
            return results
        else:
            qoaLogger.warning(f"Unsupported data: {type(data)}")
            return None
    except Exception as e:
        qoaLogger.error(f"Error {type(e)} in eva_erronous: {e.__traceback__}")
        traceback.print_exception(*sys.exc_info())


def eva_duplicate(data):
    """
    Return data/percentage of duplicate
    data: numpy array or pandas data frame
    ratio: return percentage if set to True
    """
    try:
        if "DuplicateChecker" not in globals():
            global DuplicateChecker
            from ydata_quality.duplicates import DuplicateChecker
        if is_numpyarray(data):
            data = pd.DataFrame(data)
        if is_pddataframe(data):
            dc = DuplicateChecker(df=data)
            dc_eva = dc.exact_duplicates()
            results = {}
            results["duplicateRatio"] = 100 * len(dc_eva.index) / len(data.index)
            results["totalDuplicate"] = len(dc_eva.index)
            return results
        else:
            qoaLogger.warning(f"Unsupported data: {type(data)}")
            return None
    except Exception as e:
        qoaLogger.error(f"Error {type(e)} in eva_duplicate: {e.__traceback__}")
        traceback.print_exception(*sys.exc_info())


def eva_missing(
    data, null_count=True, correlations=False, predict=False, random_state=0
):
    try:
        if "MissingsProfiler" not in globals():
            global MissingsProfiler
            from ydata_quality.missing import MissingsProfiler
        if is_numpyarray(data):
            data = pd.DataFrame(data)
        if is_pddataframe(data):
            mp = MissingsProfiler(df=data, random_state=random_state)
            results = {}
            if null_count:
                results["nullCount"] = mp.null_count()
            if correlations:
                results["correlations"] = mp.missing_correlations()
            if predict:
                results["missingPrediction"] = mp.predict_missings()
            return results
        else:
            qoaLogger.warning(f"Unsupported data: {type(data)}")
            return None
    except Exception as e:
        qoaLogger.error(f"Error {type(e)} in eva_erronous: {e.__traceback__}")
        traceback.print_exception(*sys.exc_info())


class OutlierDetector:
    def __init__(self, data):
        self.data = None
        self.update_data(data)

    def detect_outlier(self, n_data, labels=None, random_state=0, n=10, cluster=False):
        if labels is None:
            labels = []
        if is_numpyarray(n_data):
            n_data = pd.DataFrame(n_data)
        if is_pddataframe(n_data):
            if self.data is not None:
                data = None
                try:
                    data = pd.concat([self.data, n_data])
                except Exception as e:
                    qoaLogger.error(
                        f"Error {type(e)} in concatenating data: {e.__traceback__}"
                    )
                    traceback.print_exception(*sys.exc_info())
                if data is not None:
                    results = {}
                    for label in labels:
                        try:
                            if "LabelInspector" not in globals():
                                global LabelInspector
                                from ydata_quality.labelling import LabelInspector
                            li = LabelInspector(
                                df=data, label=label, random_state=random_state
                            )
                            results[label] = li.outlier_detection(
                                th=n, use_clusters=cluster
                            )
                        except Exception as e:
                            qoaLogger.error(
                                f"Error {type(e)} in LabelInspector: {e.__traceback__}"
                            )
                            traceback.print_exception(*sys.exc_info())
                    return results
                else:
                    return {"Error": "Cannot concatenate data"}
            else:
                return {"Error": "Historical data has not been set"}
        else:
            return {"Error": f"Unsupported data: {type(data)}"}

    def update_data(self, data):
        if is_numpyarray(data):
            data = pd.DataFrame(data)
        if is_pddataframe(data):
            self.data = data
            return {"Response": "Success"}
        else:
            return {"Error": f"Unsupported data: {type(data)}"}


def image_quality(image):
    if "PIL" not in globals():
        global PIL
        import PIL
    quality = {}
    if isinstance(image, bytes):
        image = PIL.Image.open(io.BytesIO(image))
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    if isinstance(image, (PIL.Image.Image, PIL.JpegImagePlugin.JpegImageFile)):
        # qoaLogger.debug(dir(image)
        quality[ImageQualityNameEnum.image_size] = image.size
        quality[ImageQualityNameEnum.color_mode] = image.mode
        quality[ImageQualityNameEnum.color_channel] = len(image.getbands())
    return quality


def eva_none(data):
    try:
        if is_pddataframe(data):
            data = data.to_numpy()
        if is_numpyarray(data):
            valid_count = np.count_nonzero(~np.isnan(data))
            none_count = np.count_nonzero(np.isnan(data))
            results = {}
            results["totalValid"] = valid_count
            results["totalNone"] = none_count
            results["noneRatio"] = valid_count / (valid_count + none_count)
            return results
        else:
            qoaLogger.warning(f"Unsupported data: {type(data)}")
            return None
    except Exception as e:
        qoaLogger.error(f"Error {type(e)} in eva_none: {e.__traceback__}")
        traceback.print_exception(*sys.exc_info())
