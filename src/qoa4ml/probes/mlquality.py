import sys
import traceback
from typing import TYPE_CHECKING

import lazy_import

from qoa4ml.utils.logger import qoa_logger
from qoa4ml.utils.qoa_utils import is_numpyarray

np = lazy_import.lazy_module("numpy")
tf = lazy_import.lazy_module("tensorflow")
if TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf


def timeseries_metric(model):
    metrics = {}
    try:
        if isinstance(model, tf.keras.Sequential):
            for metric in model.metrics:
                metrics[metric.name] = metric.result().numpy()
        return metrics
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} when querying timeseries model metrics")
        return {"Error": "Unable to get metrics"}


def ts_inference_metric(model, name):
    try:
        metrics = timeseries_metric(model)
        results = {}
        if name in metrics:
            results[name] = metrics[name]
        if "Error" in metrics:
            results["Error"] = metrics["Error"]
        return results
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} when querying timeseries {name}")
        return {"Error": f"Unable to get model {name}"}


def ts_inference_mae(model):
    try:
        metrics = ts_inference_metric(model, "mean_absolute_error")
        return {"MAE": metrics}
    except Exception as e:
        qoa_logger.exception(
            f"Error {type(e)} when querying timeseries mean absolute error"
        )
        return {"Error": "Unable to get model mean absolute error"}


def ts_inference_loss(model):
    try:
        metrics = ts_inference_metric(model, "loss")
        return {"Loss": metrics}
    except Exception as e:
        qoa_logger.exception(
            f"Error {type(e)} when querying timeseries mean absolute error"
        )
        return {"Error": "Unable to get model mean absolute error"}


def training_metric(model):
    try:
        if isinstance(model, tf.keras.Sequential):
            return model.history.history
        else:
            return None
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} when querying training metrics")
        return {"Error": "Unable to get training metrics"}


def training_loss(model):
    try:
        if isinstance(model, tf.keras.Sequential):
            return {"Training Loss": model.history.history["loss"]}
        else:
            return None
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} when querying training loss")
        return {"Error": "Unable to get training loss"}


def training_val_accuracy(model):
    try:
        if isinstance(model, tf.keras.Sequential):
            return {"Evaluate Accuracy": model.history.history["val_accuracy"]}
        else:
            return None
    except Exception as e:
        qoa_logger.exception(
            f"Error {type(e)} when querying training validation accuracy"
        )
        traceback.print_exception(*sys.exc_info())
        return {"Error": "Unable to get validation accuracy"}


def training_accuracy(model):
    try:
        if isinstance(model, tf.keras.Sequential):
            return {"Evaluate Loss": model.history.history["val_loss"]}
        else:
            return None
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} when querying training validation loss")
        return {"Error": "Unable to get validation loss"}


def classification_confidence(data, score=True):
    try:
        if score:
            return {"Confidence": 100 * np.max(data)}
        elif is_numpyarray(data):
            scores = tf.nn.softmax(data[0])
            return {"Confidence": 100 * np.max(scores)}
        else:
            return {"Error": f"Unsupported data: {type(data)}"}
    except Exception as e:
        qoa_logger.exception(f"Error {type(e)} in extracting classification confidence")
        return {"Error": "Unable to get classification confidence"}
