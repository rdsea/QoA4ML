from typing import TYPE_CHECKING, Any

import lazy_import

from qoa4ml.utils.logger import qoa_logger
from qoa4ml.utils.qoa_utils import is_numpyarray

np = lazy_import.lazy_module("numpy")
tf = lazy_import.lazy_module("tensorflow")
if TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf


def timeseries_metric(model: Any) -> dict[str, Any]:
    """Retrieve all metrics from a Keras Sequential timeseries model."""
    metrics: dict[str, Any] = {}
    try:
        if isinstance(model, tf.keras.Sequential):
            for metric in model.metrics:
                metrics[metric.name] = metric.result().numpy()
        return metrics
    except (AttributeError, RuntimeError, TypeError) as e:
        qoa_logger.exception("Error %s when querying timeseries model metrics", type(e))
        return {"Error": "Unable to get metrics"}


def ts_inference_metric(model: Any, name: str) -> dict[str, Any]:
    """Retrieve a specific inference metric by name from a timeseries model."""
    try:
        metrics = timeseries_metric(model)
        results: dict[str, Any] = {}
        if name in metrics:
            results[name] = metrics[name]
        if "Error" in metrics:
            results["Error"] = metrics["Error"]
        return results
    except (AttributeError, RuntimeError, TypeError, KeyError) as e:
        qoa_logger.exception("Error %s when querying timeseries %s", type(e), name)
        return {"Error": f"Unable to get model {name}"}


def ts_inference_mae(model: Any) -> dict[str, Any]:
    """Retrieve the mean absolute error metric from a timeseries model."""
    try:
        metrics = ts_inference_metric(model, "mean_absolute_error")
        return {"MAE": metrics}
    except (AttributeError, RuntimeError, TypeError, KeyError) as e:
        qoa_logger.exception(
            "Error %s when querying timeseries mean absolute error", type(e)
        )
        return {"Error": "Unable to get model mean absolute error"}


def ts_inference_loss(model: Any) -> dict[str, Any]:
    """Retrieve the loss metric from a timeseries model."""
    try:
        metrics = ts_inference_metric(model, "loss")
        return {"Loss": metrics}
    except (AttributeError, RuntimeError, TypeError, KeyError) as e:
        qoa_logger.exception("Error %s when querying timeseries loss", type(e))
        return {"Error": "Unable to get model loss"}


def training_metric(model: Any) -> dict[str, Any] | None:
    """Retrieve the full training history from a Keras Sequential model."""
    try:
        if isinstance(model, tf.keras.Sequential):
            return model.history.history
        else:
            return None
    except (AttributeError, RuntimeError, TypeError) as e:
        qoa_logger.exception("Error %s when querying training metrics", type(e))
        return None


def training_loss(model: Any) -> dict[str, Any] | None:
    """Retrieve the training loss history from a Keras Sequential model."""
    try:
        if isinstance(model, tf.keras.Sequential):
            return {"Training Loss": model.history.history["loss"]}
        else:
            return None
    except (AttributeError, RuntimeError, TypeError, KeyError) as e:
        qoa_logger.exception("Error %s when querying training loss", type(e))
        return {"Error": "Unable to get training loss"}


def training_val_accuracy(model: Any) -> dict[str, Any] | None:
    """Retrieve the validation accuracy history from a Keras Sequential model."""
    try:
        if isinstance(model, tf.keras.Sequential):
            return {"Evaluate Accuracy": model.history.history["val_accuracy"]}
        else:
            return None
    except (AttributeError, RuntimeError, TypeError, KeyError) as e:
        qoa_logger.exception(
            "Error %s when querying training validation accuracy", type(e)
        )
        return {"Error": "Unable to get validation accuracy"}


def training_val_loss(model: Any) -> dict[str, Any] | None:
    """Retrieve the validation loss history from a Keras Sequential model."""
    try:
        if isinstance(model, tf.keras.Sequential):
            return {"Evaluate Loss": model.history.history["val_loss"]}
        else:
            return None
    except (AttributeError, RuntimeError, TypeError, KeyError) as e:
        qoa_logger.exception("Error %s when querying training validation loss", type(e))
        return {"Error": "Unable to get validation loss"}


def classification_confidence(data: Any, score: bool = True) -> dict[str, Any]:
    """Compute classification confidence from model output scores or logits."""
    try:
        if score:
            return {"Confidence": 100 * np.max(data)}
        elif is_numpyarray(data):
            scores = tf.nn.softmax(data[0])
            return {"Confidence": 100 * np.max(scores)}
        else:
            return {"Error": f"Unsupported data: {type(data)}"}
    except (ValueError, TypeError, RuntimeError) as e:
        qoa_logger.exception(
            "Error %s in extracting classification confidence", type(e)
        )
        return {"Error": "Unable to get classification confidence"}
