from typing import Any

from qoa4ml.utils.logger import qoa_logger
from qoa4ml.utils.qoa_utils import is_numpyarray

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import tensorflow as tf
except ImportError:
    tf = None  # type: ignore[assignment]


_ML_EXTRA_HINT = (
    "tensorflow is required for qoa4ml.probes.mlquality; "
    "install with `pip install qoa4ml[ml]`"
)


def _require_tf() -> None:
    """Raise a clear ImportError instead of surfacing an AttributeError."""
    if tf is None:
        raise ImportError(_ML_EXTRA_HINT)


def _require_numpy() -> None:
    if np is None:
        raise ImportError(
            "numpy is required for classification_confidence; "
            "install with `pip install qoa4ml[ml]`"
        )


def timeseries_metric(model: Any) -> dict[str, Any]:
    """Retrieve all metrics from a Keras Sequential timeseries model.

    Returns a mapping of ``metric_name -> value``. Returns ``{}`` if
    ``model`` is not a Keras Sequential. Raises ``ImportError`` if
    TensorFlow is not installed.
    """
    _require_tf()
    metrics: dict[str, Any] = {}
    try:
        if isinstance(model, tf.keras.Sequential):
            for metric in model.metrics:
                metrics[metric.name] = _to_jsonable(metric.result().numpy())
        return metrics
    except (AttributeError, RuntimeError, TypeError) as error:
        qoa_logger.exception(f"timeseries_metric failed ({type(error).__name__})")
        return {"Error": "Unable to get metrics"}


def ts_inference_metric(model: Any, name: str) -> dict[str, Any]:
    """Retrieve a single inference metric by name; empty dict if absent."""
    _require_tf()
    try:
        metrics = timeseries_metric(model)
        if "Error" in metrics:
            return {"Error": metrics["Error"]}
        if name in metrics:
            return {name: metrics[name]}
        return {}
    except (AttributeError, RuntimeError, TypeError, KeyError) as error:
        qoa_logger.exception(
            f"ts_inference_metric({name!r}) failed ({type(error).__name__})"
        )
        return {"Error": f"Unable to get model {name}"}


def ts_inference_mae(model: Any) -> dict[str, Any]:
    """Retrieve the mean-absolute-error metric from a timeseries model.

    Returns ``{"mae": <value>}`` or ``{}`` if the model has no MAE metric.
    """
    result = ts_inference_metric(model, "mean_absolute_error")
    if "mean_absolute_error" in result:
        return {"mae": result["mean_absolute_error"]}
    return result


def ts_inference_loss(model: Any) -> dict[str, Any]:
    """Retrieve the loss metric from a timeseries model.

    Returns ``{"loss": <value>}`` or ``{}`` if the model has no loss metric.
    """
    return ts_inference_metric(model, "loss")


def training_metric(model: Any) -> dict[str, Any]:
    """Retrieve the full training history from a Keras Sequential model.

    Returns the Keras ``History.history`` dict on success, ``{}`` when the
    model is not a Keras Sequential, or ``{"Error": "..."}`` on failure.
    """
    _require_tf()
    try:
        if isinstance(model, tf.keras.Sequential):
            return model.history.history
        return {}
    except (AttributeError, RuntimeError, TypeError) as error:
        qoa_logger.exception(f"training_metric failed ({type(error).__name__})")
        return {"Error": "Unable to get training metrics"}


def training_loss(model: Any) -> dict[str, Any]:
    """Retrieve the training loss history from a Keras Sequential model.

    Returns ``{"loss": [...]}`` on success, ``{}`` when the model is not a
    Keras Sequential, or ``{"Error": "..."}`` on an unexpected failure.
    """
    _require_tf()
    try:
        if isinstance(model, tf.keras.Sequential):
            return {"loss": model.history.history["loss"]}
        return {}
    except (AttributeError, RuntimeError, TypeError, KeyError) as error:
        qoa_logger.exception(f"training_loss failed ({type(error).__name__})")
        return {"Error": "Unable to get training loss"}


def training_val_accuracy(model: Any) -> dict[str, Any]:
    """Retrieve the validation accuracy history.

    Returns ``{"val_accuracy": [...]}`` on success, ``{}`` when the model
    is not a Keras Sequential, or ``{"Error": "..."}`` on failure.
    """
    _require_tf()
    try:
        if isinstance(model, tf.keras.Sequential):
            return {"val_accuracy": model.history.history["val_accuracy"]}
        return {}
    except (AttributeError, RuntimeError, TypeError, KeyError) as error:
        qoa_logger.exception(f"training_val_accuracy failed ({type(error).__name__})")
        return {"Error": "Unable to get validation accuracy"}


def training_val_loss(model: Any) -> dict[str, Any]:
    """Retrieve the validation loss history.

    Returns ``{"val_loss": [...]}`` on success, ``{}`` when the model is
    not a Keras Sequential, or ``{"Error": "..."}`` on failure.
    """
    _require_tf()
    try:
        if isinstance(model, tf.keras.Sequential):
            return {"val_loss": model.history.history["val_loss"]}
        return {}
    except (AttributeError, RuntimeError, TypeError, KeyError) as error:
        qoa_logger.exception(f"training_val_loss failed ({type(error).__name__})")
        return {"Error": "Unable to get validation loss"}


def classification_confidence(data: Any, score: bool = True) -> dict[str, Any]:
    """Compute classification confidence from model output scores or logits."""
    _require_numpy()
    try:
        if score:
            return {"confidence": float(100 * np.max(data))}
        if is_numpyarray(data):
            _require_tf()
            scores = tf.nn.softmax(data[0])
            return {"confidence": float(100 * np.max(scores))}
        return {"Error": f"Unsupported data: {type(data)}"}
    except (ValueError, TypeError, RuntimeError) as error:
        qoa_logger.exception(
            f"classification_confidence failed ({type(error).__name__})"
        )
        return {"Error": "Unable to get classification confidence"}


def _to_jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays into JSON-serialisable Python values."""
    if np is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
