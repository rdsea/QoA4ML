# Metric Functions

QoA4ML provides built-in metric functions for evaluating data quality, ML model quality, and system resources.

## Data Quality Metrics

These functions are in `qoa4ml.utils.dataquality_utils`.

### Tabular Data

| Function | Description | Input | Output |
|---|---|---|---|
| `eva_erronous(data, errors=None)` | Count entries equal to any value in `errors` (defaults to NaN). | DataFrame or ndarray; optional list of values treated as errors | Dict with `TOTAL_ERRORS` and `ERROR_RATIOS` |
| `eva_duplicate(data)` | Detect duplicate rows. | DataFrame or ndarray | Dict with `DUPLICATE_RATIO` and `TOTAL_DUPLICATE` |
| `eva_missing(data, null_count=True, correlations=False, predict=False)` | Count missing values per column and optionally compute their correlation matrix. `predict=True` is reserved for future use and emits a `RuntimeWarning`. | DataFrame or ndarray + bool flags | Dict with `NULL_COUNT`, and `NULL_CORRELATIONS` when `correlations=True` |
| `eva_none(data)` | Summarize NaN vs valid numeric entries. | DataFrame or ndarray | Dict with `TOTAL_VALID`, `TOTAL_NONE`, `NONE_RATIO` |

### Image Data

| Function | Description | Input | Output |
|---|---|---|---|
| `image_quality(input_image)` | Evaluate image properties. | `bytes` (encoded image) or `numpy.ndarray` | Dict keyed by `ImageQualityNameEnum` with `image_size` (`(width, height)` tuple), `color_mode`, `color_channel` |

## ML Model Quality Metrics

These functions are in `qoa4ml.probes.mlquality`. They extract metrics from TensorFlow/Keras models. TensorFlow is an optional dependency; install it via `qoa4ml[ml]`.

### Inference Metrics

| Function | Description | Input | Output |
|---|---|---|---|
| `timeseries_metric(model)` | Get all metrics from a Keras Sequential model | `tf.keras.Sequential` | Dict of metric name to value |
| `ts_inference_metric(model, name)` | Get a specific metric by name | Model, metric name | Dict with the named metric |
| `ts_inference_mae(model)` | Get mean absolute error | Model | Dict with MAE value |
| `ts_inference_loss(model)` | Get loss value | Model | Dict with loss value |

### Training History Metrics

| Function | Description | Input | Output |
|---|---|---|---|
| `training_metric(model)` | Get full training history | `tf.keras.Sequential` | Dict of metric name to list of values per epoch |
| `training_loss(model)` | Get training loss history | Model | Dict with loss history |
| `training_val_accuracy(model)` | Get validation accuracy history | Model | Dict with `val_accuracy` history |
| `training_val_loss(model)` | Get validation loss history | Model | Dict with `val_loss` history |

## QoA Attribute Enums

Quality attributes are defined as enums in `qoa4ml.lang.attributes`. The tables below list every member defined in the source; do not assume additional members exist.

### Data Quality Attributes (`DataQualityEnum`)

| Attribute | Value | Description |
|---|---|---|
| `ACCURACY` | `data_accuracy` | Ratio between correct and total data received (%); namespaced to avoid colliding with `MLModelQualityEnum.ACCURACY` |
| `COMPLETENESS` | `completeness` | Ratio between received and expected data attributes |
| `TOTAL_ERRORS` | `total_errors` | Total number of erroneous data points |
| `ERROR_RATIOS` | `error_ratios` | Ratio of erroneous entries |
| `DUPLICATE_RATIO` | `duplicate_ratio` | Ratio of duplicate records |
| `TOTAL_DUPLICATE` | `total_duplicate` | Total number of duplicate entries |
| `NULL_COUNT` | `null_count` | Per-column count of null / NaN values |
| `NULL_CORRELATIONS` | `null_correlations` | Correlation matrix of null values |
| `TOTAL_VALID` | `total_valid` | Count of valid (non-NaN) numeric entries |
| `TOTAL_NONE` | `total_none` | Count of None / NaN numeric entries |
| `NONE_RATIO` | `none_ratio` | Ratio of valid entries (%) |

### ML Model Quality Attributes (`MLModelQualityEnum`)

| Attribute | Value | Description |
|---|---|---|
| `AUC` | `auc` | Area under the ROC curve |
| `ACCURACY` | `model_accuracy` | Correct predictions over total predictions; namespaced to avoid colliding with `DataQualityEnum.ACCURACY` |
| `MSE` | `mse` | Mean squared error (regression models) |
| `PRECISION` | `precision` | True positives over predicted positives |
| `RECALL` | `recall` | True positives over actual positives |

### Service Quality Attributes (`ServiceQualityEnum`)

| Attribute | Value | Description |
|---|---|---|
| `AVAILABILITY` | `availability` | Fraction of time the service is operational |
| `RELIABILITY` | `reliability` | Probability the service responds correctly |
| `RESPONSE_TIME` | `response_time` | Time for the service to respond to a request (s) |

## Resource Utilization Metrics

System and process resource metrics collected by probes:

| Metric | Description | Source |
|---|---|---|
| CPU usage (%) | Per-core and aggregate CPU utilization | `SystemMonitoringProbe` |
| Memory usage | RSS, VMS, and percentage | `ProcessMonitoringProbe` |
| GPU usage (%) | Core and memory utilization per device | `SystemMonitoringProbe` (NVIDIA) |
| Docker stats | Container CPU and memory usage | `DockerMonitoringProbe` |
| Network I/O | Bytes sent / received | `system_report()` utility |

## Usage Example

```python
from qoa4ml.lang.attributes import (
    DataQualityEnum,
    MLModelQualityEnum,
    ServiceQualityEnum,
)
from qoa4ml.qoa_client import QoaClient

client = QoaClient(config_path="config/client.yaml")

# Service-quality metric (category=0 -> service)
client.observe_metric(ServiceQualityEnum.RESPONSE_TIME, 0.125, category=0)

# Data-quality metric (category=1 -> data)
client.observe_metric(DataQualityEnum.ACCURACY, 0.97, category=1)

# Inference metric (ml-specific, no category)
client.observe_inference_metric(MLModelQualityEnum.ACCURACY, 0.92)

# Generate report
report = client.report()
```

`observe_metric(metric_name, value, category=0, description="")` accepts a `category` integer: `0` for service, `1` for data, `2` for security. Any other value raises `RuntimeError`.
