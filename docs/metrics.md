# Metric Functions

QoA4ML provides built-in metric functions for evaluating data quality, ML model quality, and system resources.

## Data Quality Metrics

These functions are in `qoa4ml.utils.dataquality_utils`.

### Tabular Data

| Function | Description | Input | Output |
|---|---|---|---|
| `eva_erronous(data, columns)` | Evaluate erroneous values in specified columns | DataFrame or ndarray, column list | Dict with error counts and ratios per column |
| `eva_duplicate(data)` | Detect duplicate rows | DataFrame or ndarray | Dict with duplicate count and ratio |
| `eva_missing(data, columns)` | Evaluate missing values in specified columns | DataFrame or ndarray, column list | Dict with missing counts and ratios per column |
| `eva_none(data, columns)` | Evaluate None/null values in specified columns | DataFrame or ndarray, column list | Dict with None counts and ratios per column |

### Image Data

| Function | Description | Input | Output |
|---|---|---|---|
| `image_quality(image)` | Evaluate image quality metrics | PIL Image, numpy array, or bytes | Dict with width, height, size, color channels, bit depth, entropy, SNR |

## ML Model Quality Metrics

These functions are in `qoa4ml.probes.mlquality`. They extract metrics from TensorFlow/Keras models.

### Inference Metrics

| Function | Description | Input | Output |
|---|---|---|---|
| `timeseries_metric(model)` | Get all metrics from a Keras Sequential model | tf.keras.Sequential | Dict of metric name -> value |
| `ts_inference_metric(model, name)` | Get a specific metric by name | Model, metric name | Dict with the named metric |
| `ts_inference_mae(model)` | Get mean absolute error | Model | Dict with MAE value |
| `ts_inference_loss(model)` | Get loss value | Model | Dict with loss value |

### Training History Metrics

| Function | Description | Input | Output |
|---|---|---|---|
| `training_metric(model)` | Get full training history | tf.keras.Sequential | Dict of metric name -> list of values per epoch |
| `training_loss(model)` | Get training loss history | Model | Dict with loss history |
| `training_val_accuracy(model)` | Get validation accuracy history | Model | Dict with val_accuracy history |
| `training_val_loss(model)` | Get validation loss history | Model | Dict with val_loss history |

## QoA Attribute Enums

Quality attributes are defined as enums in `qoa4ml.lang.attributes`:

### Data Quality Attributes

| Attribute | Value | Description |
|---|---|---|
| `ACCURACY` | `accuracy` | Ratio between correct and total data received (%) |
| `COMPLETENESS` | `completeness` | Ratio of non-empty to total data received (%) |
| `TOTAL_ERRORS` | `total_errors` | Total number of erroneous data points |
| `DUPLICATE_RATIO` | `duplicate_ratio` | Ratio of duplicate records |
| `NULL_COUNT` | `null_count` | Number of null/missing values |

### ML Model Quality Attributes

| Attribute | Value | Description |
|---|---|---|
| `PRECISION` | `precision` | Correct positive predictions / total positive predictions |
| `RECALL` | `recall` | Correct positive predictions / total actual positives |
| `F1_SCORE` | `f1_score` | Harmonic mean of precision and recall |
| `ACCURACY` | `accuracy` | Correct predictions / total predictions |
| `MSE` | `mse` | Mean squared error |
| `MAE` | `mae` | Mean absolute error |
| `AUC_ROC` | `auc_roc` | Area under ROC curve |

### Service Quality Attributes

| Attribute | Value | Description |
|---|---|---|
| `RESPONSE_TIME` | `response_time` | Time for a service to respond to a request |
| `THROUGHPUT` | `throughput` | Number of requests handled per time unit |
| `RELIABILITY` | `reliability` | Probability that a service operates without failure |
| `AVAILABILITY` | `availability` | Fraction of time the service is operational |
| `COST` | `cost` | Cost associated with using the service |

## Resource Utilization Metrics

System and process resource metrics collected by probes:

| Metric | Description | Source |
|---|---|---|
| CPU usage (%) | Per-core and aggregate CPU utilization | `SystemMonitoringProbe` |
| Memory usage | RSS, VMS, and percentage | `ProcessMonitoringProbe` |
| GPU usage (%) | Core and memory utilization per device | `SystemMonitoringProbe` (NVIDIA) |
| Docker stats | Container CPU and memory usage | `DockerMonitoringProbe` |
| Network I/O | Bytes sent/received | `system_report()` utility |

## Usage Example

```python
from qoa4ml.qoa_client import QoaClient
from qoa4ml.lang.datamodel_enum import MetricClassEnum

client = QoaClient(config_path="config/client.yaml")

# Observe a service metric
client.observe_metric("response_time", 0.125, MetricClassEnum.service)

# Observe a data quality metric
client.observe_metric("accuracy", 0.97, MetricClassEnum.data)

# Observe an inference metric
client.observe_inference_metric("confidence", 0.92)

# Generate report
report = client.report()
```
