# QoA4ML - Quality of Analytics for ML

## Source code

[QoA4ML on GitHub](https://github.com/rdsea/QoA4ML/tree/main/src/qoa4ml)

## Monitoring client

`QoaClient` is the entry point for applications that want to report metrics to the observation service. It observes metrics, assembles reports, and sends them through one or more connectors (e.g., the AMQP connector for RabbitMQ or the debug connector for local development).

A client is initialised by:

1. Pointing at a YAML/JSON config file with `config_path=...`.
2. Passing a config dict with `config_dict=...`.
3. Pointing at a registration service with `registration_url=...`; the client then fetches its connector configuration over HTTP.

### Example configuration

The top-level schema is `ClientConfig` in `src/qoa4ml/config/configs.py`. A minimal working configuration looks like:

```yaml
client:
  username: aaltosea1
  instance_name: data_handling01
  stage_id: gateway
  functionality: REST
  application_name: test
  role: ml
connector:
  - name: debug_connector
    connector_class: Debug
    config:
      silence: false
```

```python
import yaml
from qoa4ml.qoa_client import QoaClient

with open("config/client.yaml") as f:
    client_conf = yaml.safe_load(f)

client = QoaClient(config_dict=client_conf)
```

Key details:

- `connector` is a **list** of connector entries, not a dict. Each entry has `name`, `connector_class`, and `config` (matched to the connector class). `QoaClient.init_connector` currently wires only `AMQP` and `Debug`; the other classes (`MqttConnector`, `KafkaConnector`, `SocketConnector`, `PromConnector`) can be instantiated directly but are not selected through `connector_class` yet.
- `client.functionality` is a free-form string (suggested values from `FunctionalityEnum`: `REST`, `TensorFlow`, `Transformation`, `Max Aggregate`).
- `client.role` is a free-form string (suggested values from `StakeholderRoleEnum`: `ml_consumer`, `ml_provider`, `ml_infrastructure`).
- If `connector` is omitted, `registration_url` must be provided at either config level or as a constructor argument.

More examples live in `example/simple/config/` and `example/reports/config/`.

### Observing metrics

Metrics are observed via `client.observe_metric(name, value, category, description="")` where `category` is an integer:

| Category | Report type | Typical metrics |
|---|---|---|
| `0` | Service | response time, reliability, availability |
| `1` | Data | accuracy, completeness, duplicates, missing values |
| `2` | Security | security-related attributes |

Any other integer raises `RuntimeError("Report type not supported")`.

```python
from qoa4ml.lang.attributes import (
    DataQualityEnum,
    MLModelQualityEnum,
    ServiceQualityEnum,
)

client.observe_metric(ServiceQualityEnum.RESPONSE_TIME, 0.125, category=0)
client.observe_metric(DataQualityEnum.ACCURACY, 0.97, category=1)
client.observe_inference_metric(MLModelQualityEnum.ACCURACY, 0.92)
```

Other `QoaClient` methods:

- `observe_inference(value)` — record an inference prediction.
- `timer()` — start/stop a stopwatch; the second call records `RESPONSE_TIME` automatically.
- `import_previous_report(report)` — ingest a report from an upstream pipeline stage to rebuild execution and inference graphs.
- `report(submit=True)` — generate the current report and optionally dispatch it via the default or provided connectors.

## Probes

Probes are lightweight modules that capture metrics and push them through a connector. The base class is `qoa4ml.probes.probe.Probe`. Concrete implementations:

- `ProcessMonitoringProbe` — per-process CPU and memory usage (via `psutil`).
- `SystemMonitoringProbe` — host-level CPU / memory / GPU usage.
- `DockerMonitoringProbe` — container stats via the Docker SDK.
- `mlquality` (function module, not a probe class) — TensorFlow/Keras metric extractors; requires `pip install qoa4ml[ml]`.

Each concrete probe overrides `create_report()` and inherits `start_reporting(background=True)` / `stop_reporting()`. A `RepeatedTimer` drives the report cadence based on the configured `frequency` (reports per second).

## Metric model

`qoa4ml.lang.common_models.Metric` is a Pydantic model:

```python
class Metric(BaseModel):
    metric_name: MetricNameEnum
    records: list[dict | float | int | tuple | str] = []
    unit: str | None = None
    description: str | None = None
```

There is **no** `Counter`/`Gauge`/`Summary`/`Histogram` class in QoA4ML. `MetricClassEnum` exists as a StrEnum (`gauge`, `counter`, `summary`, `histogram`) for Prometheus-compatible configuration, but the project's own reporting path uses the single `Metric` model above. Treat `MetricClassEnum` as a tag, not as an inheritance hierarchy.

For constraint / contract definitions, see `qoa4ml.lang.common_models.Condition`, `MetricConstraint`, and `BaseConstraint`.

## Reports

Reports are produced by an `AbstractReport` subclass held inside `QoaClient.qoa_report`. The default is `MLReport` (`src/qoa4ml/reports/ml_reports.py`). Other implementations:

- `MLReport` — ML-focused report with service, data, and ml_inference sections.
- `GeneralApplicationReport` — flat list of metrics without ML-specific structure.
- `RoheReport` — ROHE-specific report carrying an execution graph and inference graph.

### `MLReport` JSON shape

`client.report()` returns a dict shaped like:

```json
{
  "metadata": {
    "client_config": { "...ClientInfo fields..." },
    "timestamp": 1713700000.123,
    "runtime": 42.5
  },
  "service": {
    "gateway": {
      "name": "gateway",
      "metrics": {
        "response_time": {
          "<instance_uuid>": { "metric_name": "response_time", "records": [0.125], "unit": null, "description": "" }
        }
      }
    }
  },
  "data": {
    "gateway": {
      "name": "gateway",
      "metrics": { "accuracy": { "<instance_uuid>": { "metric_name": "accuracy", "records": [0.97], "unit": null, "description": "" } } }
    }
  },
  "ml_inference": {
    "<instance_uuid>": {
      "inference_id": "<uuid>",
      "instance_id": "<uuid>",
      "functionality": "REST",
      "metrics": [ { "metric_name": "accuracy", "records": [0.92], "unit": null, "description": null } ],
      "prediction": null
    }
  }
}
```

Key methods on `MLReport` (defined in `AbstractReport`, implemented in concrete subclasses):

- `reset()` — clear the in-progress report.
- `observe_metric(report_type, stage, metric)` — called internally by `QoaClient.observe_metric`.
- `observe_inference(inference_value)` / `observe_inference_metric(metric)` — record inference data.
- `process_previous_report(previous_report_dict)` — merge an upstream report into the current one.
- `generate_report(reset=True, corr_id=None)` — snapshot the current report with metadata.

## Examples

Working examples are in [`example/`](https://github.com/rdsea/QoA4ML/tree/main/example):

- `example/simple/` — multi-stage AMQP pipeline with five client YAMLs.
- `example/reports/` — end-to-end report and contract demos using the debug connector (no broker required).
- `example/dataquality/` — data-quality utilities applied to arrays and images.
- `example/docker_report/` — Docker container monitoring probe demo.

See also `docs/metrics.md` for the full metric-function reference.

## Collectors

Collectors run server-side to receive reports from probes. The base class is `qoa4ml.collector.base_collector.BaseCollector`. Implementations:

- `AmqpCollector` — consumes from a RabbitMQ queue; exposes `on_request(ch, method, props, body)`, `start_collecting()`, `stop()`, and `get_queue() -> str`.
- `SocketCollector` — TCP server; exposes `start_collecting()` and `stop()`. Each connection is decoded as UTF-8 and passed to the `process_report` callback injected at construction.

`AmqpCollector` can accept an optional `HostObject` to receive decoded messages via `message_processing(ch, method, props, body)`.

## Connectors

Connectors push reports out to an observation service. The contract is `qoa4ml.connector.base_connector.BaseConnector`, which declares the abstract method `send_report(body_message: str)`. Implementations:

- `AmqpConnector` — publishes to a RabbitMQ exchange; supports reconnect / heartbeat.
- `MqttConnector` — publishes to an MQTT topic; requires `paho-mqtt` (`qoa4ml[ml]`).
- `KafkaConnector` — publishes to a Kafka topic; requires `confluent-kafka` (`qoa4ml[kafka]`).
- `SocketConnector` — opens a TCP connection and sends UTF-8 bytes.
- `PromConnector` — exposes metrics for a Prometheus scrape target.
- `DebugConnector` — logs the serialized report (for development only).

All connectors implement `send_report`. `MqttConnector` additionally exposes `send_data`, and its `send_report` delegates to `send_data` internally (legacy naming kept to avoid breaking existing instrumentations).

## Utilities

`qoa4ml.utils` provides supporting helpers:

- `qoa4ml.utils.qoa_utils` — config loading, logger level, cgroup detection, flatten/unflatten dict helpers, process and system sampling utilities.
- `qoa4ml.utils.dataquality_utils` — `eva_erronous`, `eva_duplicate`, `eva_missing`, `eva_none`, `image_quality` (see `docs/metrics.md`).
- `qoa4ml.utils.docker_util` — async container-stats helpers used by `DockerMonitoringProbe`.
- `qoa4ml.utils.gpu_utils` / `jetson_utils` — GPU and Jetson-specific resource probes.

## Requirements and optional extras

- Python 3.12 or newer (see `pyproject.toml`).
- Core runtime deps (`pydantic`, `fastapi`, `docker`, `tinyflux`, etc.) install automatically with `pip install qoa4ml`.
- Optional extras:
  - `qoa4ml[ml]` — `tensorflow`, `pandas`, `Pillow`, `paho-mqtt`, `prometheus-client`.
  - `qoa4ml[otel]` — OpenTelemetry instrumentation.
  - `qoa4ml[docs]` — mkdocs toolchain for building the docs site.

Attempting to use an optional integration without the extra raises a clean `ImportError` with the install hint (e.g., `MqttConnector`, `mlquality`). To monitor Docker, the Docker daemon must be reachable; the Python `docker` package is already a core dependency.
