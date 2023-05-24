# QoA4ML - Quality of Analytics for ML

## Source code
https://github.com/rdsea/QoA4ML

## Probes
* [QoA4ML Probes](https://github.com/rdsea/QoA4ML/tree/main/qoa4ml_lib/qoa4ml/probes.py): libraries and lightweight modules capturing metrics. They are integrated into suitable ML serving frameworks and ML code
* Probe properties:
  - Can be written in different languages (Python, GoLang)
  - Can have different communications to monitoring systems (depending on probes and its ML support)
  - Capture metrics with a clear definition/scope
    - e.g., Response time for an ML stage (training) or a service call (of ML APIs)
    - Thus output of probes must be correlated to objects to be monitored and the tenant
  - Support high or low-level metrics/attributes
    - depending on probes implementation
  - Can be instrumented into source code or standlone

Provide some metric classes for collecting different types of metric: Counter, Gauge, Summary, Histogram

- `Metric`: an original class providing some common functions on an metric object.
    - Attribute:
        - `metric_name`
        - `description`
        - `value`
    - Function:
        - `__init__`: let user define the metric name, description and default value.
        - `set`: set its `value` to a specific value
        - `get_val`: get current value
        - `get_name`: return metric name 
        - `get_des`: return metric description 
        - `__str__`: return information about the metric in form of string
        - `to_dict`: return information about the metric in form of dictionary
- `Counter`
    - Attribute: same as `Metric` & on further developing
    - Function:
        - `inc`: increase the value of the metric by the given number/by 1 by default.
        - `reset`: set the value back to zero.
- `Gauge`
    - Attribute: same as `Metric` & on further developing
    - Function:
        - `inc`: increase the value of the metric by a given number/by 1 by default.
        - `dec`: decrease the value of the metric by a given number/by 1 by default.
        - `set`: set the value to a given number.
- `Summary`
    - Attribute: same as `Metric` & on further developing
    - Function:
        - `inc`: increase the value of the metric by a given number/by 1 by default.
        - `dec`: decrease the value of the metric by a given number/by 1 by default.
        - `set`: set the value to a given number.
- `Histogram`
    - Attribute: same as `Metric` & on further developing
    - Function:
        - `inc`: increase the value of the metric by a given number/by 1 by default.
        - `dec`: decrease the value of the metric by a given number/by 1 by default.
        - `set`: set the value to a given number.

## [QoA4ML Reports](https://github.com/rdsea/QoA4ML/blob/main/qoa4ml_lib/qoa4ml/reports.py)

This module defines ``QoA_Report``, an object that provide functions to export monitored metric to the following schema:
```json
{
    "execution_graph":{
        "instances":{
            "@instance_id":{
                "instance_name": "@name_of_instance",
                "method": "@method/task/function",
                "previous_instance":["@list_of_previous_instance"]
            },
            ...
        },
        "last_instance": "@name_of_last_instance_in_the_graph"
    },
    "quality":{
        "data":{
            "@stage_id":{
                "@metric_name":{
                    "@instance_id": "@value"
                }
            }
        },
        "service":{
            "@stage_id":{
                "@metric_name":{
                    "@instance_id": "@value"
                }
            }
        },
        "inference":{
            "@inference_id":{
                "value": "@value",
                "confident": "@confidence",
                "accuracy": "@accuracy",
                "instance_id": "@instance_id",
                "source": ["@list_of_inferences_to_infer_this_inference"]
            }
        }
    }
}
```

The example is shown in `example/reports/qoa_report/example.txt`

- Attribute:
    - `previous_report_instance` = list previous services
    - `report_list`: list of reports from previous services
    - `previous_inference`: list previous inferences
    - `quality_report`: report all quality (data, service, inference qualtiy) of the service
    - `execution_graph`: report the execution graph
    - `report`: the final report to be sent

- Function:
    - `__init__`: init as empty report.
    - `import_report_from_file`: init QoA Report from `json` file.
    - `import_pReport`: import reports from previous service to build the execution and inference graph
    - `build_execution_graph`: build execution graph from list of previous reports
    - `build_quality_report`: build the quality report from metrics collected in runtime
    - `generate_report`: return the final report.
    - `observe_metric`: observe metrics in runtime with 3 categories: service quality, data quality, inference qualtiy. This can be extended to observe resource metrics.


## [Examples](https://github.com/rdsea/QoA4ML/tree/main/example)
https://github.com/rdsea/QoA4ML/tree/main/example




## Overview
![Class](../img/class.png)

Probes will be integrated to client program or system service to collect metrics at the edge
Probes will generate reports and sent to message broker using different connector. Coresponding collector should be used to acquire the metrics.

## Collector
The manager/orchestrator have to integrate collector to collect metric using different protocols for further analysis.
- Attribute:
- Function:
    - `__init__`: take a configuration as a `dict` containing information about the data source, eg. broker, channel, queue, etc. It can take an `object` as an attribute `host` to return the message for further processing.


    - If the collector is initiated by an object inherited class, this class must implement `message_processing` function to process the message returned by the collector. Otherwise, the collector will print the message to the console.

    - `on_request`: handle message from data source (message broker,...)

    - `start` & `stop`: start and stop consuming message

    - `get_queue`: return the queue name.

## Connector
Connectors are implement with different protocols for sending report. Example: sending report to message broker - AMQP/MQTT
- Attribute:
- Function:
    - `__init__`: take a configuration as a `dict` containing information about the data sink, eg. broker, channel, queue, etc. It can take a `bool` parameter `log` for logging messages for further processing.

    - `send_data`: a function to send data to specified `routing_key`/`queue` with a corresponding key `corr_id` to trace back message.



## Utilities
A module provide some frequently used functions and some function to directly collect system metrics.

