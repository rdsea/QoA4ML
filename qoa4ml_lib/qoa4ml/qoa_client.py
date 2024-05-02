import copy
import json
import os
import sys
import threading
import time
import traceback
import uuid
from threading import Thread
from typing import Dict, List, Optional, Union

import requests
from devtools import debug

from qoa4ml.connector.base_connector import BaseConnector
from qoa4ml.datamodels.common_models import Metric
from qoa4ml.datamodels.datamodel_enum import (
    MetricClassEnum,
    MetricNameEnum,
    ReportTypeEnum,
    ServiceAPIEnum,
    ServiceMetricNameEnum,
)
from qoa4ml.datamodels.ml_report import (
    InferenceInstance,
    LinkedInstance,
    RoheReportModel,
)
from qoa4ml.metric_mananger import MetricManager

from .connector.amqp_connector import Amqp_Connector

# from .connector.mqtt_connector import Mqtt_Connector
from .datamodels.configs import (
    AMQPConnectorConfig,
    ClientConfig,
    ConnectorConfig,
    MetricConfig,
)
from .probes.dataquality import (
    eva_duplicate,
    eva_erronous,
    eva_missing,
    eva_none,
    image_quality,
)
from .probes.mlquality import *
from .qoa_utils import (
    get_proc_cpu,
    get_proc_mem,
    load_config,
    qoaLogger,
    set_logger_level,
)
from .reports.rohe_reports import RoheReport

headers = {"Content-Type": "application/json"}


class QoaClient(object):
    # Init QoA Client
    def __init__(
        self,
        config_dict: Optional[ClientConfig] = None,
        config_path: Optional[str] = None,
        registration_url: Optional[str] = None,
        logging_level=2,
    ):
        set_logger_level(logging_level)

        if config_dict != None:
            self.configuration = config_dict

        if config_path != None:
            self.configuration = ClientConfig(**load_config(config_path))

        self.client_config = self.configuration.client

        self.metric_manager = MetricManager()
        self.connector_list: Dict[str, BaseConnector] = {}
        self.timer_flag = False
        self.method = self.client_config.functionality
        self.stage = self.client_config.stage
        self.process_monitor_flag = 0
        self.inference_flag = False

        self.instance_id = os.environ.get("INSTANCE_ID")
        if not self.instance_id:
            qoaLogger.info("INSTANCE_ID is not defined, generating random uuid")
            self.instance_id = uuid.uuid4()
        else:
            self.instance_id = uuid.UUID(self.instance_id)

        self.client_config.id = self.instance_id
        self.qoa_report = RoheReport(self.client_config)
        if self.configuration.connector:
            # init connectors offline if it's specified
            connector_conf = self.configuration.connector
            try:
                for connector in connector_conf:
                    self.connector_list[connector.name] = self.init_connector(connector)
                debug(self.connector_list)
            except Exception as e:
                qoaLogger.error(
                    str(
                        "[ERROR] - Error {} when configuring connector in QoaClient: {}".format(
                            type(e), e.__traceback__
                        )
                    )
                )
        elif registration_url or self.configuration.registration_url:
            # init connectors using configuration received from monitoring service, if it's specified
            try:
                if registration_url:
                    registration_data = self.registration(registration_url)
                else:
                    # NOTE: logically true
                    registration_data = self.registration(
                        self.configuration.registration_url
                    )
                json_data = registration_data.json()
                response = json_data["response"]
                if isinstance(response, dict):
                    connector_configs = response["connector"]
                    for connector in connector_configs:
                        connector_config = ConnectorConfig(**connector)
                        self.connector_list[
                            connector_config.name
                        ] = self.init_connector(connector_config)
                else:
                    qoaLogger.warning(
                        "Unable to register Qoa Client: connector configuration must be dictionary"
                    )
            except Exception as e:
                qoaLogger.error(
                    str(
                        "[ERROR] - Error {} when registering QoA client: {}".format(
                            type(e), e.__traceback__
                        )
                    )
                )
                traceback.print_exception(*sys.exc_info())

        if not self.connector_list:
            qoaLogger.warning("No connector initiated")
            self.default_connector = None
        else:
            # Set default connector for sending monitoring data if not specify
            self.default_connector = list(self.connector_list.keys())[0]

        # lock report to guarantee consistency
        self.lock = threading.Lock()

    def registration(self, url: str):
        # get connector configuration by registering with the monitoring service
        return requests.request(
            "POST", url, headers=headers, data=json.dumps(self.client_config)
        )

    def init_connector(self, configuration: ConnectorConfig) -> BaseConnector:
        # init connector from configuration
        if (
            configuration.connector_class == ServiceAPIEnum.amqp
            and type(configuration.config) is AMQPConnectorConfig
        ):
            return Amqp_Connector(configuration.config)

        # TODO: MQTT is both connector and collector
        #
        # if (
        #    configuration.connector_class == ServiceAPIEnum.mqtt
        #    and type(configuration.config) is MQTTConnectorConfig
        # ):
        #    return Mqtt_Connector(configuration.config)
        raise RuntimeError("Connector config is not of correct type")

    def add_metric(self, metric_configs: List[MetricConfig]):
        # Add multiple metrics
        self.metric_manager.add_metric(metric_configs)

    def init_metric(self, configuration: MetricConfig):
        # init individual metrics
        self.metric_manager.init_metric(configuration)

    def get_client_config(self):
        # TODO: what to do exactly?
        return self.client_config

    def get_metric(self, key: Optional[Union[List, str]] = None):
        # TO DO:
        return self.metric_manager.get_metric(key)

    def reset_metric(self, key: Optional[Union[List, str]] = None):
        # TO DO:
        self.metric_manager.reset_metric(key)

    def set_config(self, key, value):
        # TO DO:
        try:
            self.client_config.__setattr__(key, value)
        except Exception as e:
            qoaLogger.error(
                str(
                    "[ERROR] - Error {} when setConfig in QoA client: {}".format(
                        type(e), e.__traceback__
                    )
                )
            )

    def observe_metric(
        self,
        metric_name: MetricNameEnum,
        value,
        category: int = 0,
        metric_class: MetricClassEnum = MetricClassEnum.gauge,
        description: str = "",
        default_value: int = -1,
    ):
        self.metric_manager.observe_metric(
            metric_name, value, category, metric_class, description, default_value
        )
        metric = self.metric_manager.metric_list[metric_name]
        if category == 0:
            self.qoa_report.observe_metric(
                ReportTypeEnum.service,
                self.stage,
                Metric(metric_name=metric.name, records=[metric.value]),
            )
        elif category == 1:
            self.qoa_report.observe_metric(
                ReportTypeEnum.data,
                self.stage,
                Metric(metric_name=metric.name, records=[metric.value]),
            )

    def timer(self):
        if self.timer_flag == False:
            self.timer_flag = True
            self.timerStart = time.time()
            return {}
        else:
            self.timer_flag = False
            responseTime = {
                "startTime": self.timerStart,
                "responseTime": time.time() - self.timerStart,
            }
            self.observe_metric(
                ServiceMetricNameEnum.response_time, responseTime, category=0
            )
            return responseTime

    def import_previous_report(self, reports: RoheReportModel):
        self.qoa_report.process_previous_report(reports)

    def __str__(self):
        return self.client_config.model_dump_json() + "\n" + str(self.connector_list)

    def process_report(self, interval: int, pid: Optional[int] = None):
        report = {}
        while self.process_monitor_flag == 1:
            try:
                report["proc_cpu_stats"] = get_proc_cpu()
            except Exception as e:
                qoaLogger.error(
                    "Error {} in process cpu stat: {}".format(type(e), e.__traceback__)
                )
                traceback.print_exception(*sys.exc_info())
            try:
                report["proc_mem_stats"] = get_proc_mem()
            except Exception as e:
                qoaLogger.error(
                    "Error {} in process memory stat: {}".format(
                        type(e), e.__traceback__
                    )
                )
                traceback.print_exception(*sys.exc_info())
            try:
                self.report(report=report, submit=True)
            except Exception as e:
                qoaLogger.error(
                    "Error {} in send process report: {}".format(
                        type(e), e.__traceback__
                    )
                )
                traceback.print_exception(*sys.exc_info())
            time.sleep(interval)

    def process_monitor_start(self, interval: int, pid: Optional[int] = None):
        if self.process_monitor_flag == 0:
            if pid == None:
                pid = os.getpid()
            self.process_monitor_flag = 1
            sub_thread = Thread(target=self.process_report, args=(interval, pid))
            sub_thread.start()
        self.process_monitor_flag = 1

    def process_monitor_stop(self):
        self.process_monitor_flag = 2

    def asyn_report(self, report: dict, connectors: Optional[list] = None):
        body_mess = json.dumps(report)
        self.lock.acquire()
        if connectors == None:
            # if connectors are not specify, use default
            if self.default_connector:
                self.connector_list[self.default_connector].send_data(
                    body_mess, corr_id=str(uuid.uuid4())
                )
            else:
                qoaLogger.error(
                    f"No default connector, please specify the connector to use"
                )
        else:
            # iterate connector to send report
            for connector in connectors:
                print(connector)
        self.lock.release()

    def report(
        self,
        metrics: Optional[list] = None,
        report: Optional[RoheReportModel] = None,
        connectors: Optional[list] = None,
        submit=False,
        reset=True,
    ):
        if report == None:
            report = self.qoa_report.generate_report(metrics, reset)
        else:
            report.metadata = copy.deepcopy(self.client_config.__dict__)
            report.metadata["timestamp"] = time.time()

        if submit:
            if self.default_connector != None:
                sub_thread = Thread(target=self.asyn_report, args=(report, connectors))
                sub_thread.start()
            else:
                qoaLogger.warning("No connector available")
        return report

    def observe_inference(
        self,
        value,
        metric: Optional[Metric] = None,
        metric_list: Optional[List[Metric]] = None,
        inference_id: Optional[uuid.UUID] = None,
        dependency: List[InferenceInstance] = [],
    ):
        if not inference_id:
            inference_id = uuid.uuid4()

        inference_instance = InferenceInstance(
            id=inference_id,
            execution_instance_id=self.instance_id,
            metrics=([metric] if metric else metric_list if metric_list else []),
            prediction=value,
        )
        report = LinkedInstance[InferenceInstance](
            previous=dependency, instance=inference_instance
        )
        self.qoa_report.observe_inference(report)
        return inference_id

    def observe_inference_metric(
        self,
        metric: Optional[Metric] = None,
        metric_list: Optional[List[Metric]] = None,
    ):
        self.qoa_report.observe_inference_metric(metric, metric_list)

    def observe_erronous(self, data, errors=None):
        results = eva_erronous(data, errors=errors)
        if results != None:
            for key in results:
                self.observe_metric(key, results[key], 1)

    def observe_duplicate(self, data):
        results = eva_duplicate(data)
        if results != None:
            for key in results:
                self.observe_metric(key, results[key], 1)

    def observe_missing(
        self, data, null_count=True, correlations=False, predict=False, random_state=0
    ):
        results = eva_missing(
            data,
            null_count=null_count,
            correlations=correlations,
            predict=predict,
            random_state=random_state,
        )
        if results != None:
            for key in results:
                self.observe_metric(key, results[key], 1)

    def observe_image_quality(self, image):
        results = image_quality(image)
        if results != None:
            for key in results:
                self.observe_metric(key, results[key], 1)

    def observe_none(self, data):
        results = eva_none(data)
        if results != None:
            for key in results:
                self.observe_metric(key, results[key], 1)