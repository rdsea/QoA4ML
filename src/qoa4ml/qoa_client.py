import copy
import os
import sys
import threading
import time
import traceback
import uuid
from threading import Thread
from typing import Any, Generic, Optional, TypeVar, Union

import requests
from pydantic import create_model

# from .connector.mqtt_connector import Mqtt_Connector
from qoa4ml.config.configs import (
    AMQPConnectorConfig,
    ClientConfig,
    ClientInfo,
    ConnectorConfig,
    DebugConnectorConfig,
    DockerProbeConfig,
    ProbeConfig,
    ProcessProbeConfig,
    SystemProbeConfig,
)
from qoa4ml.connector.amqp_connector import AmqpConnector
from qoa4ml.connector.base_connector import BaseConnector
from qoa4ml.connector.debug_connector import DebugConnector
from qoa4ml.lang.attributes import ServiceQualityEnum
from qoa4ml.lang.common_models import Metric
from qoa4ml.lang.datamodel_enum import (
    MetricNameEnum,
    ReportTypeEnum,
    ServiceAPIEnum,
)
from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe
from qoa4ml.probes.probe import Probe
from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe
from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe
from qoa4ml.reports.abstract_report import AbstractReport
from qoa4ml.reports.ml_reports import MLReport
from qoa4ml.utils.logger import qoa_logger
from qoa4ml.utils.qoa_utils import (
    load_config,
    set_logger_level,
)

headers = {"Content-Type": "application/json"}


# NOTE: T must be a subtype of AbstractReport
T = TypeVar("T", bound=AbstractReport)


class QoaClient(Generic[T]):
    def __init__(
        self,
        report_cls: type[T] = MLReport,
        config_dict: Optional[dict] = None,
        config_path: Optional[str] = None,
        registration_url: Optional[str] = None,
    ):
        """
        Initialize the QoA Client with configuration settings and a report class.

        Parameters
        ----------
        report_cls : type[T], optional
            The class type for reports, default is MLReport.
        config_dict : dict, optional
            A dictionary to load the client's configuration from.
        config_path : str, optional
            Path to a JSON configuration file.
        registration_url : str, optional
            URL for registering the client and receiving configuration data.

        Notes
        -----
        - If both `config_dict` and `config_path` are provided, the `config_dict` will take precedence.
        - If neither `config_dict` nor `config_path` is provided, the client may attempt to fetch configurations from the `registration_url`.
        - The method will raise an exception if the necessary configuration details are not found.
        """
        if config_dict is not None:
            self.configuration = ClientConfig.model_validate(config_dict)

        if config_path is not None:
            self.configuration = ClientConfig.model_validate(load_config(config_path))

        set_logger_level(self.configuration.client.logging_level)
        self.client_config = self.configuration.client
        self.connector_list: dict[str, BaseConnector] = {}
        self.timer_flag = False
        self.functionality = self.client_config.functionality
        self.stage_id = self.client_config.stage_id
        self.process_monitor_flag = 0
        self.inference_flag = False

        self.instance_id = os.environ.get("INSTANCE_ID")
        if self.instance_id:
            qoa_logger.info("Setting instance_id with INSTANCE_ID")
            self.instance_id = uuid.UUID(self.instance_id)
        elif self.configuration.client.instance_id:
            qoa_logger.info("Setting instance_id with client.instance_id")
            self.instance_id = uuid.UUID(self.configuration.client.instance_id)
        else:
            self.instance_id = str(uuid.uuid4())

        self.client_config.instance_id = str(self.instance_id)
        self.client_config.id = str(uuid.uuid4())
        self.qoa_report = report_cls(self.client_config)
        if self.configuration.connector:
            connector_conf = self.configuration.connector
            try:
                for connector in connector_conf:
                    self.connector_list[connector.name] = self.init_connector(connector)
            except Exception as e:
                qoa_logger.exception(
                    f"Error {type(e)} when configuring connector in QoaClient"
                )
        elif registration_url or self.configuration.registration_url:
            try:
                if registration_url:
                    registration_data = self.registration(registration_url)
                else:
                    registration_data = self.registration(
                        self.configuration.registration_url
                    )
                json_data = registration_data.json()
                response = json_data["response"]
                if isinstance(response, dict):
                    connector_configs = response["connector"]
                    if isinstance(connector_configs, dict):
                        connector_config = ConnectorConfig(**connector_configs)
                        self.connector_list[connector_config.name] = (
                            self.init_connector(connector_config)
                        )
                    elif isinstance(connector_configs, list):
                        for config in connector_configs:
                            connector_config = ConnectorConfig(**config)
                            self.connector_list[connector_config.name] = (
                                self.init_connector(connector_config)
                            )
                else:
                    qoa_logger.warning(
                        "Unable to register Qoa Client: connector configuration must be dictionary"
                    )
            except Exception as e:
                qoa_logger.exception(f"Error {type(e)} when registering QoA client")
                traceback.print_exception(*sys.exc_info())

        if not self.connector_list:
            qoa_logger.warning("No connector initiated")
            self.default_connector = None
        else:
            self.default_connector = next(iter(self.connector_list.keys()))

        self.probes_list = None
        if self.configuration.probes:
            self.probes_list = self.init_probes(
                self.configuration.probes, self.configuration.client
            )
        self.lock = threading.Lock()

    def registration(self, url: str) -> requests.Response:
        """
        Registers the client with the monitoring service and retrieves connector configurations.

        Parameters
        ----------
        url : str
            The registration URL to fetch the connector configuration from.

        Returns
        -------
        requests.Response
            The response from the registration service, containing connector configurations.

        Notes
        -----
        This method sends a POST request to the given URL with the client's configuration in JSON format.
        """
        return requests.request(
            "POST", url, headers=headers, data=self.client_config.json()
        )

    def init_probes(
        self, probe_config_list: list[ProbeConfig], client_info: ClientInfo
    ) -> list[Probe]:
        """
        Initialize monitoring probes based on the provided probe configuration list.

        Parameters
        ----------
        probe_config_list : list[ProbeConfig]
            A list of configuration settings for each probe.
        client_info : ClientInfo
            Information about the client to be passed to each probe.

        Returns
        -------
        list[Probe]
            A list of initialized probe instances.

        Raises
        ------
        ValueError
            If an unsupported probe configuration type is provided.
        """
        probes_list: list[Probe] = []
        if self.default_connector:
            selected_connector = self.connector_list[self.default_connector]
        else:
            qoa_logger.warning("No default connector, using debug connector")
            selected_connector = DebugConnector(DebugConnectorConfig(silence=False))

        for probe_config in probe_config_list:
            if isinstance(probe_config, DockerProbeConfig):
                probes_list.append(
                    DockerMonitoringProbe(probe_config, selected_connector, client_info)
                )
            elif isinstance(probe_config, ProcessProbeConfig):
                probes_list.append(
                    ProcessMonitoringProbe(
                        probe_config, selected_connector, client_info
                    )
                )
            elif isinstance(probe_config, SystemProbeConfig):
                probes_list.append(
                    SystemMonitoringProbe(probe_config, selected_connector, client_info)
                )
            else:
                raise ValueError(
                    f"Probe config type {type(probe_config)} is not supported yet"
                )
        return probes_list

    def init_connector(self, configuration: ConnectorConfig) -> BaseConnector:
        """
        Initialize a connector based on the configuration provided.

        Parameters
        ----------
        configuration : ConnectorConfig
            Configuration settings for initializing the connector.

        Returns
        -------
        BaseConnector
            An instance of the connector (e.g., AMQP, Debug).

        Raises
        ------
        RuntimeError
            If the connector configuration type is not supported.
        """
        if configuration.connector_class == ServiceAPIEnum.amqp and isinstance(
            configuration.config, AMQPConnectorConfig
        ):
            return AmqpConnector(configuration.config)
        elif configuration.connector_class == ServiceAPIEnum.debug and isinstance(
            configuration.config, DebugConnectorConfig
        ):
            return DebugConnector(configuration.config)

        raise RuntimeError("Connector config is not of correct type")

    def get_client_config(self) -> ClientConfig:
        """
        Get the current client configuration.

        Returns
        -------
        ClientConfig
            The client's current configuration settings.
        """
        return self.client_config

    def set_config(self, key: str, value: Any) -> None:
        """
        Update a specific configuration setting by key.

        Parameters
        ----------
        key : str
            The configuration attribute name to be updated.
        value : Any
            The value to set for the specified key.

        Raises
        ------
        Exception
            Logs an error if setting the configuration value fails.
        """
        try:
            self.client_config.__setattr__(key, value)
        except Exception as e:
            qoa_logger.exception(f"Error {type(e)} when setConfig in QoA client")

    def observe_metric(
        self,
        metric_name: MetricNameEnum,
        value: Any,
        category: int = 0,
        description: str = "",
    ) -> None:
        """
        Observe and report a metric.

        Parameters
        ----------
        metric_name : MetricNameEnum
            The name of the metric being observed.
        value : Any
            The value of the observed metric.
        category : int, optional
            The category of the metric (0: service, 1: data, 2: security), default is 0.
        description : str, optional
            An optional description of the observed metric, default is "".

        Raises
        ------
        RuntimeError
            If the category type is not supported.
        """
        if category == 0:
            report_type = ReportTypeEnum.service
        elif category == 1:
            report_type = ReportTypeEnum.data
        elif category == 2:
            report_type = ReportTypeEnum.security
        else:
            raise RuntimeError("Report type not supported")

        self.qoa_report.observe_metric(
            report_type,
            self.stage_id,
            Metric(metric_name=metric_name, records=[value], description=description),
        )

    def timer(self) -> dict:
        """
        Start or stop a timer and record the response time.

        Returns
        -------
        dict
            A dictionary containing the start time and response time.

        Notes
        -----
        - When called for the first time, it starts the timer.
        - When called again, it stops the timer and records the response time as a metric.
        """
        if self.timer_flag is False:
            self.timer_flag = True
            self.timerStart = time.time()
            return {}
        else:
            self.timer_flag = False
            response_time = {
                "startTime": self.timerStart,
                "responseTime": time.time() - self.timerStart,
            }
            self.observe_metric(
                ServiceQualityEnum.RESPONSE_TIME, response_time, category=0
            )
            return response_time

    def import_previous_report(self, reports: Union[dict, list[dict]]) -> None:
        """
        Import and process previous reports.

        Parameters
        ----------
        reports : Union[dict, list[dict]]
            A single report or a list of reports to be processed.
        """
        if isinstance(reports, list):
            for report in reports:
                self.qoa_report.process_previous_report(report)
        else:
            self.qoa_report.process_previous_report(reports)

    def asyn_report(self, body_mess: str, connectors: Optional[list] = None) -> None:
        """
        Asynchronously send a report through the connectors.

        Parameters
        ----------
        body_mess : str
            The message body to be sent.
        connectors : list, optional
            A list of connectors to send the report through. If None, the default connector is used.

        Notes
        -----
        Uses threading to send reports asynchronously.
        """
        self.lock.acquire()
        if connectors is None:
            if self.default_connector:
                chosen_connector = self.connector_list[self.default_connector]
                if isinstance(chosen_connector, AmqpConnector):
                    if not chosen_connector.check_connection():
                        chosen_connector.reconnect()

                    chosen_connector.send_report(body_mess, corr_id=str(uuid.uuid4()))
                else:
                    chosen_connector.send_report(body_mess)
            else:
                qoa_logger.error(
                    "No default connector, please specify the connector to use"
                )
        else:
            pass
        self.lock.release()

    def report(
        self,
        report: Optional[dict] = None,
        connectors: Optional[list] = None,
        submit: bool = False,
        reset: bool = True,
        corr_id: Optional[str] = None,
    ) -> str:
        """
        Generate a report and optionally submit it through the default connector.

        Parameters
        ----------
        report : dict, optional
            The report data to be submitted. If None, a report will be generated.
        connectors : list, optional
            A list of connectors through which to send the report, default is None.
        submit : bool, optional
            Whether to submit the report, default is False.
        reset : bool, optional
            Whether to reset the report state after submission, default is True.
        corr_id : str, optional
            The correlation ID for the report, default is None.

        Returns
        -------
        str
            The JSON-encoded report.

        Notes
        -----
        The method will create a report based on the current state if none is provided.
        If `submit` is True, the report will be sent through the default or specified connectors.
        """
        if report is None:
            return_report = self.qoa_report.generate_report(reset, corr_id=corr_id)
        else:
            user_defined_report_model = create_model(
                "UserDefinedReportModel",
                metadata=(dict, ...),
                timestamp=(float, ...),
                report=(dict, ...),
            )
            return_report = user_defined_report_model(
                report=report,
                metadata=copy.deepcopy(self.client_config.__dict__),
                timestamp=time.time(),
            )

        if submit:
            if self.default_connector is not None:
                sub_thread = Thread(
                    target=self.asyn_report,
                    args=(return_report.model_dump_json(), connectors),
                )
                sub_thread.start()
            else:
                qoa_logger.warning("No connector available")
        return return_report.model_dump(mode="json")

    def start_all_probes(self) -> None:
        """
        Start all probes for monitoring, running them in the background.

        Raises
        ------
        RuntimeError
            If no probes have been initialized.

        Notes
        -----
        If the probe takes a long time to report and the main process exits, no report may be sent.
        """
        if not self.probes_list:
            raise RuntimeError(
                "There is no initiated probes, please recheck the config"
            )
        for probe in self.probes_list:
            probe.start_reporting()

    def stop_all_probes(self) -> None:
        """
        Stop all running probes.

        Raises
        ------
        RuntimeError
            If no probes have been initialized.

        Notes
        -----
        This method stops the background monitoring activities of all active probes.
        """
        if not self.probes_list:
            raise RuntimeError(
                "There are no initiated probes, please recheck the config"
            )
        for probe in self.probes_list:
            probe.stop_reporting()

    def observe_inference(self, inference_value: Any) -> None:
        """
        Observe and record inference data.

        Parameters
        ----------
        inference_value : Any
            The value of the inference to be observed.

        Notes
        -----
        This method is used to record predictions or inference results for later analysis.
        """
        self.qoa_report.observe_inference(inference_value)

    def observe_inference_metric(
        self,
        metric_name: MetricNameEnum,
        value: Any,
    ) -> None:
        """
        Observe and report a specific inference metric.

        Parameters
        ----------
        metric_name : MetricNameEnum
            The name of the inference metric being observed.
        value : Any
            The value of the observed metric.

        Notes
        -----
        This method can be used to log performance metrics, evaluation scores, etc. during inference.
        """
        metric = Metric(metric_name=metric_name, records=[value])
        self.qoa_report.observe_inference_metric(metric)

    def __str__(self) -> str:
        """
        Returns a string representation of the client's configuration and connectors.

        Returns
        -------
        str
            JSON representation of the client configuration and a string representation of the connector list.

        Notes
        -----
        This method is particularly useful for debugging and logging purposes.
        """
        return self.client_config.model_dump_json() + "\n" + str(self.connector_list)
