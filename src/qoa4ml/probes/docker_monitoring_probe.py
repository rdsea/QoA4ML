import json
import time

import docker

from qoa4ml.config.configs import ClientInfo, DockerProbeConfig
from qoa4ml.connector.base_connector import BaseConnector
from qoa4ml.probes.probe import Probe
from qoa4ml.reports.resources_report_model import DockerReport
from qoa4ml.utils.docker_util import get_docker_stats
from qoa4ml.utils.logger import qoa_logger


class DockerMonitoringProbe(Probe):
    """
    DockerMonitoringProbe is responsible for monitoring Docker containers and creating reports.

    Parameters
    ----------
    config : DockerProbeConfig
        Configuration settings for the Docker monitoring probe.
    connector : BaseConnector
        Connector to send the report data.
    client_info : ClientInfo
        Information about the client.

    Attributes
    ----------
    config : DockerProbeConfig
        The Docker monitoring probe configuration.
    obs_service_url : str
        The URL of the observation service, if registration is required.
    docker_client : docker.DockerClient
        The Docker client for communicating with Docker API.

    Methods
    -------
    create_report() -> str
        Create a report based on Docker container statistics.
    """

    def __init__(
        self,
        config: DockerProbeConfig,
        connector: BaseConnector,
        client_info: ClientInfo,
    ) -> None:
        """
        Initialize an instance of DockerMonitoringProbe.

        Parameters
        ----------
        config : DockerProbeConfig
            Configuration settings for the Docker monitoring probe.
        connector : BaseConnector
            Connector to send the report data.
        client_info : ClientInfo
            Information about the client.
        """
        super().__init__(config, connector, client_info)
        self.config = config
        if self.config.require_register:
            self.obs_service_url = self.config.obs_service_url
        self.docker_client = docker.from_env()

    def create_report(self) -> str:
        """
        Create a report based on Docker container statistics.

        Returns
        -------
        str
            JSON-encoded report containing Docker container statistics.

        Notes
        -----
        - This method collects statistics for the specified Docker containers.
        - If the report dictionary is empty, it adds a 2-second delay to prevent fast looping.
        - In case of a RuntimeError, an error message is returned in a JSON format.
        """
        try:
            reports = get_docker_stats(self.docker_client, self.config.container_list)
            docker_report = DockerReport(
                metadata=self.client_info,
                timestamp=time.time(),
                container_reports=reports,
            )
            reports_dict = docker_report.model_dump()
            if not reports_dict:
                time.sleep(2)
            return json.dumps(reports_dict)
        except RuntimeError:
            qoa_logger.exception(
                "RuntimeError occurred, possibly due to running in the background!"
            )
        return json.dumps({"error": "RuntimeError"})
