import pytest
from pydantic import ValidationError

from qoa4ml.config.configs import (
    AMQPCollectorConfig,
    AMQPConnectorConfig,
    ClientConfig,
    ClientInfo,
    CollectorConfig,
    ConnectorConfig,
    DebugConnectorConfig,
    DockerProbeConfig,
    ExporterConfig,
    GroupMetricConfig,
    KafkaCollectorConfig,
    KafkaConnectorConfig,
    MetricConfig,
    MQTTConnectorConfig,
    NodeAggregatorConfig,
    ProbeConfig,
    ProcessProbeConfig,
    PrometheusConnectorConfig,
    SocketCollectorConfig,
    SocketConnectorConfig,
    SystemProbeConfig,
)
from qoa4ml.lang.datamodel_enum import EnvironmentEnum, MetricClassEnum, ServiceAPIEnum


# ---------------------------------------------------------------------------
# ClientInfo
# ---------------------------------------------------------------------------
class TestClientInfo:
    def test_defaults(self):
        info = ClientInfo()
        assert info.id == ""
        assert info.name == ""
        assert info.user_id == ""
        assert info.username == ""
        assert info.instance_id == ""
        assert info.instance_name == ""
        assert info.stage_id == ""
        assert info.functionality == ""
        assert info.application_name == ""
        assert info.role == ""
        assert info.run_id == ""
        assert info.environment == EnvironmentEnum.edge
        assert info.custom_info == ""
        assert info.logging_level == 2

    def test_with_values(self):
        info = ClientInfo(
            name="test_client",
            username="alice",
            environment=EnvironmentEnum.cloud,
            logging_level=4,
            custom_info={"key": "value"},
        )
        assert info.name == "test_client"
        assert info.username == "alice"
        assert info.environment == EnvironmentEnum.cloud
        assert info.logging_level == 4
        assert info.custom_info == {"key": "value"}

    def test_custom_info_accepts_string(self):
        info = ClientInfo(custom_info="some_info")
        assert info.custom_info == "some_info"

    def test_custom_info_accepts_dict(self):
        info = ClientInfo(custom_info={"a": 1})
        assert info.custom_info == {"a": 1}


# ---------------------------------------------------------------------------
# Connector / Collector configs
# ---------------------------------------------------------------------------
class TestAMQPConnectorConfig:
    def test_valid(self):
        cfg = AMQPConnectorConfig(
            end_point="amqp://localhost",
            exchange_name="ex",
            exchange_type="topic",
            out_routing_key="key",
        )
        assert cfg.end_point == "amqp://localhost"
        assert cfg.health_check_disable is False

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AMQPConnectorConfig(
                end_point="amqp://localhost",
                exchange_name="ex",
                exchange_type="topic",
            )


class TestAMQPCollectorConfig:
    def test_valid(self):
        cfg = AMQPCollectorConfig(
            end_point="amqp://localhost",
            exchange_name="ex",
            exchange_type="topic",
            in_routing_key="key",
            in_queue="queue",
        )
        assert cfg.in_queue == "queue"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AMQPCollectorConfig(end_point="amqp://localhost")


class TestMQTTConnectorConfig:
    def test_valid(self):
        cfg = MQTTConnectorConfig(
            in_queue="in",
            out_queue="out",
            broker_url="mqtt://localhost",
            broker_port=1883,
            broker_keepalive=60,
            client_id="c1",
        )
        assert cfg.broker_port == 1883

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            MQTTConnectorConfig(in_queue="in")


class TestSocketConfigs:
    def test_connector(self):
        cfg = SocketConnectorConfig(host="localhost", port=5000)
        assert cfg.host == "localhost"
        assert cfg.port == 5000

    def test_collector(self):
        cfg = SocketCollectorConfig(host="0.0.0.0", port=5000, backlog=5, bufsize=1024)
        assert cfg.bufsize == 1024

    def test_connector_missing_port(self):
        with pytest.raises(ValidationError):
            SocketConnectorConfig(host="localhost")

    def test_collector_missing_fields(self):
        with pytest.raises(ValidationError):
            SocketCollectorConfig(host="0.0.0.0")


class TestKafkaConfigs:
    def test_connector(self):
        cfg = KafkaConnectorConfig(topic="t", broker_url="localhost:9092")
        assert cfg.topic == "t"

    def test_collector_defaults(self):
        cfg = KafkaCollectorConfig(
            topic="t", broker_url="localhost:9092", group_id="g1"
        )
        assert cfg.auto_offset_reset == "earliest"
        assert cfg.poll_interval == 1.0

    def test_collector_custom_values(self):
        cfg = KafkaCollectorConfig(
            topic="t",
            broker_url="localhost:9092",
            group_id="g1",
            auto_offset_reset="latest",
            poll_interval=0.5,
        )
        assert cfg.auto_offset_reset == "latest"
        assert cfg.poll_interval == 0.5


class TestDebugConnectorConfig:
    def test_valid(self):
        cfg = DebugConnectorConfig(silence=True)
        assert cfg.silence is True

    def test_missing_silence(self):
        with pytest.raises(ValidationError):
            DebugConnectorConfig()


class TestPrometheusConnectorConfig:
    def test_empty_model(self):
        cfg = PrometheusConnectorConfig()
        assert cfg is not None


# ---------------------------------------------------------------------------
# CollectorConfig / ConnectorConfig wrappers
# ---------------------------------------------------------------------------
class TestCollectorConfig:
    def test_with_amqp(self):
        cfg = CollectorConfig(
            name="amqp_col",
            collector_class=ServiceAPIEnum.amqp,
            config={
                "end_point": "amqp://localhost",
                "exchange_name": "ex",
                "exchange_type": "topic",
                "in_routing_key": "key",
                "in_queue": "queue",
            },
        )
        assert cfg.collector_class == ServiceAPIEnum.amqp


class TestConnectorConfig:
    def test_with_debug(self):
        cfg = ConnectorConfig(
            name="debug_conn",
            connector_class=ServiceAPIEnum.debug,
            config={"silence": True},
        )
        assert cfg.connector_class == ServiceAPIEnum.debug

    def test_config_optional(self):
        cfg = ConnectorConfig(
            name="prom_conn",
            connector_class=ServiceAPIEnum.rest,
            config=None,
        )
        assert cfg.config is None

    def test_config_default_none(self):
        cfg = ConnectorConfig(
            name="prom_conn",
            connector_class=ServiceAPIEnum.rest,
        )
        assert cfg.config is None


# ---------------------------------------------------------------------------
# MetricConfig / GroupMetricConfig
# ---------------------------------------------------------------------------
class TestMetricConfig:
    def test_valid(self):
        cfg = MetricConfig(
            name="accuracy",
            metric_class=MetricClassEnum.gauge,
            default_value=0,
            category=1,
        )
        assert cfg.name == "accuracy"
        assert cfg.description is None

    def test_with_description(self):
        cfg = MetricConfig(
            name="response_time",
            metric_class=MetricClassEnum.histogram,
            default_value=100,
            category=2,
            description="Measures response time",
        )
        assert cfg.description == "Measures response time"

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            MetricConfig(name="accuracy")


class TestGroupMetricConfig:
    def test_valid(self):
        mc = MetricConfig(
            name="accuracy",
            metric_class=MetricClassEnum.gauge,
            default_value=0,
            category=1,
        )
        gmc = GroupMetricConfig(name="group1", metric_configs=[mc])
        assert len(gmc.metric_configs) == 1


# ---------------------------------------------------------------------------
# ProbeConfig and subclasses
# ---------------------------------------------------------------------------
class TestProbeConfig:
    def test_base_probe(self):
        cfg = ProbeConfig(
            probe_type="custom",
            frequency=5,
            require_register=True,
            log_latency_flag=False,
        )
        assert cfg.probe_type == "custom"
        assert cfg.environment == EnvironmentEnum.edge
        assert cfg.obs_service_url is None
        assert cfg.latency_logging_path is None

    def test_process_probe_defaults(self):
        cfg = ProcessProbeConfig(
            frequency=1,
            require_register=False,
            log_latency_flag=False,
        )
        assert cfg.probe_type == "process"
        assert cfg.pid is None

    def test_system_probe(self):
        cfg = SystemProbeConfig(
            frequency=1,
            require_register=False,
            log_latency_flag=False,
            node_name="node1",
        )
        assert cfg.probe_type == "system"
        assert cfg.node_name == "node1"

    def test_docker_probe(self):
        cfg = DockerProbeConfig(
            frequency=1,
            require_register=False,
            log_latency_flag=False,
            container_list=["c1", "c2"],
        )
        assert cfg.probe_type == "docker"
        assert cfg.container_list == ["c1", "c2"]

    def test_docker_probe_default_empty_list(self):
        cfg = DockerProbeConfig(
            frequency=1,
            require_register=False,
            log_latency_flag=False,
        )
        assert cfg.container_list == []


# ---------------------------------------------------------------------------
# ClientConfig with model_validator
# ---------------------------------------------------------------------------
class TestClientConfig:
    def test_minimal(self):
        cfg = ClientConfig(client=ClientInfo())
        assert cfg.client.name == ""
        assert cfg.connector is None
        assert cfg.collector is None
        assert cfg.probes is None
        assert cfg.registration_url is None

    def test_from_dict(self, sample_client_config_dict):
        cfg = ClientConfig(**sample_client_config_dict)
        assert cfg.client.name == "qoa_client_test"
        assert len(cfg.connector) == 1
        assert cfg.connector[0].connector_class == ServiceAPIEnum.debug

    def test_model_validator_with_probes(self, sample_client_config_with_probes_dict):
        cfg = ClientConfig(**sample_client_config_with_probes_dict)
        assert cfg.probes is not None
        assert len(cfg.probes) == 3
        assert isinstance(cfg.probes[0], DockerProbeConfig)
        assert isinstance(cfg.probes[1], SystemProbeConfig)
        assert isinstance(cfg.probes[2], ProcessProbeConfig)

    def test_model_validator_docker_probe_fields(
        self, sample_client_config_with_probes_dict
    ):
        cfg = ClientConfig(**sample_client_config_with_probes_dict)
        docker_probe = cfg.probes[0]
        assert isinstance(docker_probe, DockerProbeConfig)
        assert docker_probe.container_list == ["test"]
        assert docker_probe.environment == EnvironmentEnum.edge

    def test_model_validator_system_probe_fields(
        self, sample_client_config_with_probes_dict
    ):
        cfg = ClientConfig(**sample_client_config_with_probes_dict)
        system_probe = cfg.probes[1]
        assert isinstance(system_probe, SystemProbeConfig)
        assert system_probe.node_name == "Edge1"

    def test_model_validator_process_probe_fields(
        self, sample_client_config_with_probes_dict
    ):
        cfg = ClientConfig(**sample_client_config_with_probes_dict)
        process_probe = cfg.probes[2]
        assert isinstance(process_probe, ProcessProbeConfig)
        assert process_probe.pid is None

    def test_no_probes_passes_validator(self, sample_client_config_dict):
        cfg = ClientConfig(**sample_client_config_dict)
        assert cfg.probes is None

    def test_empty_probes_list(self, sample_client_config_dict):
        data = {**sample_client_config_dict, "probes": []}
        cfg = ClientConfig(**data)
        assert cfg.probes == []


# ---------------------------------------------------------------------------
# NodeAggregatorConfig / ExporterConfig / OdopObsConfig
# ---------------------------------------------------------------------------
class TestNodeAggregatorConfig:
    def test_valid(self):
        cfg = NodeAggregatorConfig(
            socket_collector_config=SocketCollectorConfig(
                host="0.0.0.0", port=5000, backlog=5, bufsize=1024
            ),
            environment=EnvironmentEnum.edge,
            query_method="pull",
            data_separator=",",
            unit_conversion={"bytes": "megabytes"},
        )
        assert cfg.query_method == "pull"


class TestExporterConfig:
    def test_valid(self):
        na = NodeAggregatorConfig(
            socket_collector_config=SocketCollectorConfig(
                host="0.0.0.0", port=5000, backlog=5, bufsize=1024
            ),
            environment=EnvironmentEnum.edge,
            query_method="pull",
            data_separator=",",
            unit_conversion={},
        )
        cfg = ExporterConfig(host="0.0.0.0", port=9090, node_aggregator=na)
        assert cfg.port == 9090
