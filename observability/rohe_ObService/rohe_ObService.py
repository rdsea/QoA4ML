import argparse
import uuid

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from rohe_Agent import Rohe_Agent

from qoa4ml.utils.logger import qoa_logger
from qoa4ml.utils.qoa_utils import get_parent_dir, load_config

app = Flask(__name__)
api = Api(app)


application_list: dict = {}
agent_list: dict = {}


_REQUIRED_CONFIG_KEYS = ("database", "connector", "collector")


class Rohe_ObService(Resource):  # noqa: N801 - preserved external class name
    """Flask-RESTful resource that registers QoA client applications.

    Configuration dictionary must supply three keys:

    - ``database``: mapping consumed by :class:`Rohe_Agent` to build a
      Mongo connection (keys: ``url``, ``db_name``, ``metric_collection``).
    - ``connector``: mapping from connector-name to ``{"conf": {...}}`` where
      ``conf`` contains AMQP wiring (``exchange_name``, ``out_routing_key``).
    - ``collector``: same shape as ``connector``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in kwargs]
        if missing:
            raise ValueError(
                f"Rohe_ObService requires config keys {_REQUIRED_CONFIG_KEYS}; "
                f"missing: {missing}"
            )
        self.conf = kwargs
        self.db_config = self.conf["database"]
        self.connector_config = self.conf["connector"]
        self.collector_config = self.conf["collector"]

    def get(self):
        args = request.query_string.decode("utf-8").split("&")
        return jsonify({"status": args})

    def post(self):
        if not request.is_json:
            return jsonify(
                {"status": "error", "response": {"Error": "JSON body required"}}
            )

        args = request.get_json(force=True)
        qoa_logger.debug(f"registration request: {args}")
        response: dict = {}
        if "application_name" not in args:
            response["Error"] = "Application name not found"
            return jsonify({"status": "success", "response": response})

        application_name = args["application_name"]
        if application_name not in application_list:
            application_list[application_name] = {}
            application_list[application_name]["id"] = str(uuid.uuid4())
            application_list[application_name]["client_count"] = 0
            response[application_name] = f"Application {application_name} created"
        else:
            response[application_name] = "OK"

        application_list[application_name]["client_count"] += 1

        # Prepare connector for QoA Client
        connector = self.connector_config.copy()
        for key in list(connector.keys()):
            connector_i = connector[key]
            i_config = connector_i["conf"]
            i_config["exchange_name"] = f"{application_name}_exchange"
            i_config["out_routing_key"] = str(application_name)
            for optional_key in ("user_id", "stage_id", "instance_name"):
                if optional_key in args:
                    i_config["out_routing_key"] = (
                        i_config["out_routing_key"] + "." + args[optional_key]
                    )
            i_config["out_routing_key"] = (
                i_config["out_routing_key"]
                + ".client"
                + str(application_list[application_name]["client_count"])
            )
        response["application_id"] = application_list[application_name]["id"]
        response["connector"] = connector

        # Prepare QoA Agent
        application_id = application_list[application_name]["id"]
        if application_id not in agent_list:
            agent_db_config = self.db_config.copy()
            agent_db_config["db_name"] = application_name + "_" + application_id
            agent_db_config["metric_collection"] = "metric_collection"
            collector_config = self.collector_config.copy()
            for key in list(collector_config.keys()):
                collector_i = collector_config[key]
                i_config = collector_i["conf"]
                i_config["exchange_name"] = f"{application_name}_exchange"
                i_config["out_routing_key"] = f"{application_name}.#"

            agent_config = {
                "database": agent_db_config,
                "collector": collector_config,
            }
            agent = Rohe_Agent(agent_config)
            agent_id = str(uuid.uuid4())
            agent_list[application_id] = {
                agent_id: {
                    "agent": agent,
                    "configuration": agent_config,
                    "status": "starting",
                }
            }
            agent.start()
            agent_list[application_id][agent_id]["status"] = "running"

        return jsonify({"status": "success", "response": response})

    def put(self):
        # Intentional no-op stub; body-driven semantics are not yet defined.
        return jsonify({"status": True})

    def delete(self):
        if request.is_json:
            args = request.get_json(force=True)
            return jsonify({"status": args})
        return jsonify({"status": []})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument for Rohe Observation Service"
    )
    parser.add_argument("--conf", help="configuration file", default=None)
    args = parser.parse_args()
    config_file = args.conf
    if not config_file:
        config_file = get_parent_dir(__file__, 2) + "/config/rohe_obs_conf.json"
        qoa_logger.info(f"using default config: {config_file}")
    configuration = load_config(config_file)

    api.add_resource(
        Rohe_ObService, "/registration", resource_class_kwargs=configuration
    )
    app.run(debug=True, port=5001)
