import argparse
import copy
import hmac
import os
import re
import threading
import uuid

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from rohe_Agent import Rohe_Agent

from qoa4ml.utils.logger import qoa_logger
from qoa4ml.utils.qoa_utils import get_parent_dir, load_config

app = Flask(__name__)
api = Api(app)


# Module-level state mutated from request handlers. Flask's dev server is
# threaded, so all reads/writes go through ``_state_lock``.
application_list: dict = {}
agent_list: dict = {}
_state_lock = threading.Lock()

# Cap how many distinct applications the service will hold so an
# unauthenticated flood cannot grow ``application_list`` without bound.
# Note: there is no per-application agent cap because the registration
# handler enforces a strict 1:1 application_id → agent invariant: once
# an agent exists for an application_id it is reused, never replaced.
_MAX_APPLICATIONS = 1000

# Identifier fields that flow into AMQP routing keys, exchange names, and
# Mongo database names. Restrict to a safe character set so callers cannot
# inject AMQP topic wildcards (``#``/``*``), Mongo path metacharacters
# (``$.\0``), or path separators.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_OPTIONAL_IDENTIFIER_FIELDS = ("user_id", "stage_id", "instance_name")
_REQUIRED_CONFIG_KEYS = ("database", "connector", "collector")


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _expected_token() -> str | None:
    """Return the API token configured via ``ROHE_OBS_TOKEN`` or ``None``."""
    token = os.environ.get("ROHE_OBS_TOKEN", "").strip()
    return token or None


def _request_token_authorized() -> bool:
    """Compare the bearer token (if configured) using a constant-time check.

    No-op pass-through when ``ROHE_OBS_TOKEN`` is unset, preserving the
    historical "open registration" behaviour for local dev. When set, the
    header ``Authorization: Bearer <token>`` is required and compared with
    :func:`hmac.compare_digest` to avoid timing leaks.
    """
    expected = _expected_token()
    if expected is None:
        return True
    header = request.headers.get("Authorization", "")
    scheme, _, presented = header.partition(" ")
    # RFC 7235 §2.1: auth scheme is case-insensitive.
    if scheme.lower() != "bearer":
        return False
    presented = presented.strip()
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def _validate_identifier(value: object, field: str) -> str:
    """Return ``value`` if it looks safe to interpolate into routing keys."""
    if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
        raise ValueError(
            f"Invalid value for {field!r}: must match {_IDENTIFIER_RE.pattern}"
        )
    return value


def _error(response_message: str, status_code: int) -> tuple:
    # flask-restful serialises the return value itself, so hand back a plain
    # dict (not a jsonified Response) or the tuple nests a Response that
    # json.dumps then chokes on.
    return (
        {"status": "error", "response": {"Error": response_message}},
        status_code,
    )


class Rohe_ObService(Resource):  # noqa: N801 - preserved external class name
    """Flask-RESTful resource that registers QoA client applications.

    Configuration dictionary must supply three keys:

    - ``database``: mapping consumed by :class:`Rohe_Agent` to build a
      Mongo connection (keys: ``url``, ``db_name``, ``metric_collection``).
    - ``connector``: mapping from connector-name to ``{"conf": {...}}`` where
      ``conf`` contains AMQP wiring (``exchange_name``, ``out_routing_key``).
    - ``collector``: same shape as ``connector``.

    Optional environment variables:

    - ``ROHE_OBS_TOKEN`` — when set, a bearer token is required on every
      mutating request (POST/DELETE).
    - ``ROHE_OBS_DEBUG`` — when truthy, enable Flask's debugger (unsafe!).
    - ``ROHE_OBS_HOST`` — bind host (default ``127.0.0.1``).
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
        # Echoing arbitrary query strings is a footgun for any future caller
        # that renders the response. Return a fixed status payload instead.
        return jsonify({"status": "ok"})

    def post(self):
        if not _request_token_authorized():
            return _error("Unauthorized", 401)
        if not request.is_json:
            return _error("JSON body required", 400)

        args = request.get_json(silent=True)
        if not isinstance(args, dict):
            return _error("JSON body required", 400)
        qoa_logger.debug(f"registration request: {args}")
        if "application_name" not in args:
            return _error("Application name not found", 400)

        try:
            application_name = _validate_identifier(
                args["application_name"], "application_name"
            )
            for field in _OPTIONAL_IDENTIFIER_FIELDS:
                if field in args:
                    _validate_identifier(args[field], field)
        except ValueError as error:
            return _error(str(error), 400)

        with _state_lock:
            if application_name not in application_list:
                if len(application_list) >= _MAX_APPLICATIONS:
                    return _error(
                        "Application registry full; refusing new registration",
                        503,
                    )
                application_list[application_name] = {
                    "id": str(uuid.uuid4()),
                    "client_count": 0,
                }
                created = True
            else:
                created = False
            application_list[application_name]["client_count"] += 1
            application_id = application_list[application_name]["id"]
            client_count = application_list[application_name]["client_count"]

        response: dict = {
            application_name: (
                f"Application {application_name} created" if created else "OK"
            )
        }

        # Prepare connector for QoA Client. Deep-copy so we never mutate
        # the module-shared template that other concurrent requests read.
        connector = copy.deepcopy(self.connector_config)
        for key in list(connector.keys()):
            i_config = connector[key]["conf"]
            i_config["exchange_name"] = f"{application_name}_exchange"
            i_config["out_routing_key"] = application_name
            for optional_key in _OPTIONAL_IDENTIFIER_FIELDS:
                if optional_key in args:
                    i_config["out_routing_key"] = (
                        i_config["out_routing_key"] + "." + args[optional_key]
                    )
            i_config["out_routing_key"] = (
                i_config["out_routing_key"] + ".client" + str(client_count)
            )
        response["application_id"] = application_id
        response["connector"] = connector

        # Prepare QoA Agent. The registration enforces a strict 1:1
        # application_id → agent invariant: if an agent already exists for
        # this application we reuse it, never create a second one. That
        # invariant is what naturally bounds ``agent_list`` size (it is
        # already bounded by ``_MAX_APPLICATIONS`` on the application side).
        # The Mongo db_name uses the server-side UUID so user input cannot
        # pick a name that collides with another tenant.
        with _state_lock:
            agents_for_app = agent_list.setdefault(application_id, {})
            if agents_for_app:
                spawned_agent = None
                spawned_agent_id = None
            else:
                agent_db_config = copy.deepcopy(self.db_config)
                agent_db_config["db_name"] = "rohe_app_" + application_id
                agent_db_config["metric_collection"] = "metric_collection"
                collector_config = copy.deepcopy(self.collector_config)
                for key in list(collector_config.keys()):
                    i_config = collector_config[key]["conf"]
                    i_config["exchange_name"] = f"{application_name}_exchange"
                    i_config["out_routing_key"] = f"{application_name}.#"

                agent_config = {
                    "database": agent_db_config,
                    "collector": collector_config,
                }
                agent = Rohe_Agent(agent_config)
                agent_id = str(uuid.uuid4())
                agents_for_app[agent_id] = {
                    "agent": agent,
                    "configuration": agent_config,
                    "status": "starting",
                }
                spawned_agent = agent
                spawned_agent_id = agent_id

        if spawned_agent is not None:
            try:
                spawned_agent.start()
            except Exception as error:
                # Don't leave a permanent "starting" entry — that would brick
                # the application_id forever. Rollback is best-effort and
                # idempotent under contention: re-read from ``agent_list``
                # each access so we never operate on a stale snapshot.
                with _state_lock:
                    bucket = agent_list.get(application_id, {})
                    bucket.pop(spawned_agent_id, None)
                    if not bucket:
                        agent_list.pop(application_id, None)
                qoa_logger.exception(
                    f"Rohe_Agent.start() failed ({type(error).__name__})"
                )
                # Keep the specific error in the server log; don't forward
                # AMQP/Mongo connection strings or driver diagnostics to the
                # HTTP caller.
                return _error("Agent start failed", 500)
            with _state_lock:
                # Re-read through ``agent_list`` so we pick up any concurrent
                # rollback rather than blindly writing through a stale alias.
                bucket = agent_list.get(application_id, {})
                if spawned_agent_id in bucket:
                    bucket[spawned_agent_id]["status"] = "running"

        return jsonify({"status": "success", "response": response})

    def put(self):
        # Intentional no-op stub; body-driven semantics are not yet defined.
        return jsonify({"status": True})

    def delete(self):
        if not _request_token_authorized():
            return _error("Unauthorized", 401)
        # Delete semantics are not yet defined; return a fixed payload
        # rather than reflecting unsanitized user JSON back to the caller.
        return jsonify({"status": "ok"})


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
    debug_enabled = _truthy_env("ROHE_OBS_DEBUG")
    bind_host = os.environ.get("ROHE_OBS_HOST", "127.0.0.1")
    if debug_enabled:
        qoa_logger.warning(
            "ROHE_OBS_DEBUG is enabled; the Werkzeug debugger is unsafe to expose"
        )
    if _expected_token() is None:
        qoa_logger.warning(
            "ROHE_OBS_TOKEN is not set; the registration endpoint is open to "
            "any client that can reach %s",
            bind_host,
        )
    app.run(debug=debug_enabled, host=bind_host, port=5001)
