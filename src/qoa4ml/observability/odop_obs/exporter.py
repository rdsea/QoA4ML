import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from qoa4ml.config.configs import ExporterConfig
from qoa4ml.observability.odop_obs.node_aggregator import NodeAggregator

# Use a named logger; the host application controls logging config.
logger = logging.getLogger(__name__)


class Exporter:
    """HTTP exporter that serves aggregated node metrics over FastAPI.

    Composition: a ``NodeAggregator`` ingests reports from its
    ``SocketCollector`` and makes them queryable via an attached router.
    This class wires that router into a FastAPI app and runs it with
    uvicorn.

    Instantiate from user code:

        exporter = Exporter(config, odop_path)
        exporter.start()  # blocks
    """

    def __init__(self, config: ExporterConfig, odop_path: Path) -> None:
        self.app = FastAPI()
        self.config = config
        self.node_aggregator = NodeAggregator(self.config.node_aggregator, odop_path)
        self.app.include_router(self.node_aggregator.router)

    def start(self) -> None:
        self.node_aggregator.start()
        logger.info(f"Exporter starting on {self.config.host}:{self.config.port}")
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)
