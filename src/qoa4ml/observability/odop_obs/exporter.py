import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from qoa4ml.config.configs import ExporterConfig
from qoa4ml.observability.odop_obs.node_aggregator import NodeAggregator

logging.basicConfig(
    format="%(asctime)s:%(levelname)s -- %(message)s", level=logging.INFO
)


class Exporter:
    def __init__(self, config: ExporterConfig, odop_path: Path) -> None:
        self.app = FastAPI()
        self.config = config
        self.node_aggregator = NodeAggregator(self.config.node_aggregator, odop_path)
        self.app.include_router(self.node_aggregator.router)

    def start(self):
        self.node_aggregator.start()
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)
