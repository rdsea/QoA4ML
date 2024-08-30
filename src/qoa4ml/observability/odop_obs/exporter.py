import logging

import uvicorn
from fastapi import FastAPI

from ...config.configs import ExporterConfig
from .node_aggregator import NodeAggregator

logging.basicConfig(
    format="%(asctime)s:%(levelname)s -- %(message)s", level=logging.INFO
)


class Exporter:
    def __init__(self, config: ExporterConfig) -> None:
        self.app = FastAPI()
        self.config = config
        self.node_aggregator = NodeAggregator(self.config.node_aggregator)
        self.app.include_router(self.node_aggregator.router)

    def start(self):
        self.node_aggregator.start()
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)
