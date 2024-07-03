from abc import ABC, abstractmethod
from typing import Dict

from ..lang.common_models import Metric
from ..reports.ml_report_model import BaseReport


class AbstractReport(ABC):
    @abstractmethod
    def __init__(self, client_config):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def generate_report(self, reset: bool = True) -> BaseReport:
        pass

    @abstractmethod
    def process_previous_report(self, previous_report_dict: Dict):
        pass

    @abstractmethod
    def observe_metric(self, report_type, stage, metric: Metric):
        pass

    @abstractmethod
    def observe_inference(self, inference_value):
        pass

    @abstractmethod
    def observe_inference_metric(self, metric: Metric):
        pass
