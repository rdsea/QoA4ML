from importlib.metadata import version

from qoa4ml.config.configs import ClientConfig
from qoa4ml.qoa_client import QoaClient

__version__ = version("qoa4ml")

__all__ = [
    "ClientConfig",
    "QoaClient",
    "__version__",
]
