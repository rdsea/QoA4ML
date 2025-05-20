from __future__ import annotations

from pydantic import BaseModel

from qoa4ml.config.configs import ClientInfo


class BaseMetadata(BaseModel):
    client_info: ClientInfo | None = None


class ProcessMetadata(BaseMetadata):
    pid: str
    user: str


class SystemMetadata(BaseMetadata):
    node_name: str
    model: str | None = None


class ResourceReport(BaseModel):
    metadata: dict | None = None
    usage: dict


class ProcessReport(BaseModel):
    metadata: ProcessMetadata
    timestamp: float
    cpu: ResourceReport
    gpu: ResourceReport | None = None
    mem: ResourceReport


class SystemReport(BaseModel):
    metadata: SystemMetadata
    timestamp: float
    cpu: ResourceReport
    gpu: ResourceReport | None = None
    mem: ResourceReport


class DockerContainerMetadata(BaseMetadata):
    id: str
    image: str


class DockerContainerReport(BaseModel):
    metadata: DockerContainerMetadata
    timestamp: float
    cpu: ResourceReport
    gpu: ResourceReport | None = None
    mem: ResourceReport


class DockerReport(BaseModel):
    metadata: ClientInfo
    timestamp: float
    container_reports: list[DockerContainerReport] = []
