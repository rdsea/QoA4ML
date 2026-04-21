import asyncio
import time

import docker
from docker.models.containers import Container

from qoa4ml.reports.resources_report_model import (
    DockerContainerMetadata,
    DockerContainerReport,
    ResourceReport,
)

BYTES_TO_MB = 1024.0 * 1024.0


def _compute_cpu_percentage(stat: dict) -> float:
    """Compute container CPU %, guarding against idle-sample divide-by-zero."""
    cpu_stats = stat.get("cpu_stats", {})
    precpu_stats = stat.get("precpu_stats", {})
    usage_delta = cpu_stats.get("cpu_usage", {}).get(
        "total_usage", 0
    ) - precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
    system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get(
        "system_cpu_usage", 0
    )
    len_cpu = cpu_stats.get("online_cpus") or 0
    if system_delta <= 0 or len_cpu <= 0:
        # Two consecutive samples with no elapsed system time, or cpu_stats
        # missing (common for containers that just started). Report 0%
        # rather than crashing the probe.
        return 0.0
    return (usage_delta / system_delta) * len_cpu * 100


def _pick_image_tag(image) -> str:
    """Return a stable image identifier even when the image has no tags."""
    if image is None:
        return ""
    tags = getattr(image, "tags", None) or []
    if tags:
        return tags[0]
    return getattr(image, "id", "") or ""


async def get_container_stats(
    container: Container,
) -> DockerContainerReport:
    timestamp = time.time()
    stat = await asyncio.to_thread(container.stats, stream=False)

    cpu_percentage = _compute_cpu_percentage(stat)
    memory_bytes = stat.get("memory_stats", {}).get("usage", 0) or 0
    memory_mb = memory_bytes / BYTES_TO_MB

    container_id = container.id
    if not container_id:
        raise RuntimeError("container id is None")

    return DockerContainerReport(
        metadata=DockerContainerMetadata(
            id=container_id, image=_pick_image_tag(container.image)
        ),
        timestamp=timestamp,
        cpu=ResourceReport(usage={"cpu_percentage": cpu_percentage}),
        mem=ResourceReport(usage={"memory_usage": memory_mb}),
    )


async def get_all_container_stats(client):
    tasks = []
    for container in client.containers.list():
        if container.status == "running":
            tasks.append(get_container_stats(container))
    results = await asyncio.gather(*tasks)
    return results


async def get_container_list_stats(
    client: docker.DockerClient, container_list: list[str]
):
    tasks = []
    for container_name in container_list:
        container = client.containers.get(container_name)
        if container.status == "running":
            tasks.append(get_container_stats(container))
    results = await asyncio.gather(*tasks)
    return results


def get_docker_stats(
    client: docker.DockerClient, container_list: list[str]
) -> list[DockerContainerReport]:
    if container_list:
        return asyncio.run(get_container_list_stats(client, container_list))

    return asyncio.run(get_all_container_stats(client))
