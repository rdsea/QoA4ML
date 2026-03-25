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


async def get_container_stats(
    container: Container,
) -> DockerContainerReport:
    container_stats: dict[str, dict] = {}
    container_stats["cpu"] = {}
    container_stats["mem"] = {}

    timestamp = time.time()
    stat = await asyncio.to_thread(container.stats, stream=False)

    usage_delta = (
        stat["cpu_stats"]["cpu_usage"]["total_usage"]
        - stat["precpu_stats"]["cpu_usage"]["total_usage"]
    )
    system_delta = (
        stat["cpu_stats"]["system_cpu_usage"] - stat["precpu_stats"]["system_cpu_usage"]
    )
    len_cpu = stat["cpu_stats"]["online_cpus"]
    cpu_percentage = (usage_delta / system_delta) * len_cpu * 100
    container_stats["cpu"] = {"cpu_percentage": cpu_percentage}

    # Bytes to Mb
    container_stats["mem"] = {
        "memory_usage": stat["memory_stats"]["usage"] / BYTES_TO_MB
    }

    container_id = container.id
    container_image = container.image
    if not container_id or not container_image:
        raise RuntimeError("container id or container image is None")
    return DockerContainerReport(
        metadata=DockerContainerMetadata(
            id=container_id, image=container_image.tags[0]
        ),
        timestamp=timestamp,
        cpu=ResourceReport(usage={"cpu_percentage": cpu_percentage}),
        mem=ResourceReport(
            usage={"memory_usage": stat["memory_stats"]["usage"] / BYTES_TO_MB}
        ),
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
