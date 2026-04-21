"""Regression tests for qoa4ml.utils.docker_util safety."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from qoa4ml.utils.docker_util import (
    _compute_cpu_percentage,
    _pick_image_tag,
    get_container_stats,
)


class TestComputeCpuPercentage:
    def test_zero_system_delta_returns_zero(self):
        # Regression: `(usage_delta / system_delta)` used to raise
        # ZeroDivisionError when two consecutive samples had identical
        # system_cpu_usage (happens frequently on idle containers).
        stat = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 1000},
                "system_cpu_usage": 50000,
                "online_cpus": 4,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 500},
                "system_cpu_usage": 50000,
            },
        }
        assert _compute_cpu_percentage(stat) == 0.0

    def test_zero_online_cpus_returns_zero(self):
        stat = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 2000},
                "system_cpu_usage": 60000,
                "online_cpus": 0,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1000},
                "system_cpu_usage": 50000,
            },
        }
        assert _compute_cpu_percentage(stat) == 0.0

    def test_missing_keys_return_zero(self):
        # Containers that just started often lack precpu_stats.
        assert _compute_cpu_percentage({}) == 0.0

    def test_normal_case(self):
        stat = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 2000},
                "system_cpu_usage": 60000,
                "online_cpus": 2,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1000},
                "system_cpu_usage": 50000,
            },
        }
        # (1000/10000) * 2 * 100 = 20.0
        assert _compute_cpu_percentage(stat) == pytest.approx(20.0)


class TestPickImageTag:
    def test_picks_first_tag_when_available(self):
        image = MagicMock()
        image.tags = ["myapp:latest", "myapp:v1"]
        image.id = "sha256:abc"
        assert _pick_image_tag(image) == "myapp:latest"

    def test_falls_back_to_id_when_tags_empty(self):
        # Regression: images referenced by digest have no tags, and
        # `tags[0]` raised IndexError, killing the probe.
        image = MagicMock()
        image.tags = []
        image.id = "sha256:deadbeef"
        assert _pick_image_tag(image) == "sha256:deadbeef"

    def test_returns_empty_string_when_no_tags_and_no_id(self):
        image = MagicMock()
        image.tags = []
        image.id = None
        assert _pick_image_tag(image) == ""

    def test_returns_empty_string_when_image_none(self):
        assert _pick_image_tag(None) == ""


class TestGetContainerStats:
    def test_survives_untagged_image(self):
        # Regression: previously `container.image.tags[0]` crashed.
        container = MagicMock()
        container.id = "abc123"
        container.image = MagicMock(tags=[], id="sha256:feed")
        container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 1000},
                "system_cpu_usage": 10000,
                "online_cpus": 1,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 500},
                "system_cpu_usage": 5000,
            },
            "memory_stats": {"usage": 1024 * 1024},
        }

        report = asyncio.run(get_container_stats(container))
        assert report.metadata.id == "abc123"
        assert report.metadata.image == "sha256:feed"
        assert report.mem.usage == {"memory_usage": 1.0}

    def test_survives_idle_cpu_samples(self):
        container = MagicMock()
        container.id = "abc123"
        container.image = MagicMock(tags=["web:latest"], id="sha256:1")
        container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 1000},
                "system_cpu_usage": 5000,
                "online_cpus": 1,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1000},
                "system_cpu_usage": 5000,
            },
            "memory_stats": {"usage": 0},
        }

        report = asyncio.run(get_container_stats(container))
        assert report.cpu.usage == {"cpu_percentage": 0.0}
        assert report.mem.usage == {"memory_usage": 0.0}
