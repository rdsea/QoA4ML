"""Regression tests for Probe.reporting and RepeatedTimer error resilience."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from qoa4ml.config.configs import ProbeConfig
from qoa4ml.connector.base_connector import BaseConnector
from qoa4ml.probes.probe import Probe
from qoa4ml.utils.repeated_timer import RepeatedTimer


class _StubProbe(Probe):
    """Minimal concrete Probe for testing the base class."""

    def __init__(self, connector, *, raise_in_create=False, raise_in_send=False):
        cfg = ProbeConfig(
            probe_type="stub",
            frequency=10,
            require_register=False,
            log_latency_flag=False,
        )
        super().__init__(cfg, connector, client_info=None)
        self.raise_in_create = raise_in_create
        self.raise_in_send = raise_in_send
        self.create_calls = 0

    def create_report(self):
        self.create_calls += 1
        if self.raise_in_create:
            raise RuntimeError("boom-create")
        return "ok"


class TestProbeReporting:
    def test_create_report_exception_is_logged_not_raised(self, caplog):
        # Regression: a failing create_report used to propagate into
        # RepeatedTimer._target and kill the daemon thread silently.
        connector = MagicMock(spec=BaseConnector)
        probe = _StubProbe(connector, raise_in_create=True)

        with caplog.at_level("ERROR"):
            probe.reporting()  # must NOT raise

        assert probe.create_calls == 1
        connector.send_report.assert_not_called()
        assert any("create_report failed" in r.getMessage() for r in caplog.records)

    def test_send_report_exception_is_logged_not_raised(self, caplog):
        connector = MagicMock(spec=BaseConnector)
        connector.send_report.side_effect = OSError("network is down")
        probe = _StubProbe(connector)

        with caplog.at_level("ERROR"):
            probe.reporting()  # must NOT raise

        connector.send_report.assert_called_once_with("ok")
        assert any("send_report failed" in r.getMessage() for r in caplog.records)


@pytest.mark.integration
class TestRepeatedTimerResilience:
    def test_single_tick_failure_does_not_stop_scheduler(self):
        # Regression: previously `_target` had no try/except, so a single
        # exception from the callable killed the thread.
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("transient failure")

        timer = RepeatedTimer(0.05, flaky)
        try:
            # Wait long enough for at least 3 ticks.
            deadline = time.time() + 1.0
            while call_count["n"] < 3 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            timer.stop()

        assert call_count["n"] >= 3, (
            f"scheduler stopped after first exception (ran {call_count['n']} times)"
        )

    def test_stop_joins_cleanly(self):
        timer = RepeatedTimer(0.05, lambda: None)
        timer.stop()
        assert not timer.thread.is_alive()


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    # Probe.start_reporting sleeps to align to wall-clock second boundary;
    # no test exercises start_reporting here, but keep behavior fast if added.
    real_sleep = time.sleep

    def short_sleep(secs):
        real_sleep(min(secs, 0.01))

    monkeypatch.setattr("qoa4ml.probes.probe.time.sleep", short_sleep)
