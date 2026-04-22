import prometheus_client as pr

_KNOWN_PROM_TYPES = {"Gauge", "Counter", "Summary", "Histogram"}


class PromConnector:
    def __init__(self, info: dict) -> None:
        self.info = info["metric"]
        self.port = info["port"]
        self.metrics: dict = {}
        for key in self.info:
            metric_type = self.info[key]["Type"]
            if metric_type not in _KNOWN_PROM_TYPES:
                raise ValueError(
                    f"PromConnector: unknown metric type {metric_type!r} for key {key!r}; "
                    f"expected one of {sorted(_KNOWN_PROM_TYPES)}"
                )
            self.metrics[key] = {}
            if metric_type == "Gauge":
                self.metrics[key]["metric"] = pr.Gauge(
                    self.info[key]["Prom_name"], self.info[key]["Description"]
                )
            elif metric_type == "Counter":
                self.metrics[key]["metric"] = pr.Counter(
                    self.info[key]["Prom_name"], self.info[key]["Description"]
                )
            elif metric_type == "Summary":
                self.metrics[key]["metric"] = pr.Summary(
                    self.info[key]["Prom_name"], self.info[key]["Description"]
                )
            elif metric_type == "Histogram":
                self.metrics[key]["metric"] = pr.Histogram(
                    self.info[key]["Prom_name"],
                    self.info[key]["Description"],
                    buckets=(tuple(self.info[key]["Buckets"])),
                )
            self.metrics[key]["violation"] = pr.Counter(
                self.info[key]["Prom_name"] + "_violation",
                self.info[key]["Description"] + " (violation)",
            )
        pr.start_http_server(int(info["port"]))

    def inc(self, key: str, num: float = 1) -> None:
        metric_type = self.info[key]["Type"]
        if metric_type in ("Gauge", "Counter"):
            self.metrics[key]["metric"].inc(num)
        else:
            raise ValueError(
                f"PromConnector.inc({key!r}): {metric_type} does not support inc()"
            )

    def dec(self, key: str, num: float = 1) -> None:
        metric_type = self.info[key]["Type"]
        if metric_type == "Gauge":
            self.metrics[key]["metric"].dec(num)
        else:
            raise ValueError(
                f"PromConnector.dec({key!r}): only Gauge supports dec(); got {metric_type}"
            )

    def set(self, key: str, num: float = 1) -> None:
        """Set a Gauge or observe a Histogram/Summary value.

        Raises ``ValueError`` for Counter keys — Counters do not support
        `set`; callers must use :meth:`inc` instead. Previous silent
        translation to ``.inc(num)`` masked misuse.
        """
        metric_type = self.info[key]["Type"]
        if metric_type == "Gauge":
            self.metrics[key]["metric"].set(num)
        elif metric_type in ("Histogram", "Summary"):
            self.metrics[key]["metric"].observe(num)
        elif metric_type == "Counter":
            raise ValueError(
                f"PromConnector.set({key!r}): Counter metrics cannot be set; use inc() instead"
            )
        else:
            raise ValueError(
                f"PromConnector.set({key!r}): unknown type {metric_type!r}"
            )

    def observe(self, key: str, val: float) -> None:
        metric_type = self.info[key]["Type"]
        if metric_type in ("Summary", "Histogram"):
            self.metrics[key]["metric"].observe(val)
        else:
            raise ValueError(
                f"PromConnector.observe({key!r}): {metric_type} does not support observe()"
            )

    def inc_violation(self, key: str, num: float = 1) -> None:
        self.metrics[key]["violation"].inc(num)

    def render_violation_counts(self) -> dict[str, bytes]:
        """Render the current violation counters as Prometheus text payloads.

        Replaces the legacy ``update_violation_count`` whose name implied
        mutation but only called :func:`prometheus_client.generate_latest`
        (a renderer) and threw the result away. Callers can now use the
        returned bytes for exposition endpoints.
        """
        return {
            key: pr.generate_latest(self.metrics[key]["violation"])
            for key in self.metrics
        }
