import prometheus_client as pr


class PromConnector:
    def __init__(self, info):
        self.info = info["metric"]
        self.port = info["port"]
        self.metrics = {}
        for key in self.info:
            self.metrics[key] = {}
            if self.info[key]["Type"] == "Gauge":
                self.metrics[key]["metric"] = pr.Gauge(
                    self.info[key]["Prom_name"], self.info[key]["Description"]
                )
            if self.info[key]["Type"] == "Counter":
                self.metrics[key]["metric"] = pr.Counter(
                    self.info[key]["Prom_name"], self.info[key]["Description"]
                )
            if self.info[key]["Type"] == "Summary":
                self.metrics[key]["metric"] = pr.Summary(
                    self.info[key]["Prom_name"], self.info[key]["Description"]
                )
            if self.info[key]["Type"] == "Histogram":
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

    def inc(self, key, num=1):
        if self.info[key]["Type"] in ["Gauge", "Counter"]:
            self.metrics[key]["metric"].inc(num)

    def dec(self, key, num=1):
        if self.info[key]["Type"] == "Gauge":
            self.metrics[key]["metric"].dec(num)

    def set(self, key, num=1):
        if self.info[key]["Type"] == "Gauge":
            self.metrics[key]["metric"].set(num)
        elif self.info[key]["Type"] == "Counter":
            self.metrics[key]["metric"].inc(num)
        elif self.info[key]["Type"] in ["Histogram", "Summary"]:
            self.metrics[key]["metric"].observe(num)

    def observe(self, key, val):
        if self.info[key]["Type"] in ["Summary", "Histogram"]:
            self.metrics[key]["metric"].observe(val)

    def inc_violation(self, key, num=1):
        self.metrics[key]["violation"].inc(num)

    def update_violation_count(self):
        for key in self.metrics:
            pr.generate_latest(self.metrics[key]["violation"])
