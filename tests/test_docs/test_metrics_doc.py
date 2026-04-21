"""Guard rail tests that keep `docs/metrics.md` in sync with the source enums.

If `metrics.md` references an enum member that does not exist in the code,
these tests fail and the doc is updated instead of being silently wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

from qoa4ml.lang.attributes import (
    DataQualityEnum,
    MLModelQualityEnum,
    ServiceQualityEnum,
)

METRICS_DOC = Path(__file__).resolve().parents[2] / "docs" / "metrics.md"


def _extract_enum_rows(doc: str, header: str) -> list[tuple[str, str]]:
    """Return (attribute, value) pairs from the markdown table after `header`."""
    section_match = re.search(
        rf"### {re.escape(header)}.*?\n(.*?)(?:\n### |\n## |\Z)",
        doc,
        flags=re.DOTALL,
    )
    assert section_match, f"section {header!r} not found in metrics.md"
    section = section_match.group(1)

    rows: list[tuple[str, str]] = []
    for line in section.splitlines():
        row_match = re.match(r"\|\s*`([A-Z_0-9]+)`\s*\|\s*`([a-z_0-9]+)`\s*\|", line)
        if row_match:
            rows.append((row_match.group(1), row_match.group(2)))
    assert rows, f"no enum rows parsed from section {header!r}"
    return rows


def test_data_quality_enum_matches_doc():
    rows = _extract_enum_rows(
        METRICS_DOC.read_text(), "Data Quality Attributes (`DataQualityEnum`)"
    )
    for name, value in rows:
        assert hasattr(DataQualityEnum, name), f"DataQualityEnum.{name} missing"
        assert getattr(DataQualityEnum, name).value == value


def test_ml_model_quality_enum_matches_doc():
    rows = _extract_enum_rows(
        METRICS_DOC.read_text(), "ML Model Quality Attributes (`MLModelQualityEnum`)"
    )
    for name, value in rows:
        assert hasattr(MLModelQualityEnum, name), f"MLModelQualityEnum.{name} missing"
        assert getattr(MLModelQualityEnum, name).value == value


def test_service_quality_enum_matches_doc():
    rows = _extract_enum_rows(
        METRICS_DOC.read_text(), "Service Quality Attributes (`ServiceQualityEnum`)"
    )
    for name, value in rows:
        assert hasattr(ServiceQualityEnum, name), f"ServiceQualityEnum.{name} missing"
        assert getattr(ServiceQualityEnum, name).value == value


def test_usage_example_uses_valid_category():
    doc = METRICS_DOC.read_text()
    # observe_metric category must be an int; reject stale MetricClassEnum usage
    assert "MetricClassEnum.service" not in doc
    assert "MetricClassEnum.data" not in doc
    assert "category=0" in doc
    assert "category=1" in doc
