"""Regression tests for qoa4ml.probes.mlquality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestImportGuard:
    def test_missing_tf_raises_clear_importerror(self, monkeypatch):
        # Regression: with `lazy-import` removed, missing TF used to surface
        # as `AttributeError: 'NoneType' has no attribute 'keras'` deep
        # inside a try/except → returned a confusing `{"Error": ...}`.
        # Now it raises a clean ImportError.
        from qoa4ml.probes import mlquality

        monkeypatch.setattr(mlquality, "tf", None)
        with pytest.raises(ImportError, match=r"qoa4ml\[ml\]"):
            mlquality.timeseries_metric(object())

    def test_missing_tf_all_entry_points(self, monkeypatch):
        from qoa4ml.probes import mlquality

        monkeypatch.setattr(mlquality, "tf", None)
        for func in (
            mlquality.timeseries_metric,
            mlquality.training_metric,
            mlquality.training_loss,
            mlquality.training_val_accuracy,
            mlquality.training_val_loss,
        ):
            with pytest.raises(ImportError):
                func(object())

    def test_classification_confidence_requires_numpy(self, monkeypatch):
        from qoa4ml.probes import mlquality

        monkeypatch.setattr(mlquality, "np", None)
        with pytest.raises(ImportError, match="numpy"):
            mlquality.classification_confidence([0.1, 0.9])


class TestReturnShapes:
    def _fake_tf_with_sequential(self, monkeypatch, metric_specs=None, history=None):
        """Patch tf with a minimal fake that supports isinstance + history."""
        from qoa4ml.probes import mlquality

        class _Sequential:
            pass

        class _Keras:
            Sequential = _Sequential

        class _TF:
            keras = _Keras

        monkeypatch.setattr(mlquality, "tf", _TF)

        model = _Sequential()
        if metric_specs:
            metrics = []
            for name, value in metric_specs:
                m = MagicMock()
                m.name = name
                m.result.return_value = MagicMock(numpy=lambda v=value: v)
                metrics.append(m)
            model.metrics = metrics
        else:
            model.metrics = []

        if history is not None:
            hist = MagicMock()
            hist.history = history
            model.history = hist
        return mlquality, model

    def test_ts_inference_mae_returns_normalized_key(self, monkeypatch):
        mlquality, model = self._fake_tf_with_sequential(
            monkeypatch, metric_specs=[("mean_absolute_error", 0.25)]
        )
        # Regression: doc advertises `{"mae": value}`; old impl returned
        # `{"MAE": {"mean_absolute_error": value}}` (double-nested).
        assert mlquality.ts_inference_mae(model) == {"mae": 0.25}

    def test_ts_inference_loss_returns_loss_key(self, monkeypatch):
        mlquality, model = self._fake_tf_with_sequential(
            monkeypatch, metric_specs=[("loss", 0.5)]
        )
        # Regression: old impl returned `{"Loss": {"loss": 0.5}}`.
        assert mlquality.ts_inference_loss(model) == {"loss": 0.5}

    def test_training_loss_uses_lowercase_key(self, monkeypatch):
        mlquality, model = self._fake_tf_with_sequential(
            monkeypatch, history={"loss": [0.9, 0.7]}
        )
        # Regression: key was `"Training Loss"`; docs/metrics.md says `loss`.
        assert mlquality.training_loss(model) == {"loss": [0.9, 0.7]}

    def test_training_val_accuracy_key(self, monkeypatch):
        mlquality, model = self._fake_tf_with_sequential(
            monkeypatch, history={"val_accuracy": [0.8, 0.85]}
        )
        assert mlquality.training_val_accuracy(model) == {"val_accuracy": [0.8, 0.85]}

    def test_training_val_loss_key(self, monkeypatch):
        mlquality, model = self._fake_tf_with_sequential(
            monkeypatch, history={"val_loss": [0.6, 0.5]}
        )
        assert mlquality.training_val_loss(model) == {"val_loss": [0.6, 0.5]}

    def test_classification_confidence_key(self, monkeypatch):
        pytest.importorskip("numpy")
        from qoa4ml.probes import mlquality

        result = mlquality.classification_confidence([0.2, 0.9, 0.1], score=True)
        # Regression: key was `"Confidence"`; docs now advertise `confidence`.
        assert "confidence" in result
