"""SHAP-based explainability utilities for the intrusion detection models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Generate SHAP-based explanations for model predictions and save plots to disk."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the explainer and resolve artifact paths."""
        self.project_root = Path(__file__).resolve().parents[1]
        self.models_dir = self.project_root / "models"
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = Path(model_path) if model_path is not None else self.models_dir / "xgboost.joblib"
        self.model: Optional[XGBClassifier] = None
        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[Union[np.ndarray, list[np.ndarray]]] = None

    def load_model(self) -> XGBClassifier:
        """Load the trained XGBoost model from disk."""
        logger.info("Loading XGBoost model for SHAP explanation")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load XGBoost model")
            raise RuntimeError("Unable to load the trained classifier model") from exc

        return self.model

    def _prepare_explainer(self, X: Union[pd.DataFrame, np.ndarray]) -> shap.Explainer:
        """Create a SHAP explainer for the loaded model."""
        if self.model is None:
            self.load_model()

        try:
            self.explainer = shap.Explainer(self.model, X)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialize SHAP explainer")
            raise RuntimeError("Unable to initialize SHAP explainer") from exc

        return self.explainer

    def explain_dataset(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Compute SHAP values for the supplied dataset."""
        explainer = self._prepare_explainer(X)
        try:
            self.shap_values = explainer(X)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to compute SHAP values")
            raise RuntimeError("Unable to compute SHAP values") from exc

        return self.shap_values

    def generate_summary_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        output_file: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate and save a SHAP summary plot."""
        if self.shap_values is None:
            self.explain_dataset(X)

        path = self._resolve_output_path(output_file, "shap_summary_plot.png")
        try:
            shap.summary_plot(self.shap_values, X, show=False)
            shap.plots.savefig(path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to generate SHAP summary plot")
            raise RuntimeError("Unable to save SHAP summary plot") from exc

        logger.info("Saved SHAP summary plot to %s", path)
        return path

    def generate_waterfall_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_index: int = 0,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate and save a SHAP waterfall plot for a single prediction."""
        if self.shap_values is None:
            self.explain_dataset(X)

        path = self._resolve_output_path(output_file, "shap_waterfall_plot.png")
        try:
            shap.plots.waterfall(self.shap_values[sample_index], show=False)
            shap.plots.savefig(path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to generate SHAP waterfall plot")
            raise RuntimeError("Unable to save SHAP waterfall plot") from exc

        logger.info("Saved SHAP waterfall plot to %s", path)
        return path

    def generate_force_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_index: int = 0,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate and save a SHAP force plot for a single prediction."""
        if self.shap_values is None:
            self.explain_dataset(X)

        path = self._resolve_output_path(output_file, "shap_force_plot.png")
        try:
            force_plot = shap.plots.force(self.shap_values[sample_index], matplotlib=False)
            shap.save_html(path.with_suffix(".html"), force_plot)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to generate SHAP force plot")
            raise RuntimeError("Unable to save SHAP force plot") from exc

        logger.info("Saved SHAP force plot to %s", path.with_suffix(".html"))
        return path.with_suffix(".html")

    def get_top_features(self, X: Union[pd.DataFrame, np.ndarray], top_n: int = 5) -> pd.DataFrame:
        """Return the top contributing features for the first sample in the dataset."""
        if self.shap_values is None:
            self.explain_dataset(X)

        try:
            if isinstance(self.shap_values, list):
                values = self.shap_values[0]
            else:
                values = self.shap_values

            if hasattr(values, "values"):
                feature_values = values.values[0]
            else:
                feature_values = np.asarray(values[0])

            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{index}" for index in range(len(feature_values))]

            importance_frame = pd.DataFrame(
                {"feature": feature_names, "shap_value": feature_values}
            )
            importance_frame = importance_frame.sort_values(by="shap_value", ascending=False)
            return importance_frame.head(top_n).reset_index(drop=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to extract top SHAP features")
            raise RuntimeError("Unable to determine top contributing features") from exc

    def _resolve_output_path(self, output_file: Optional[Union[str, Path]], default_name: str) -> Path:
        """Resolve a writable output path for a SHAP plot."""
        if output_file is None:
            return self.reports_dir / default_name

        path = Path(output_file)
        if not path.is_absolute():
            path = self.reports_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
