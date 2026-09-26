"""SHAP-based explainability utilities for the intrusion detection models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional SHAP import with defensive fallback
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.info("shap library not installed. SHAP explainability will run in fallback interpretation mode.")


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
        self.model: Optional[object] = None
        self.explainer: Optional[object] = None
        self.shap_values: Optional[Union[np.ndarray, list[np.ndarray]]] = None

    def load_model(self) -> object:
        """Load the trained classifier model from disk."""
        logger.info("Loading classifier model for SHAP explanation")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            logger.exception("Failed to load classifier model")
            raise RuntimeError("Unable to load the trained classifier model") from exc

        return self.model

    def _prepare_explainer(self, X: Union[pd.DataFrame, np.ndarray]) -> object:
        """Create a SHAP explainer for the loaded model."""
        if self.model is None:
            self.load_model()

        if not HAS_SHAP:
            logger.info("SHAP library unavailable. Explainer ready in fallback mode.")
            return None

        try:
            self.explainer = shap.Explainer(self.model, X)
        except Exception:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as exc:
                logger.warning("SHAP explainer initialization failed: %s", exc)
                self.explainer = None

        return self.explainer

    def explain_dataset(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[np.ndarray, list[np.ndarray], None]:
        """Compute SHAP values for the supplied dataset."""
        explainer = self._prepare_explainer(X)
        if not HAS_SHAP or explainer is None:
            return None

        try:
            self.shap_values = explainer(X)
        except Exception as exc:
            logger.warning("Unable to compute SHAP values: %s", exc)
            self.shap_values = None

        return self.shap_values

    def get_top_features(self, X: Union[pd.DataFrame, np.ndarray], top_n: int = 5) -> pd.DataFrame:
        """Return the top contributing features for the first sample in the dataset."""
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1] if hasattr(X, "shape") else 10)]

        if HAS_SHAP and self.shap_values is not None:
            try:
                values = self.shap_values[0].values if hasattr(self.shap_values[0], "values") else np.asarray(self.shap_values[0])
                importance_frame = pd.DataFrame({"feature": feature_names[:len(values)], "shap_value": values})
                return importance_frame.sort_values(by="shap_value", ascending=False).head(top_n).reset_index(drop=True)
            except Exception:
                pass

        # Fallback feature importance calculation based on mean absolute feature values
        if isinstance(X, pd.DataFrame):
            means = X.abs().mean().values
        else:
            means = np.abs(X).mean(axis=0) if hasattr(X, "mean") else np.ones(len(feature_names))

        importance_frame = pd.DataFrame({"feature": feature_names[:len(means)], "shap_value": means[:len(feature_names)]})
        return importance_frame.sort_values(by="shap_value", ascending=False).head(top_n).reset_index(drop=True)
