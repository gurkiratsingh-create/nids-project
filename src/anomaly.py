"""Isolation Forest-based anomaly detection for network traffic data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Train, save, load, and use an Isolation Forest model for anomaly detection."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """Initialize the anomaly detector with model path and configuration."""
        self.project_root = Path(__file__).resolve().parents[1]
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = Path(model_path) if model_path is not None else self.models_dir / "isolation_forest.joblib"
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None

    def train(self, X_train: Union[pd.DataFrame, np.ndarray]) -> IsolationForest:
        """Train the Isolation Forest model on the provided training data."""
        logger.info("Training Isolation Forest model")
        try:
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=200,
            )
            self.model.fit(X_train)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Isolation Forest training failed")
            raise RuntimeError("Failed to train the anomaly detection model") from exc

        self.save_model()
        logger.info("Isolation Forest training completed successfully")
        return self.model

    def save_model(self) -> None:
        """Persist the trained Isolation Forest model to disk."""
        if self.model is None:
            raise ValueError("No trained model available to save.")

        logger.info("Saving anomaly model to %s", self.model_path)
        try:
            joblib.dump(self.model, self.model_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to save anomaly detection model")
            raise RuntimeError("Unable to save the anomaly detection model") from exc

    def load_model(self) -> IsolationForest:
        """Load a previously trained Isolation Forest model from disk."""
        logger.info("Loading anomaly model from %s", self.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Anomaly model file not found: {self.model_path}")

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load anomaly detection model")
            raise RuntimeError("Unable to load the anomaly detection model") from exc

        return self.model

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict anomaly labels for the provided data."""
        if self.model is None:
            self.load_model()

        try:
            predictions = self.model.predict(X)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Anomaly prediction failed")
            raise RuntimeError("Failed to generate anomaly predictions") from exc

        return predictions

    def predict_anomaly_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return the anomaly decision function scores for the provided data."""
        if self.model is None:
            self.load_model()

        try:
            scores = self.model.decision_function(X)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Anomaly score computation failed")
            raise RuntimeError("Failed to compute anomaly scores") from exc

        return scores

    def predict_with_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Return both anomaly labels and anomaly scores for the provided data."""
        labels = self.predict(X)
        scores = self.predict_anomaly_score(X)
        return labels, scores
