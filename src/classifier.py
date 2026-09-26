"""Stacking Ensemble Attack Classification for Network Intrusion Detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

# Optional XGBoost import with scikit-learn fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("xgboost not installed; falling back to HistGradientBoostingClassifier.")


class AttackClassifier:
    """Train, evaluate, save, load, and use a Stacking Ensemble classifier for attack detection."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        random_state: int = 42,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        use_stacking: bool = True,
    ) -> None:
        """Initialize the classifier with model configuration and artifact paths."""
        self.project_root = Path(__file__).resolve().parents[1]
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = Path(model_path) if model_path is not None else self.models_dir / "xgboost.joblib"
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_stacking = use_stacking
        self.model: Optional[Union[StackingClassifier, RandomForestClassifier]] = None

    def _create_primary_gb_classifier(self):
        """Create XGBoost or fallback HistGradientBoosting classifier."""
        if HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
            )
        return HistGradientBoostingClassifier(
            max_iter=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

    def train(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: np.ndarray) -> object:
        """Train the Stacking Ensemble classifier combining Gradient Boosting, RF, ExtraTrees, and HistGB with a Meta-Learner."""
        logger.info("Training Stacking Ensemble classifier")
        
        gb_base = self._create_primary_gb_classifier()

        if not self.use_stacking:
            self.model = gb_base
            self.model.fit(X_train, y_train)
            self.save_model()
            return self.model

        rf_base = RandomForestClassifier(
            n_estimators=100,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        et_base = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

        hgb_base = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        estimators = [
            ("gb_base", gb_base),
            ("rf", rf_base),
            ("extra_trees", et_base),
            ("hist_gb", hgb_base),
        ]

        meta_learner = LogisticRegression(max_iter=1000, random_state=self.random_state)

        try:
            self.model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3,
                n_jobs=-1,
                passthrough=False,
            )
            self.model.fit(X_train, y_train)
            logger.info("Stacking Ensemble classifier trained successfully")
        except Exception as exc:
            logger.warning("Stacking Ensemble failed (%s). Falling back to standalone base classifier.", exc)
            self.model = gb_base
            self.model.fit(X_train, y_train)

        self.save_model()
        return self.model

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: np.ndarray,
    ) -> dict[str, object]:
        """Evaluate the trained classifier and return standard classification metrics."""
        if self.model is None:
            self.load_model()

        try:
            predictions = self.model.predict(X_test)
        except Exception as exc:
            logger.exception("Model prediction during evaluation failed")
            raise RuntimeError("Failed to generate predictions for evaluation") from exc

        metrics: dict[str, object] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "classification_report": classification_report(y_test, predictions),
        }

        logger.info("Evaluation metrics: Accuracy=%.4f, Weighted F1=%.4f", metrics["accuracy"], metrics["f1"])
        return metrics

    def save_model(self) -> None:
        """Persist the trained classifier to disk."""
        if self.model is None:
            raise ValueError("No trained model available to save.")

        logger.info("Saving classifier to %s", self.model_path)
        try:
            joblib.dump(self.model, self.model_path)
        except Exception as exc:
            logger.exception("Failed to save classifier")
            raise RuntimeError("Unable to save the classifier") from exc

    def load_model(self) -> object:
        """Load a previously trained classifier from disk."""
        logger.info("Loading classifier from %s", self.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Classifier model file not found: {self.model_path}")

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            logger.exception("Failed to load classifier")
            raise RuntimeError("Unable to load the classifier") from exc

        return self.model

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the attack class label for the provided samples."""
        if self.model is None:
            self.load_model()

        try:
            return self.model.predict(X)
        except Exception as exc:
            logger.exception("Attack prediction failed")
            raise RuntimeError("Failed to predict attack type") from exc

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities for the provided samples."""
        if self.model is None:
            self.load_model()

        try:
            return self.model.predict_proba(X)
        except Exception as exc:
            logger.exception("Probability prediction failed")
            raise RuntimeError("Failed to predict attack probabilities") from exc
