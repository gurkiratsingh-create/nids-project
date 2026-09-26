"""Adversarial Perturbation & Evasion Robustness Testing for NIDS Models."""

from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdversarialDefender:
    """Generates synthetic adversarial feature perturbations to test model robustness against evasive network traffic."""

    def __init__(self, perturbation_scale: float = 0.05, random_state: int = 42) -> None:
        """Initialize the defender with noise scale configuration."""
        self.perturbation_scale = perturbation_scale
        self.random_state = random_state

    def generate_adversarial_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        attack_type: str = "feature_jitter",
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply feature perturbation noise simulating evasive packet timing or payload padding."""
        np.random.seed(self.random_state)
        
        if isinstance(X, pd.DataFrame):
            X_adv = X.copy()
            numeric_cols = X_adv.select_dtypes(include=[np.number]).columns
            
            if attack_type == "feature_jitter":
                # Add Gaussian noise to packet duration and inter-arrival timing
                noise = np.random.normal(0, self.perturbation_scale, size=X_adv[numeric_cols].shape)
                X_adv[numeric_cols] = X_adv[numeric_cols] + noise
            elif attack_type == "payload_padding":
                # Artificially inflate packet length features to simulate byte padding
                padding_mask = [col for col in numeric_cols if "Length" in col or "Size" in col or "Bytes" in col]
                if padding_mask:
                    X_adv[padding_mask] = X_adv[padding_mask] * (1.0 + np.abs(np.random.normal(0, self.perturbation_scale, size=X_adv[padding_mask].shape)))
            return X_adv

        X_adv_arr = np.array(X, copy=True)
        noise = np.random.normal(0, self.perturbation_scale, size=X_adv_arr.shape)
        return X_adv_arr + noise

    def evaluate_evasion_robustness(
        self,
        model: object,
        X_clean: Union[pd.DataFrame, np.ndarray],
        y_true: np.ndarray,
        perturbation_scales: list[float] = [0.01, 0.05, 0.10, 0.20],
    ) -> pd.DataFrame:
        """Benchmark model accuracy decay across escalating adversarial perturbation scales."""
        logger.info("Evaluating model evasion robustness across perturbation levels %s", perturbation_scales)
        results = []
        
        # Clean baseline
        if hasattr(model, "predict"):
            y_pred_clean = model.predict(X_clean)
            acc_clean = (y_pred_clean == y_true).mean()
            results.append({"scale": 0.0, "accuracy": acc_clean, "decay": 0.0})
            
            for scale in perturbation_scales:
                self.perturbation_scale = scale
                X_adv = self.generate_adversarial_samples(X_clean, attack_type="feature_jitter")
                y_pred_adv = model.predict(X_adv)
                acc_adv = (y_pred_adv == y_true).mean()
                decay = acc_clean - acc_adv
                results.append({"scale": scale, "accuracy": acc_adv, "decay": decay})

        df_res = pd.DataFrame(results)
        logger.info("Evasion robustness evaluation completed successfully.")
        return df_res
