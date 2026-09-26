"""Statistical Concept Drift Monitoring for Dynamic Network Flows."""

from __future__ import annotations

import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class ConceptDriftMonitor:
    """Monitors distribution shifts in continuous network flow features using two-sample statistical tests."""

    def __init__(self, significance_level: float = 0.05) -> None:
        """Initialize the monitor with a target statistical significance threshold alpha."""
        self.significance_level = significance_level

    def detect_feature_drift(
        self,
        reference_data: Union[pd.DataFrame, np.ndarray],
        current_data: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, Union[bool, dict]]:
        """Perform Kolmogorov-Smirnov (KS) two-sample test on continuous features."""
        logger.info("Executing Kolmogorov-Smirnov drift test on incoming network stream...")
        
        drift_results = {}
        drift_detected_flag = False

        if isinstance(reference_data, pd.DataFrame) and isinstance(current_data, pd.DataFrame):
            common_cols = [col for col in reference_data.columns if col in current_data.columns]
            numeric_cols = reference_data[common_cols].select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                stat, p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
                is_drifted = p_value < self.significance_level
                if is_drifted:
                    drift_detected_flag = True

                drift_results[col] = {
                    "ks_statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": bool(is_drifted),
                }

        return {
            "overall_drift_detected": drift_detected_flag,
            "feature_details": drift_results,
        }

    def page_hinkley_test(self, data_stream: np.ndarray, delta: float = 0.005, threshold: float = 50.0) -> bool:
        """Page-Hinkley cumulative sum test for detecting abrupt distribution shifts in stream mean."""
        mean_val = np.mean(data_stream)
        cum_sum = 0.0
        min_cum_sum = 0.0

        for val in data_stream:
            cum_sum += val - mean_val - delta
            if cum_sum < min_cum_sum:
                min_cum_sum = cum_sum
            if cum_sum - min_cum_sum > threshold:
                logger.warning("Page-Hinkley drift alarm triggered! Continuous mean shifted drastically.")
                return True

        return False
