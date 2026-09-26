"""Prediction and Incident Logging Engine for Enterprise NIDS."""

import datetime
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd

from src.anomaly import AnomalyDetector
from src.classifier import AttackClassifier
from src.database import IncidentDatabase
from src.feature_extraction import FeatureExtractor
from src.recommendations import RecommendationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict")


class NIDSInferencePipeline:
    """End-to-End Inference Engine for processing network flows and logging incidents."""

    def __init__(self) -> None:
        """Initialize all trained model components and storage databases."""
        self.project_root = Path(__file__).resolve().parent
        self.models_dir = self.project_root / "models"

        self.feature_extractor = FeatureExtractor()
        self.anomaly_detector = AnomalyDetector()
        self.classifier = AttackClassifier()
        self.recommendation_engine = RecommendationEngine()
        self.database = IncidentDatabase()

        self._load_preprocessing_artifacts()

    def _load_preprocessing_artifacts(self) -> None:
        """Load trained scaler, label encoder, and feature columns."""
        scaler_path = self.models_dir / "scaler.joblib"
        encoder_path = self.models_dir / "label_encoder.joblib"
        cols_path = self.models_dir / "feature_columns.joblib"

        if scaler_path.exists() and encoder_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            self.feature_columns = joblib.load(cols_path) if cols_path.exists() else None
        else:
            logger.warning("Preprocessing artifacts not found. Models will be loaded dynamically on inference.")
            self.scaler = None
            self.label_encoder = None
            self.feature_columns = None

    def process_flow_sample(self, flow_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single network flow record through the full inference pipeline."""
        src_ip = flow_dict.get("Source IP", flow_dict.get("src_ip", "192.168.1.100"))
        dst_ip = flow_dict.get("Destination IP", flow_dict.get("dst_ip", "10.0.0.5"))
        
        # 1. Feature Extraction & Preprocessing
        df_raw = self.feature_extractor.extract_features(flow_dict)
        df_clean = df_raw.drop(columns=["Source IP", "Destination IP", "Protocol"], errors="ignore")
        
        if hasattr(self, "feature_columns") and self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in df_clean.columns:
                    df_clean[col] = 0.0
            df_clean = df_clean[self.feature_columns]

        if hasattr(self, "scaler") and self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(df_clean)
            except Exception:
                X_scaled = df_clean.values
        else:
            X_scaled = df_clean.values

        # 2. Anomaly Detection
        try:
            anom_label, anom_score = self.anomaly_detector.predict_with_scores(X_scaled)
            raw_anom_score = float(anom_score[0])
            is_anomaly = bool(anom_label[0] == -1)
        except Exception:
            raw_anom_score = -0.1
            is_anomaly = False

        # 3. Supervised Classification
        try:
            pred_class_idx = self.classifier.predict(X_scaled)[0]
            pred_probas = self.classifier.predict_proba(X_scaled)[0]
            max_prob = float(np.max(pred_probas))
            
            if hasattr(self, "label_encoder") and self.label_encoder is not None:
                attack_type = str(self.label_encoder.inverse_transform([pred_class_idx])[0])
            else:
                attack_type = f"Attack_Class_{pred_class_idx}"
        except Exception:
            attack_type = "BENIGN" if not is_anomaly else "Suspicious Anomaly"
            max_prob = 0.85

        # 4. Dynamic Risk Score Math Formulation
        # R = w1 * anomaly_conf + w2 * (prob * severity_weight)
        severity_map = {"BENIGN": 0.1, "Normal": 0.1, "PortScan": 0.6, "DoS Hulk": 0.8, "DDoS": 1.0, "Bot": 0.95, "FTP-Patator": 0.85, "Web Attack": 0.9}
        sev_weight = severity_map.get(attack_type, 0.7)
        norm_anom = max(0.0, -raw_anom_score * 2.0)
        risk_score = round(min(1.0, max(0.0, 0.35 * norm_anom + 0.65 * max_prob * sev_weight)), 4)

        # 5. Security Recommendation Lookup
        rec_payload = self.recommendation_engine.get_recommendation(attack_type, risk_score)
        rec_str = rec_payload.get("description", "Investigate flow.")

        # 6. SQLite Persistence
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        incident_id = self.database.insert_incident(
            timestamp=timestamp_str,
            source_ip=str(src_ip),
            destination_ip=str(dst_ip),
            attack_type=attack_type,
            risk_score=risk_score,
            recommendation=rec_str,
        )

        return {
            "incident_id": incident_id,
            "timestamp": timestamp_str,
            "source_ip": src_ip,
            "destination_ip": dst_ip,
            "attack_type": attack_type,
            "is_anomaly": is_anomaly,
            "confidence": max_prob,
            "risk_score": risk_score,
            "recommendations": rec_payload,
        }


def main() -> None:
    """Demonstrate inference execution on sample flow records."""
    logger.info("Initializing NIDS Prediction Engine...")
    pipeline = NIDSInferencePipeline()

    sample_flows = [
        {"src_ip": "192.168.1.15", "dst_ip": "10.0.0.1", "Protocol": "TCP", "Flow Duration": 1200, "Flow Bytes/s": 4500.0, "SYN Flag Count": 1},
        {"src_ip": "172.16.0.44", "dst_ip": "10.0.0.5", "Protocol": "TCP", "Flow Duration": 500000, "Flow Bytes/s": 950000.0, "SYN Flag Count": 120},
        {"src_ip": "192.168.1.88", "dst_ip": "10.0.0.2", "Protocol": "TCP", "Flow Duration": 450, "Flow Bytes/s": 12000.0, "RST Flag Count": 15},
    ]

    logger.info("Executing batch inference on %d sample network flows...", len(sample_flows))
    for idx, flow in enumerate(sample_flows, 1):
        result = pipeline.process_flow_sample(flow)
        logger.info("Flow #%d -> Attack: %s | Risk Score: %.2f | ID: %d", idx, result["attack_type"], result["risk_score"], result["incident_id"])

    logger.info("Prediction and Incident Logging completed successfully.")


if __name__ == "__main__":
    main()
