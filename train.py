"""End-to-End Model Training Pipeline Script for Enterprise NIDS."""

import logging
import sys
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly import AnomalyDetector
from src.classifier import AttackClassifier
from src.explainability import SHAPExplainer
from src.preprocessing import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def main() -> None:
    """Run full training pipeline: Preprocessing -> Isolation Forest -> Stacking Classifier -> SHAP Initialization."""
    logger.info("==================================================")
    logger.info(" Starting NIDS Enterprise Model Training Pipeline ")
    logger.info("==================================================")

    # 1. Data Preprocessing & Scaling with SMOTE Rebalancing
    preprocessor = DataPreprocessor(use_smote=True)
    logger.info("Step 1/4: Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    logger.info("Dataset prepared. Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)

    # 2. Unsupervised Anomaly Detection (Isolation Forest)
    logger.info("Step 2/4: Training Isolation Forest Anomaly Detector...")
    anomaly_detector = AnomalyDetector(contamination=0.05, random_state=42)
    anomaly_detector.train(X_train)
    logger.info("Isolation Forest model trained and saved successfully.")

    # 3. Supervised Stacking Ensemble Attack Classification
    logger.info("Step 3/4: Training Stacking Ensemble Classifier (XGBoost + RF + ExtraTrees + HistGB)...")
    classifier = AttackClassifier(use_stacking=True, random_state=42)
    classifier.train(X_train, y_train)
    
    # Evaluate Classifier Performance
    metrics = classifier.evaluate(X_test, y_test)
    logger.info("Classifier Test Accuracy: %.4f", metrics["accuracy"])
    logger.info("Classifier Weighted F1-Score: %.4f", metrics["f1"])

    # 4. Initialize SHAP Explainer
    logger.info("Step 4/4: Initializing SHAP Explainability Engine...")
    try:
        explainer = SHAPExplainer()
        explainer.explain_dataset(X_test.iloc[:50] if hasattr(X_test, "iloc") else X_test[:50])
        logger.info("SHAP explainer initialized successfully.")
    except Exception as exc:
        logger.warning("SHAP explainer initialization deferred: %s", exc)

    logger.info("==================================================")
    logger.info(" Pipeline Training Completed Successfully!       ")
    logger.info(" All model artifacts persisted to 'models/' dir.  ")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
