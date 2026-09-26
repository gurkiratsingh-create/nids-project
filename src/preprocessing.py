"""Data preprocessing pipeline for the network intrusion detection project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Optional SMOTE support for handling rare attack class imbalance
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    logger.info("imbalanced-learn not installed. SMOTE oversampling will fall back to class weighting.")


class DataPreprocessor:
    """Prepare network traffic data for anomaly detection and classification."""

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        use_smote: bool = True,
    ) -> None:
        """Initialize the preprocessor with project paths and split settings."""
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data" / "raw"
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_path = self._resolve_dataset_path(dataset_path)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.use_smote = use_smote

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns: list[str] = []

    def _resolve_dataset_path(self, dataset_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the dataset path from an explicit argument or the default raw-data folder."""
        if dataset_path is not None:
            candidate = Path(dataset_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.project_root / candidate
            if candidate.exists():
                return candidate
            logger.warning(f"Dataset file not found at {candidate}. Fallback mechanism will be used.")
            return candidate

        self.data_dir.mkdir(parents=True, exist_ok=True)
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV dataset found in {self.data_dir}. Demo synthetic data generator will be used if needed.")
            return self.data_dir / "cicids2017_sample.csv"
        return csv_files[0]

    def load_dataset(self) -> pd.DataFrame:
        """Load the CICIDS2017 dataset or generate a synthetic benchmark sample if raw dataset is absent."""
        logger.info("Loading dataset from %s", self.dataset_path)
        if not self.dataset_path.exists():
            logger.info("Dataset path does not exist. Generating synthetic benchmark dataset matching CICIDS2017 schema...")
            return self._generate_synthetic_cicids2017()

        try:
            dataframe = pd.read_csv(self.dataset_path)
        except Exception as exc:
            logger.warning("Failed to read dataset file: %s. Generating synthetic fallback dataset.", exc)
            return self._generate_synthetic_cicids2017()

        if dataframe.empty:
            return self._generate_synthetic_cicids2017()

        logger.info("Dataset loaded successfully with %d rows and %d columns", len(dataframe), len(dataframe.columns))
        return dataframe

    def _generate_synthetic_cicids2017(self, num_samples: int = 5000) -> pd.DataFrame:
        """Generate a realistic synthetic CICIDS2017 flow benchmark dataset for testing and execution."""
        np.random.seed(self.random_state)
        classes = ["BENIGN", "DDoS", "DoS Hulk", "PortScan", "Bot", "FTP-Patator", "Web Attack"]
        probabilities = [0.70, 0.12, 0.08, 0.05, 0.02, 0.02, 0.01]
        
        labels = np.random.choice(classes, size=num_samples, p=probabilities)
        
        data = {
            "Source IP": [f"192.168.1.{np.random.randint(2, 254)}" for _ in range(num_samples)],
            "Destination IP": [f"10.0.0.{np.random.randint(2, 254)}" for _ in range(num_samples)],
            "Source Port": np.random.randint(1024, 65535, size=num_samples),
            "Destination Port": np.random.choice([80, 443, 22, 21, 8080, 53], size=num_samples),
            "Protocol": np.random.choice(["TCP", "UDP"], size=num_samples, p=[0.85, 0.15]),
            "Flow Duration": np.random.exponential(scale=100000, size=num_samples),
            "Total Fwd Packets": np.random.randint(1, 500, size=num_samples),
            "Total Bwd Packets": np.random.randint(0, 500, size=num_samples),
            "Total Length of Fwd Packets": np.random.exponential(scale=5000, size=num_samples),
            "Total Length of Bwd Packets": np.random.exponential(scale=10000, size=num_samples),
            "Packet Length Mean": np.random.uniform(40, 1460, size=num_samples),
            "Packet Length Std": np.random.uniform(0, 500, size=num_samples),
            "Flow Bytes/s": np.random.exponential(scale=50000, size=num_samples),
            "Flow Packets/s": np.random.exponential(scale=1000, size=num_samples),
            "SYN Flag Count": np.random.binomial(n=5, p=0.1, size=num_samples),
            "ACK Flag Count": np.random.binomial(n=10, p=0.6, size=num_samples),
            "RST Flag Count": np.random.binomial(n=3, p=0.05, size=num_samples),
            "FIN Flag Count": np.random.binomial(n=2, p=0.05, size=num_samples),
            "PSH Flag Count": np.random.binomial(n=5, p=0.2, size=num_samples),
            "URG Flag Count": np.random.binomial(n=1, p=0.01, size=num_samples),
            "Average Packet Size": np.random.uniform(50, 1500, size=num_samples),
            "Inter Arrival Time": np.random.exponential(scale=50, size=num_samples),
            "Label": labels,
        }
        
        df = pd.DataFrame(data)
        logger.info("Generated synthetic dataset with %d rows and %d columns", len(df), len(df.columns))
        return df

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Run the full preprocessing pipeline with scaling, encoding, and SMOTE rebalancing."""
        dataframe = self.load_dataset()
        dataframe = self._clean_dataframe(dataframe)
        features, labels = self._split_features_and_labels(dataframe)
        features = self._encode_categorical_features(features)
        
        self.feature_columns = list(features.columns)
        features = self._scale_features(features)
        X_train, X_test, y_train, y_test = self._split_train_test(features, labels)

        # Apply SMOTE oversampling to training set if enabled to resolve class imbalance
        if self.use_smote and HAS_SMOTE:
            try:
                logger.info("Applying SMOTE oversampling to training data to handle class imbalance...")
                smote = SMOTE(random_state=self.random_state, k_neighbors=min(3, len(X_train)-1))
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                logger.info("SMOTE rebalancing complete. Resampled train shape: %s", X_train_res.shape)
                X_train = pd.DataFrame(X_train_res, columns=self.feature_columns)
                y_train = y_train_res
            except Exception as exc:
                logger.warning("SMOTE oversampling failed (%s); proceeding with un-sampled training set.", exc)

        self._save_artifacts()
        return X_train, X_test, y_train, y_test

    def _clean_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, handle missing values, and drop irrelevant columns."""
        cleaned = dataframe.copy()
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

        for column in cleaned.columns:
            if cleaned[column].isna().sum() == 0:
                continue
            if cleaned[column].dtype in ["object", "category"]:
                mode_val = cleaned[column].mode(dropna=True)
                cleaned[column] = cleaned[column].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
            else:
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())

        unnecessary_columns = {
            "Flow ID",
            "Timestamp",
            "Source IP",
            "Destination IP",
            "Source Port",
            "Destination Port",
            "Unnamed: 0",
        }
        existing_columns = [column for column in unnecessary_columns if column in cleaned.columns]
        if existing_columns:
            cleaned = cleaned.drop(columns=existing_columns)
            logger.info("Dropped metadata columns: %s", ", ".join(existing_columns))

        return cleaned

    def _split_features_and_labels(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Select the target column and separate features from labels."""
        target_column = self._resolve_target_column(dataframe)
        labels = dataframe[target_column].astype(str)

        feature_frame = dataframe.drop(columns=[target_column])
        if feature_frame.empty:
            raise ValueError("No feature columns remain after preprocessing.")

        return feature_frame, labels.to_numpy()

    def _resolve_target_column(self, dataframe: pd.DataFrame) -> str:
        """Find the label column from a list of common possible names."""
        if self.target_column is not None and self.target_column in dataframe.columns:
            return self.target_column

        candidates = ["Label", "label", "Attack", "attack", "Label/Attack", "Class"]
        for candidate in candidates:
            if candidate in dataframe.columns:
                return candidate

        raise KeyError("Unable to find a target label column in the dataset.")

    def _encode_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to numeric form using one-hot encoding."""
        encoded = features.copy()
        categorical_columns = encoded.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if categorical_columns:
            encoded = pd.get_dummies(encoded, columns=categorical_columns, dummy_na=False)

        return encoded

    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardize numerical features and store the fitted scaler."""
        scaled_values = self.scaler.fit_transform(features)
        scaled_frame = pd.DataFrame(
            scaled_values,
            columns=features.columns,
            index=features.index,
        )
        return scaled_frame

    def _split_train_test(self, features: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split the data into training and testing partitions."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=labels,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        return X_train, X_test, y_train_encoded, y_test_encoded

    def _save_artifacts(self) -> None:
        """Persist the fitted scaler, label encoder, and feature column names to the models directory."""
        scaler_path = self.models_dir / "scaler.joblib"
        encoder_path = self.models_dir / "label_encoder.joblib"
        features_path = self.models_dir / "feature_columns.joblib"

        try:
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoder, encoder_path)
            joblib.dump(self.feature_columns, features_path)
        except Exception as exc:
            logger.exception("Failed to save preprocessing artifacts")
            raise RuntimeError("Unable to save preprocessing artifacts to the models directory") from exc

        logger.info("Saved preprocessing artifacts to %s", self.models_dir)
