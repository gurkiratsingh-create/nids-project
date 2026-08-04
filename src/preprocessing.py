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


class DataPreprocessor:
    """Prepare network traffic data for anomaly detection and classification."""

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
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

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def _resolve_dataset_path(self, dataset_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the dataset path from an explicit argument or the default raw-data folder."""
        if dataset_path is not None:
            candidate = Path(dataset_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.project_root / candidate
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"Dataset file not found: {candidate}")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV dataset found in {self.data_dir}. Place the CICIDS2017 CSV file there."
            )
        return csv_files[0]

    def load_dataset(self) -> pd.DataFrame:
        """Load the CICIDS2017 dataset from disk."""
        logger.info("Loading dataset from %s", self.dataset_path)
        try:
            dataframe = pd.read_csv(self.dataset_path)
        except FileNotFoundError as exc:
            logger.exception("Dataset file was not found")
            raise RuntimeError(f"Unable to locate the dataset file: {self.dataset_path}") from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to read dataset")
            raise RuntimeError(f"Unable to read the dataset file: {self.dataset_path}") from exc

        if dataframe.empty:
            raise ValueError("The loaded dataset is empty.")

        logger.info("Dataset loaded successfully with %d rows and %d columns", len(dataframe), len(dataframe.columns))
        return dataframe

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Run the full preprocessing pipeline and return train/test feature and label splits."""
        dataframe = self.load_dataset()
        dataframe = self._clean_dataframe(dataframe)
        features, labels = self._split_features_and_labels(dataframe)
        features = self._encode_categorical_features(features)
        features = self._scale_features(features)
        X_train, X_test, y_train, y_test = self._split_train_test(features, labels)
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
                cleaned[column] = cleaned[column].fillna(cleaned[column].mode(dropna=True).iloc[0] if not cleaned[column].mode(dropna=True).empty else "unknown")
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
            logger.info("Dropped columns: %s", ", ".join(existing_columns))

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
        """Persist the fitted scaler and label encoder to the models directory."""
        scaler_path = self.models_dir / "scaler.joblib"
        encoder_path = self.models_dir / "label_encoder.joblib"

        try:
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoder, encoder_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to save preprocessing artifacts")
            raise RuntimeError("Unable to save preprocessing artifacts to the models directory") from exc

        logger.info("Saved scaler to %s", scaler_path)
        logger.info("Saved label encoder to %s", encoder_path)
