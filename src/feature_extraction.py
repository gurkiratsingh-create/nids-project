"""Feature extraction utilities for network traffic analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and validate network traffic features for machine learning pipelines."""

    def __init__(self) -> None:
        """Initialize the feature extractor with a canonical feature schema."""
        self._required_features = [
            "Source IP",
            "Destination IP",
            "Source Port",
            "Destination Port",
            "Protocol",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Bwd Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Packet Length Mean",
            "Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "SYN Flag Count",
            "ACK Flag Count",
            "RST Flag Count",
            "FIN Flag Count",
            "PSH Flag Count",
            "URG Flag Count",
            "Average Packet Size",
            "Inter Arrival Time",
        ]

    def extract_features(self, data: Union[pd.DataFrame, dict[str, Any], list[dict[str, Any]], str, Path]) -> pd.DataFrame:
        """Extract a feature DataFrame from structured flow data, CSV files, or simulated live traffic."""
        if isinstance(data, (str, Path)):
            return self._extract_from_csv(Path(data))

        if isinstance(data, dict):
            return self._extract_from_mapping(data)

        if isinstance(data, list):
            return self._extract_from_records(data)

        if isinstance(data, pd.DataFrame):
            return self._extract_from_dataframe(data)

        raise TypeError("Unsupported data type for feature extraction")

    def validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate the feature matrix and ensure required columns exist and are usable."""
        if features.empty:
            raise ValueError("No feature data provided for validation")

        missing_columns = [feature for feature in self._required_features if feature not in features.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")

        validated = features.copy()
        validated = self._handle_missing_values(validated)
        validated = self._coerce_numeric_columns(validated)
        validated = self._normalize_string_columns(validated)

        return validated

    def preprocess_live_packet(self, packet: dict[str, Any]) -> pd.DataFrame:
        """Prepare a single live or simulated packet into a one-row feature DataFrame."""
        if not isinstance(packet, dict):
            raise TypeError("Live packet data must be provided as a dictionary")

        row = self._build_feature_row(packet)
        return pd.DataFrame([row])

    def get_feature_names(self) -> list[str]:
        """Return the list of canonical feature names expected by the model pipeline."""
        return list(self._required_features)

    def _extract_from_csv(self, path: Path) -> pd.DataFrame:
        """Load features from a CSV file and validate them."""
        logger.info("Loading features from CSV: %s", path)
        try:
            dataframe = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to read feature CSV")
            raise RuntimeError(f"Unable to read CSV feature file: {path}") from exc

        return self._extract_from_dataframe(dataframe)

    def _extract_from_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Convert an existing DataFrame into the canonical feature schema."""
        logger.info("Extracting features from structured DataFrame")
        try:
            normalized = self._standardize_columns(dataframe)
            validated = self.validate_features(normalized)
            return validated
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to extract features from DataFrame")
            raise RuntimeError("Feature extraction from DataFrame failed") from exc

    def _extract_from_mapping(self, mapping: dict[str, Any]) -> pd.DataFrame:
        """Convert a single mapping into a one-row feature DataFrame."""
        logger.info("Extracting features from mapping input")
        return self.preprocess_live_packet(mapping)

    def _extract_from_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of dictionaries into a DataFrame of extracted features."""
        logger.info("Extracting features from list of records")
        if not records:
            raise ValueError("At least one record is required for feature extraction")

        frames = []
        for record in records:
            frames.append(self.preprocess_live_packet(record))

        combined = pd.concat(frames, ignore_index=True)
        return self.validate_features(combined)

    def _standardize_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to the canonical feature schema where possible."""
        normalized = dataframe.copy()
        rename_map = {
            "src_ip": "Source IP",
            "source ip": "Source IP",
            "dst_ip": "Destination IP",
            "destination ip": "Destination IP",
            "src_port": "Source Port",
            "source port": "Source Port",
            "dst_port": "Destination Port",
            "destination port": "Destination Port",
            "protocol": "Protocol",
            "flow duration": "Flow Duration",
            "total_fwd_packets": "Total Fwd Packets",
            "total bwd packets": "Total Bwd Packets",
            "total_length_of_fwd_packets": "Total Length of Fwd Packets",
            "total_length_of_bwd_packets": "Total Length of Bwd Packets",
            "packet_length_mean": "Packet Length Mean",
            "packet_length_std": "Packet Length Std",
            "flow_bytes_s": "Flow Bytes/s",
            "flow_packets_s": "Flow Packets/s",
            "syn_flag_count": "SYN Flag Count",
            "ack_flag_count": "ACK Flag Count",
            "rst_flag_count": "RST Flag Count",
            "fin_flag_count": "FIN Flag Count",
            "psh_flag_count": "PSH Flag Count",
            "urg_flag_count": "URG Flag Count",
            "average_packet_size": "Average Packet Size",
            "inter_arrival_time": "Inter Arrival Time",
        }

        normalized = normalized.rename(columns=rename_map)
        return normalized

    def _build_feature_row(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Create a one-row feature dictionary from a packet or flow record."""
        feature_row = {
            "Source IP": packet.get("Source IP", packet.get("src_ip", "unknown")),
            "Destination IP": packet.get("Destination IP", packet.get("dst_ip", "unknown")),
            "Source Port": packet.get("Source Port", packet.get("src_port", 0)),
            "Destination Port": packet.get("Destination Port", packet.get("dst_port", 0)),
            "Protocol": packet.get("Protocol", packet.get("protocol", "unknown")),
            "Flow Duration": packet.get("Flow Duration", packet.get("flow_duration", 0)),
            "Total Fwd Packets": packet.get("Total Fwd Packets", packet.get("total_fwd_packets", 0)),
            "Total Bwd Packets": packet.get("Total Bwd Packets", packet.get("total_bwd_packets", 0)),
            "Total Length of Fwd Packets": packet.get("Total Length of Fwd Packets", packet.get("total_length_of_fwd_packets", 0)),
            "Total Length of Bwd Packets": packet.get("Total Length of Bwd Packets", packet.get("total_length_of_bwd_packets", 0)),
            "Packet Length Mean": packet.get("Packet Length Mean", packet.get("packet_length_mean", 0.0)),
            "Packet Length Std": packet.get("Packet Length Std", packet.get("packet_length_std", 0.0)),
            "Flow Bytes/s": packet.get("Flow Bytes/s", packet.get("flow_bytes_s", 0.0)),
            "Flow Packets/s": packet.get("Flow Packets/s", packet.get("flow_packets_s", 0.0)),
            "SYN Flag Count": packet.get("SYN Flag Count", packet.get("syn_flag_count", 0)),
            "ACK Flag Count": packet.get("ACK Flag Count", packet.get("ack_flag_count", 0)),
            "RST Flag Count": packet.get("RST Flag Count", packet.get("rst_flag_count", 0)),
            "FIN Flag Count": packet.get("FIN Flag Count", packet.get("fin_flag_count", 0)),
            "PSH Flag Count": packet.get("PSH Flag Count", packet.get("psh_flag_count", 0)),
            "URG Flag Count": packet.get("URG Flag Count", packet.get("urg_flag_count", 0)),
            "Average Packet Size": packet.get("Average Packet Size", packet.get("average_packet_size", 0.0)),
            "Inter Arrival Time": packet.get("Inter Arrival Time", packet.get("inter_arrival_time", 0.0)),
        }

        return feature_row

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in a pragmatic, model-safe way."""
        filled = features.copy()
        for column in filled.columns:
            if filled[column].isna().sum() == 0:
                continue
            if filled[column].dtype == "object":
                filled[column] = filled[column].fillna("unknown")
            else:
                filled[column] = filled[column].fillna(0)
        return filled

    def _coerce_numeric_columns(self, features: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric-like columns to numeric values for model consumption."""
        converted = features.copy()
        for column in converted.columns:
            if column in {"Source IP", "Destination IP", "Protocol"}:
                continue
            try:
                converted[column] = pd.to_numeric(converted[column], errors="coerce")
            except Exception:
                converted[column] = converted[column].astype("float64")
        return converted

    def _normalize_string_columns(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize string-like columns so they remain consistent for downstream preprocessing."""
        normalized = features.copy()
        for column in ["Source IP", "Destination IP", "Protocol"]:
            if column in normalized.columns:
                normalized[column] = normalized[column].astype(str).str.strip()
        return normalized
