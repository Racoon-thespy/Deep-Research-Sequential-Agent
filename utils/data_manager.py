"""
Data management utilities for Multi-Agent Market Research System
Provides higher-level operations on structured data (CSV, JSON, YAML)
using pandas for dataframes and FileManager for safe file I/O.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from utils.file_manager import FileManager
from utils.logger import get_logger

logger = get_logger(__name__)


class DataManager:
    """Handles structured dataset operations, integrates with FileManager."""

    def __init__(self):
        self.file_manager = FileManager()

    # -------- CSV OPERATIONS (Pandas-based) --------
    def load_csv_df(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load CSV into pandas DataFrame.
        Returns empty DataFrame if file does not exist or fails to load.
        """
        try:
            if not Path(file_path).exists():
                logger.warning(f"CSV file not found: {file_path}")
                return pd.DataFrame()
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Loaded CSV into DataFrame: {file_path} (shape={df.shape})")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV into DataFrame: {file_path} | {e}")
            return pd.DataFrame()

    def save_csv_df(
        self, df: pd.DataFrame, file_path: Union[str, Path], add_timestamp: bool = False, **kwargs
    ) -> bool:
        """
        Save pandas DataFrame to CSV using FileManager for atomic writes.
        """
        try:
            records = df.to_dict(orient="records")
            result = self.file_manager.save_csv(records, file_path, add_timestamp=add_timestamp)
            if result:
                logger.info(f"Saved DataFrame to CSV: {file_path} (rows={len(df)})")
            return result
        except Exception as e:
            logger.error(f"Failed to save DataFrame to CSV: {file_path} | {e}")
            return False

    # -------- JSON OPERATIONS --------
    def load_json_dict(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON into dict. Returns empty dict if missing or invalid.
        """
        data = self.file_manager.load_json(file_path)
        return data or {}

    def save_json_dict(
        self, data: Dict[str, Any], file_path: Union[str, Path], add_timestamp: bool = False
    ) -> bool:
        """Save dict to JSON file."""
        return self.file_manager.save_json(data, file_path, add_timestamp=add_timestamp)

    # -------- YAML OPERATIONS --------
    def load_yaml_dict(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load YAML into dict. Returns empty dict if missing or invalid.
        """
        data = self.file_manager.load_yaml(file_path)
        return data or {}

    def save_yaml_dict(
        self, data: Dict[str, Any], file_path: Union[str, Path], add_timestamp: bool = False
    ) -> bool:
        """Save dict to YAML file."""
        return self.file_manager.save_yaml(data, file_path, add_timestamp=add_timestamp)

    # -------- DATA CACHING --------
    def cache_dataframe(self, df: pd.DataFrame, cache_dir: Union[str, Path], name: str) -> Path:
        """
        Cache a DataFrame to a CSV file inside a cache directory.
        Returns Path to cached file.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{name}.csv"
        self.save_csv_df(df, cache_path)
        logger.debug(f"Cached DataFrame at {cache_path}")
        return cache_path

    def load_cached_dataframe(self, cache_dir: Union[str, Path], name: str) -> pd.DataFrame:
        """
        Load a cached DataFrame if available, else return empty DataFrame.
        """
        cache_path = Path(cache_dir) / f"{name}.csv"
        return self.load_csv_df(cache_path)

