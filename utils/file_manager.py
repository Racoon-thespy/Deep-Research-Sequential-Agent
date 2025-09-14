"""
File management utilities for Multi-Agent Market Research System
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import yaml
import csv
import tempfile

from config.settings import OUTPUTS_DIR, REPORTS_DIR, DATASETS_DIR, PROPOSALS_DIR
from utils.logger import get_logger, log_file_operation

logger = get_logger(__name__)

class FileManager:
    """Centralized file management utilities"""
    
    def __init__(self):
        self.ensure_directories_exist()
    
    def ensure_directories_exist(self):
        """Ensure all output directories exist"""
        directories = [OUTPUTS_DIR, REPORTS_DIR, DATASETS_DIR, PROPOSALS_DIR]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ensured: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {str(e)}")
    
    def _with_timestamp(self, file_path: Path, add_timestamp: bool) -> Path:
        """Return file path with timestamp if requested"""
        if not add_timestamp:
            return file_path
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return file_path.with_name(f"{file_path.stem}_{timestamp}{file_path.suffix}")

    def _atomic_write(self, file_path: Path, write_func) -> bool:
        """Safely write to a file using a temp file and atomic rename"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', delete=False, dir=file_path.parent, encoding='utf-8') as tmp:
                temp_path = Path(tmp.name)
                write_func(tmp)
            shutil.move(str(temp_path), str(file_path))
            return True
        except Exception as e:
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise e

    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path], 
                  indent: int = 2, add_timestamp: bool = False) -> bool:
        """Save data as JSON file"""
        file_path = self._with_timestamp(Path(file_path), add_timestamp)
        try:
            self._atomic_write(file_path, lambda f: json.dump(data, f, indent=indent, ensure_ascii=False, default=str))
            log_file_operation("write", str(file_path), "SUCCESS", file_path.stat().st_size)
            return True
        except Exception as e:
            log_file_operation("write", str(file_path), f"FAILED: {str(e)}")
            logger.error(f"Failed to save JSON file {file_path}: {str(e)}")
            return False
    
    def load_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load JSON file"""
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                logger.warning(f"JSON file not found: {file_path}")
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log_file_operation("read", str(file_path), "SUCCESS", file_path.stat().st_size)
            return data
        except Exception as e:
            log_file_operation("read", str(file_path), f"FAILED: {str(e)}")
            logger.error(f"Failed to load JSON file {file_path}: {str(e)}")
            return None

    def save_markdown(self, content: str, file_path: Union[str, Path], add_timestamp: bool = False) -> bool:
        """Save content as Markdown file"""
        file_path = self._with_timestamp(Path(file_path), add_timestamp)
        try:
            self._atomic_write(file_path, lambda f: f.write(content))
            log_file_operation("write", str(file_path), "SUCCESS", len(content.encode('utf-8')))
            return True
        except Exception as e:
            log_file_operation("write", str(file_path), f"FAILED: {str(e)}")
            logger.error(f"Failed to save Markdown file {file_path}: {str(e)}")
            return False

    def load_markdown(self, file_path: Union[str, Path]) -> Optional[str]:
        """Load Markdown file"""
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                logger.warning(f"Markdown file not found: {file_path}")
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            log_file_operation("read", str(file_path), "SUCCESS", file_path.stat().st_size)
            return content
        except Exception as e:
            log_file_operation("read", str(file_path), f"FAILED: {str(e)}")
            logger.error(f"Failed to load Markdown file {file_path}: {str(e)}")
            return None

    def save_yaml(self, data: Dict[str, Any], file_path: Union[str, Path], add_timestamp: bool = False) -> bool:
        """Save data as YAML file"""
        file_path = self._with_timestamp(Path(file_path), add_timestamp)
        try:
            self._atomic_write(file_path, lambda f: yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True))
            log_file_operation("write", str(file_path), "SUCCESS", file_path.stat().st_size)
            return True
        except Exception as e:
            log_file_operation("write", str(file_path), f"FAILED: {str(e)}")
            logger.error(f"Failed to save YAML file {file_path}: {str(e)}")
            return False

    def load_yaml(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load YAML file"""
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                logger.warning(f"YAML file not found: {file_path}")
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            log_file_operation("read", str(file_path), "SUCCESS", file_path.stat().st_size)
            return data
        except Exception as e:
            log_file_operation("read", str(file_path), f"FAILED: {str(e)}")
            logger.error(f"Failed to load YAML file {file_path}: {str(e)}")
            return None

    def save_csv(self, data: List[Dict[str, Any]], file_path: Union[str, Path], 
                 fieldnames: Optional[List[str]] = None, add_timestamp: bool = False) -> bool:
        """Save data as CSV file"""
        file_path = self._with_timestamp(Path(file_path), add_timestamp)
        try:
            if not data:
                logger.warning("No data to save to CSV")
                return False
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            self._atomic_write(file_path, lambda f: self._write_csv(f, data, fieldnames))
            log_file_operation("write", str(file_path), "SUCCESS", file_path.stat().st_size)
            return True
        except Exception as e:
            log_file_operation("write", str(file_path), f"FAILED: {str(e)}")
            logger.error(f"Failed to save CSV file {file_path}: {str(e)}")
            return False

    def _write_csv(self, file_obj, data: List[Dict[str, Any]], fieldnames: List[str]):
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)

    def load_csv(self, file_path: Union[str, Path]) -> Optional[List[Dict[str, Any]]]:
        """Load CSV file"""
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                logger.warning(f"CSV file not found: {file_path}")
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            log_file_operation("read", str(file_path), "SUCCESS", file_path.stat().st_size)
            return data
        except Exception as e:
            log_file_operation("read", str(file_path), f"FAILED: {str(e)}")
            logger.error(f"Failed to load CSV file {file_path}: {str(e)}")
            return None
