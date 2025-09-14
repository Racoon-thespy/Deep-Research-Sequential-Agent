"""
Input validation utilities for Multi-Agent Market Research System
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse
from config.settings import (
    SUPPORTED_INDUSTRIES,
    MIN_COMPANY_NAME_LENGTH,
    MAX_COMPANY_NAME_LENGTH,
    MIN_INDUSTRY_DESCRIPTION_LENGTH,
    MAX_INDUSTRY_DESCRIPTION_LENGTH,
    ERROR_MESSAGES
)
from utils.logger import get_logger, log_warning, log_error

logger = get_logger(__name__)

class ValidationError(Exception):
    """Custom validation error"""
    pass

def validate_company_name(company_name: str) -> Tuple[bool, str]:
    """
    Validate company name
    
    Args:
        company_name (str): Company name to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not company_name or not company_name.strip():
        return False, "Company name cannot be empty"
    
    company_name = company_name.strip()
    
    if len(company_name) < MIN_COMPANY_NAME_LENGTH:
        return False, f"Company name too short (minimum {MIN_COMPANY_NAME_LENGTH} characters)"
    
    if len(company_name) > MAX_COMPANY_NAME_LENGTH:
        return False, f"Company name too long (maximum {MAX_COMPANY_NAME_LENGTH} characters)"
    
    # Check for valid characters (letters, numbers, spaces, common punctuation)
    if not re.match(r'^[a-zA-Z0-9\s\.\-_&,()]+$', company_name):
        return False, "Company name contains invalid characters"
    
    return True, ""

def validate_industry(industry: str) -> Tuple[bool, str]:
    """
    Validate industry selection
    
    Args:
        industry (str): Industry to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not industry or not industry.strip():
        return False, "Industry cannot be empty"
    
    industry = industry.strip()
    
    if industry not in SUPPORTED_INDUSTRIES:
        return False, f"Industry must be one of: {', '.join(SUPPORTED_INDUSTRIES)}"
    
    return True, ""

def validate_industry_description(description: str) -> Tuple[bool, str]:
    """
    Validate industry description
    
    Args:
        description (str): Description to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not description or not description.strip():
        return False, "Industry description cannot be empty"
    
    description = description.strip()
    
    if len(description) < MIN_INDUSTRY_DESCRIPTION_LENGTH:
        return False, f"Description too short (minimum {MIN_INDUSTRY_DESCRIPTION_LENGTH} characters)"
    
    if len(description) > MAX_INDUSTRY_DESCRIPTION_LENGTH:
        return False, f"Description too long (maximum {MAX_INDUSTRY_DESCRIPTION_LENGTH} characters)"
    
    return True, ""

def validate_api_key(api_key: str, api_name: str) -> Tuple[bool, str]:
    """
    Validate API key format
    
    Args:
        api_key (str): API key to validate
        api_name (str): Name of the API service
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not api_key or not api_key.strip():
        return False, f"{api_name} API key cannot be empty"
    
    api_key = api_key.strip()
    
    # Basic format validation
    if len(api_key) < 10:
        return False, f"{api_name} API key appears to be too short"
    
    # Check for placeholder values
    placeholder_patterns = [
        "your_api_key",
        "your_key_here",
        "replace_me",
        "api_key_here",
        "insert_key"
    ]
    
    if any(pattern in api_key.lower() for pattern in placeholder_patterns):
        return False, f"{api_name} API key appears to be a placeholder value"
    
    return True, ""

def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate URL format
    
    Args:
        url (str): URL to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not url or not url.strip():
        return False, "URL cannot be empty"
    
    url = url.strip()
    
    try:
        result = urlparse(url)
        if not result.scheme or not result.netloc:
            return False, "Invalid URL format"
        
        if result.scheme not in ['http', 'https']:
            return False, "URL must use HTTP or HTTPS protocol"
            
    except Exception as e:
        return False, f"Invalid URL: {str(e)}"
    
    return True, ""

def validate_search_query(query: str) -> Tuple[bool, str]:
    """
    Validate search query
    
    Args:
        query (str): Search query to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Search query cannot be empty"
    
    query = query.strip()
    
    if len(query) < 2:
        return False, "Search query too short (minimum 2 characters)"
    
    if len(query) > 500:
        return False, "Search query too long (maximum 500 characters)"
    
    return True, ""

def validate_json_data(data: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Validate JSON data
    
    Args:
        data (str): JSON string to validate
        
    Returns:
        Tuple[bool, str, Optional[Dict]]: (is_valid, error_message, parsed_data)
    """
    if not data or not data.strip():
        return False, "JSON data cannot be empty", None
    
    try:
        parsed_data = json.loads(data)
        return True, "", parsed_data
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}", None

def validate_file_path(file_path: str, must_exist: bool = False) -> Tuple[bool, str]:
    """
    Validate file path
    
    Args:
        file_path (str): File path to validate
        must_exist (bool): Whether file must already exist
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not file_path or not file_path.strip():
        return False, "File path cannot be empty"
    
    try:
        from pathlib import Path
        path = Path(file_path.strip())
        
        if must_exist and not path.exists():
            return False, f"File does not exist: {file_path}"
        
        # Check for valid file extension for expected file types
        valid_extensions = ['.md', '.txt', '.json', '.csv', '.xlsx', '.pdf']
        if path.suffix and path.suffix.lower() not in valid_extensions:
            log_warning(f"Unusual file extension: {path.suffix}")
        
        return True, ""
        
    except Exception as e:
        return False, f"Invalid file path: {str(e)}"

def validate_use_case_data(use_case: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate use case data structure
    
    Args:
        use_case (Dict[str, Any]): Use case data to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    required_fields = [
        'title', 'description', 'problem_statement', 
        'solution', 'benefits', 'complexity'
    ]
    
    for field in required_fields:
        if field not in use_case:
            errors.append(f"Missing required field: {field}")
        elif not use_case[field] or not str(use_case[field]).strip():
            errors.append(f"Empty required field: {field}")
    
    # Validate complexity level
    if 'complexity' in use_case:
        valid_complexity = ['Low', 'Medium', 'High']
        if use_case['complexity'] not in valid_complexity:
            errors.append(f"Invalid complexity level. Must be one of: {valid_complexity}")
    
    # Validate benefits structure
    if 'benefits' in use_case and isinstance(use_case['benefits'], list):
        if len(use_case['benefits']) == 0:
            errors.append("Benefits list cannot be empty")
    
    return len(errors) == 0, errors

def validate_dataset_info(dataset: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate dataset information
    
    Args:
        dataset (Dict[str, Any]): Dataset info to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    required_fields = ['title', 'url', 'description', 'source']
    
    for field in required_fields:
        if field not in dataset:
            errors.append(f"Missing required field: {field}")
        elif not dataset[field] or not str(dataset[field]).strip():
            errors.append(f"Empty required field: {field}")
    
    # Validate URL
    if 'url' in dataset:
        is_valid_url, url_error = validate_url(dataset['url'])
        if not is_valid_url:
            errors.append(f"Invalid URL: {url_error}")
    
    # Validate source
    if 'source' in dataset:
        valid_sources = ['kaggle', 'huggingface', 'github', 'other']
        if dataset['source'].lower() not in valid_sources:
            errors.append(f"Invalid source. Must be one of: {valid_sources}")
    
    return len(errors) == 0, errors

def sanitize_input(text: str, max_length: int = None) -> str:
    """
    Sanitize user input
    
    Args:
        text (str): Text to sanitize
        max_length (int): Maximum length to truncate to
        
    Returns:
        str: Sanitized text
    """
    if not text:
        return ""
    
    # Remove dangerous characters and scripts
    sanitized = re.sub(r'[<>"\']', '', text.strip())
    
    # Remove excessive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Truncate if necessary
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length].strip()
    
    return sanitized

def validate_system_requirements() -> Tuple[bool, List[str]]:
    """
    Validate system requirements and configuration
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        from config.settings import validate_config
        if not validate_config():
            errors.append("Configuration validation failed - check API keys")
    except ImportError:
        errors.append("Configuration module not found")
    except Exception as e:
        errors.append(f"Configuration error: {str(e)}")
    
    # Check required directories
    try:
        from config.settings import OUTPUTS_DIR, REPORTS_DIR, DATASETS_DIR, PROPOSALS_DIR
        directories = [OUTPUTS_DIR, REPORTS_DIR, DATASETS_DIR, PROPOSALS_DIR]
        
        for directory in directories:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {directory}: {str(e)}")
    except Exception as e:
        errors.append(f"Directory validation error: {str(e)}")
    
    return len(errors) == 0, errors

class InputValidator:
    """Class-based validator for chaining validations"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def add_validation(self, validation_func, *args, **kwargs):
        """Add a validation function to the chain"""
        try:
            result = validation_func(*args, **kwargs)
            if isinstance(result, tuple) and len(result) >= 2:
                is_valid, error_msg = result[0], result[1]
                if not is_valid:
                    self.errors.append(error_msg)
        except Exception as e:
            self.errors.append(f"Validation error: {str(e)}")
        return self
    
    def is_valid(self) -> bool:
        """Check if all validations passed"""
        return len(self.errors) == 0
    
    def get_errors(self) -> List[str]:
        """Get all validation errors"""
        return self.errors.copy()
    
    def get_error_summary(self) -> str:
        """Get formatted error summary"""
        if not self.errors:
            return "All validations passed"
        return "Validation errors:\n" + "\n".join(f"- {error}" for error in self.errors)

def validate_research_input(company_name: str, industry: str, 
                          description: str = None) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation for research input
    
    Args:
        company_name (str): Company name
        industry (str): Industry selection
        description (str): Optional industry description
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    validator = InputValidator()
    
    validator.add_validation(validate_company_name, company_name)
    validator.add_validation(validate_industry, industry)
    
    if description:
        validator.add_validation(validate_industry_description, description)
    
    return validator.is_valid(), validator.get_errors()