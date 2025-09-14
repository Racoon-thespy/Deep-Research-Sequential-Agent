"""
Configuration settings for Multi-Agent Market Research System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DATASETS_DIR = OUTPUTS_DIR / "datasets"
PROPOSALS_DIR = OUTPUTS_DIR / "proposals"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Create directories if they don't exist
for directory in [OUTPUTS_DIR, REPORTS_DIR, DATASETS_DIR, PROPOSALS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Application Settings
APP_NAME = os.getenv("APP_NAME", "Multi-Agent Market Research System")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / os.getenv("LOG_FILE", "outputs/logs/app.log")

# API Rate Limits (requests per minute)
RATE_LIMITS = {
    "serper": int(os.getenv("SERPER_RATE_LIMIT", 100)),
    "gemini": int(os.getenv("GEMINI_RATE_LIMIT", 60)),
    "kaggle": int(os.getenv("KAGGLE_RATE_LIMIT", 30)),
    "github": int(os.getenv("GITHUB_RATE_LIMIT", 5000))
}

# Search Configuration
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 10))
DEFAULT_SEARCH_TIMEOUT = int(os.getenv("DEFAULT_SEARCH_TIMEOUT", 30))

# Report Generation Settings
REPORT_FORMAT = os.getenv("REPORT_FORMAT", "markdown")
INCLUDE_TIMESTAMPS = os.getenv("INCLUDE_TIMESTAMPS", "True").lower() == "true"
AUTO_SAVE_REPORTS = os.getenv("AUTO_SAVE_REPORTS", "True").lower() == "true"

# Gemini Model Configuration
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_TOKENS = 8192
GEMINI_TOP_P = 0.8
GEMINI_TOP_K = 40

# Industry Categories
SUPPORTED_INDUSTRIES = [
    "Automotive",
    "Manufacturing",
    "Finance",
    "Banking",
    "Healthcare",
    "Retail",
    "E-commerce",
    "Technology",
    "Education",
    "Real Estate",
    "Energy",
    "Transportation",
    "Media & Entertainment",
    "Telecommunications",
    "Agriculture",
    "Construction",
    "Hospitality",
    "Legal Services",
    "Consulting",
    "Government"
]

# Use Case Categories
USE_CASE_CATEGORIES = [
    "Process Automation",
    "Customer Experience",
    "Predictive Analytics",
    "Natural Language Processing",
    "Computer Vision",
    "Recommendation Systems",
    "Fraud Detection",
    "Supply Chain Optimization",
    "Quality Assurance",
    "Risk Management",
    "Document Processing",
    "Chatbots & Virtual Assistants",
    "Personalization",
    "Inventory Management",
    "Maintenance Prediction",
    "Marketing Automation",
    "Sales Intelligence",
    "Financial Analytics",
    "Compliance Monitoring",
    "Cybersecurity"
]

# Dataset Sources
DATASET_SOURCES = {
    "kaggle": {
        "base_url": "https://www.kaggle.com/datasets",
        "api_url": "https://www.kaggle.com/api/v1",
        "search_endpoint": "/datasets/list"
    },
    "huggingface": {
        "base_url": "https://huggingface.co/datasets",
        "api_url": "https://huggingface.co/api",
        "search_endpoint": "/datasets"
    },
    "github": {
        "base_url": "https://github.com",
        "api_url": "https://api.github.com",
        "search_endpoint": "/search/repositories"
    }
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Validation Settings
MIN_COMPANY_NAME_LENGTH = 2
MAX_COMPANY_NAME_LENGTH = 100
MIN_INDUSTRY_DESCRIPTION_LENGTH = 10
MAX_INDUSTRY_DESCRIPTION_LENGTH = 1000

# Error Messages
ERROR_MESSAGES = {
    "missing_api_key": "Required API key is missing. Please check your .env file.",
    "invalid_company_name": f"Company name must be between {MIN_COMPANY_NAME_LENGTH} and {MAX_COMPANY_NAME_LENGTH} characters.",
    "invalid_industry": "Please select a valid industry from the supported list.",
    "search_failed": "Search operation failed. Please try again.",
    "file_not_found": "Required file not found. Please check file paths.",
    "rate_limit_exceeded": "API rate limit exceeded. Please wait before making more requests."
}

def validate_config():
    """
    Validate that all required configuration is present
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [GOOGLE_API_KEY, SERPER_API_KEY]
    missing_keys = [key for key in required_keys if not key]
    
    if missing_keys:
        print(f"Missing required API keys. Please check your .env file.")
        return False
    
    return True

def get_model_config():
    """
    Get Gemini model configuration
    
    Returns:
        dict: Model configuration parameters
    """
    return {
        "model": GEMINI_MODEL,
        "temperature": GEMINI_TEMPERATURE,
        "max_tokens": GEMINI_MAX_TOKENS,
        "top_p": GEMINI_TOP_P,
        "top_k": GEMINI_TOP_K
    }