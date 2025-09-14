"""
Logging configuration for Multi-Agent Market Research System
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from config.settings import LOG_LEVEL, LOG_FILE, LOGS_DIR

class AgentLogger:
    """Custom logger for the multi-agent system"""
    
    def __init__(self, name: str = "MultiAgentSystem"):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logger with file and console handlers"""
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler
        if not LOGS_DIR.exists():
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        """Get the configured logger instance"""
        return self.logger

# Global logger instance
_logger_instance = None

def get_logger(name: str = "MultiAgentSystem") -> logging.Logger:
    """Get logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AgentLogger(name)
    return _logger_instance.get_logger()


# -------------------------
# Logging functions
# -------------------------

def log_agent_start(agent_name: str, task_description: str):
    logger = get_logger()
    logger.info(f"{agent_name} started: {task_description}")

def log_agent_complete(agent_name: str, duration: float, status: str = "SUCCESS"):
    logger = get_logger()
    logger.info(f"{agent_name} completed in {duration:.2f}s - {status}")

def log_search_query(query: str, source: str, results_count: int):
    logger = get_logger()
    logger.info(f"Search [{source}]: '{query}' - {results_count} results")

def log_api_call(api_name: str, endpoint: str, status_code: int):
    logger = get_logger()
    logger.debug(f"API [{api_name}]: {endpoint} - {status_code}")

def log_file_operation(operation: str, file_path: str, status: str = "SUCCESS"):
    logger = get_logger()
    logger.debug(f"File {operation}: {file_path} - {status}")

def log_error(error_msg: str, agent_name: str = None, exception: Exception = None):
    logger = get_logger()
    context = f"[{agent_name}] " if agent_name else ""
    if exception:
        logger.error(f"{context}{error_msg}: {str(exception)}", exc_info=True)
    else:
        logger.error(f"{context}{error_msg}")

def log_warning(warning_msg: str, agent_name: str = None):
    logger = get_logger()
    context = f"[{agent_name}] " if agent_name else ""
    logger.warning(f"{context}{warning_msg}")

def log_system_info(info_msg: str):
    logger = get_logger()
    logger.info(f"System: {info_msg}")

def log_user_action(action: str, details: dict = None):
    """
    Log user actions for analytics / debugging.
    """
    logger = get_logger()
    if details:
        logger.info(f"[USER ACTION] {action} - {details}")
    else:
        logger.info(f"[USER ACTION] {action}")


# -------------------------
# Context manager for agent execution logging
# -------------------------

class AgentExecutionLogger:
    """Context manager for logging agent execution"""
    
    def __init__(self, agent_name: str, task_description: str):
        self.agent_name = agent_name
        self.task_description = task_description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        log_agent_start(self.agent_name, self.task_description)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            status = "ERROR" if exc_type else "SUCCESS"
            log_agent_complete(self.agent_name, duration, status)
        
        if exc_type:
            log_error(f"Agent execution failed: {exc_val}", self.agent_name, exc_val)
        
        return False  # Don't suppress exceptions
