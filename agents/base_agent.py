"""
Base agent class for Multi-Agent Market Research System
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import google.generativeai as genai

from config.settings import get_model_config, GOOGLE_API_KEY
from utils.logger import get_logger, log_agent_start, log_agent_complete, log_error, AgentExecutionLogger
from utils.validation import InputValidator

logger = get_logger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_name: str, agent_description: str):
        """
        Initialize base agent
        
        Args:
            agent_name (str): Name of the agent
            agent_description (str): Description of agent's purpose
        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.execution_history = []
        self.current_task = None
        
        # Initialize Gemini model
        self.setup_llm()
    
    def setup_llm(self):
        """Setup Google Gemini LLM"""
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model_config = get_model_config()
            
            generation_config = {
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"],
                "top_k": model_config["top_k"],
                "max_output_tokens": model_config["max_tokens"]
            }
            
            self.model = genai.GenerativeModel(
                model_name=model_config["model"],
                generation_config=generation_config
            )
            
            logger.info(f"LLM initialized for {self.agent_name}")
            
        except Exception as e:
            log_error(f"Failed to initialize LLM for {self.agent_name}: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Generate response using Gemini LLM
        
        Args:
            prompt (str): Input prompt
            context (Dict[str, Any]): Additional context data
            
        Returns:
            str: Generated response
        """
        try:
            # Add context to prompt if provided
            if context:
                context_str = self._format_context(context)
                full_prompt = f"Context:\n{context_str}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            if response.text:
                return response.text.strip()
            else:
                raise Exception("Empty response from LLM")
                
        except Exception as e:
            log_error(f"LLM generation failed for {self.agent_name}: {str(e)}")
            raise
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context data for inclusion in prompt
        
        Args:
            context (Dict[str, Any]): Context data
            
        Returns:
            str: Formatted context string
        """
        formatted_lines = []
        
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                formatted_lines.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                formatted_lines.append(f"{key}: {value}")
        
        return "\n".join(formatted_lines)
    
    @abstractmethod
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent's main task
        
        Args:
            task_input (Dict[str, Any]): Input parameters for the task
            
        Returns:
            Dict[str, Any]: Task execution results
        """
        pass
    
    @abstractmethod
    def validate_input(self, task_input: Dict[str, Any]) -> tuple:
        """
        Validate input parameters
        
        Args:
            task_input (Dict[str, Any]): Input to validate
            
        Returns:
            tuple: (is_valid, error_messages)
        """
        pass
    
    def execute_with_logging(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task with comprehensive logging and error handling
        
        Args:
            task_input (Dict[str, Any]): Task input parameters
            
        Returns:
            Dict[str, Any]: Execution results with metadata
        """
        task_description = task_input.get('description', f"Execute {self.agent_name}")
        
        with AgentExecutionLogger(self.agent_name, task_description):
            try:
                # Validate input
                is_valid, errors = self.validate_input(task_input)
                if not is_valid:
                    return {
                        "success": False,
                        "errors": errors,
                        "agent": self.agent_name,
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Execute task
                self.current_task = task_input
                results = self.execute_task(task_input)
                
                # Add metadata
                results.update({
                    "success": True,
                    "agent": self.agent_name,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": self._calculate_execution_time()
                })
                
                # Log to history
                self._add_to_history(task_input, results)
                
                return results
                
            except Exception as e:
                log_error(f"Task execution failed for {self.agent_name}", self.agent_name, e)
                return {
                    "success": False,
                    "error": str(e),
                    "agent": self.agent_name,
                    "timestamp": datetime.now().isoformat()
                }
            finally:
                self.current_task = None
    
    def _calculate_execution_time(self) -> float:
        """Calculate execution time for current task"""
        if hasattr(self, '_start_time'):
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0
    
    def _add_to_history(self, task_input: Dict[str, Any], results: Dict[str, Any]):
        """Add execution to history"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_input": task_input,
            "results": results,
            "success": results.get("success", False)
        }
        
        self.execution_history.append(history_entry)
        
        # Keep only last 10 executions
        if len(self.execution_history) > 10:
            self.execution_history = self.execution_history[-10:]
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information and status
        
        Returns:
            Dict[str, Any]: Agent information
        """
        return {
            "name": self.agent_name,
            "description": self.agent_description,
            "status": "busy" if self.current_task else "idle",
            "execution_count": len(self.execution_history),
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None,
            "success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate from execution history"""
        if not self.execution_history:
            return 100.0
        
        successful = sum(1 for entry in self.execution_history if entry["success"])
        return (successful / len(self.execution_history)) * 100
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        logger.info(f"Cleared execution history for {self.agent_name}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of agent executions
        
        Returns:
            Dict[str, Any]: Execution summary
        """
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 100.0,
                "average_execution_time": 0.0
            }
        
        successful = [entry for entry in self.execution_history if entry["success"]]
        failed = [entry for entry in self.execution_history if not entry["success"]]
        
        # Calculate average execution time
        execution_times = [
            entry["results"].get("execution_time", 0) 
            for entry in self.execution_history 
            if entry["results"].get("execution_time")
        ]
        
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful),
            "failed_executions": len(failed),
            "success_rate": (len(successful) / len(self.execution_history)) * 100,
            "average_execution_time": avg_time
        }

class AgentValidationMixin:
    """Mixin class for common validation methods"""
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> tuple:
        """
        Validate that required fields are present and non-empty
        
        Args:
            data (Dict[str, Any]): Data to validate
            required_fields (List[str]): List of required field names
            
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not data[field] or (isinstance(data[field], str) and not data[field].strip()):
                errors.append(f"Empty required field: {field}")
        
        return len(errors) == 0, errors
    
    def validate_string_field(self, value: Any, field_name: str, 
                             min_length: int = 1, max_length: int = None) -> tuple:
        """
        Validate string field
        
        Args:
            value (Any): Value to validate
            field_name (str): Field name for error messages
            min_length (int): Minimum length
            max_length (int): Maximum length
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(value, str):
            return False, f"{field_name} must be a string"
        
        value = value.strip()
        
        if len(value) < min_length:
            return False, f"{field_name} must be at least {min_length} characters"
        
        if max_length and len(value) > max_length:
            return False, f"{field_name} must be no more than {max_length} characters"
        
        return True, ""