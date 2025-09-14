"""
Resource Agent for Multi-Agent Market Research System
Responsible for discovering relevant datasets, GitHub repos,
and open resources for the selected use cases and industry.
"""

from typing import Dict, Any, List
from config.settings import DATASETS_DIR, DATASET_SOURCES, MAX_SEARCH_RESULTS
from utils.logger import get_logger, log_error
from utils.file_manager import FileManager
from utils.validation import InputValidator
from base_agent import BaseAgent, AgentValidationMixin

import requests

logger = get_logger(__name__)


class ResourceAgent(BaseAgent, AgentValidationMixin):
    """Agent responsible for finding and managing resource links"""

    def __init__(self):
        super().__init__(
            agent_name="ResourceAgent",
            agent_description="Finds and aggregates relevant datasets, code repositories, and resources."
        )
        self.file_manager = FileManager()

    def validate_input(self, task_input: Dict[str, Any]) -> tuple:
        """
        Validate input parameters for resource search.
        """
        required_fields = ["industry", "use_cases"]
        is_valid, errors = self.validate_required_fields(task_input, required_fields)

        if not is_valid:
            return False, errors

        if not isinstance(task_input["use_cases"], list) or len(task_input["use_cases"]) == 0:
            errors.append("use_cases must be a non-empty list")

        return len(errors) == 0, errors

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute resource discovery task.
        """
        try:
            industry = task_input["industry"]
            use_cases = task_input["use_cases"]

            logger.info(f"🔍 Searching resources for industry '{industry}' and {len(use_cases)} use cases...")

            resources = {
                "datasets": self._search_kaggle_datasets(industry, use_cases),
                "huggingface": self._search_huggingface_datasets(industry, use_cases),
                "github_repos": self._search_github_repos(industry, use_cases)
            }

            saved = self.file_manager.save_json(
                resources,
                DATASETS_DIR / f"resources_{industry.lower().replace(' ', '_')}.json",
                add_timestamp=True
            )

            return {
                "resources": resources,
                "saved_to_file": saved
            }

        except Exception as e:
            log_error(f"Resource discovery failed", self.agent_name, e)
            return {"resources": {}, "saved_to_file": False}

    def _search_kaggle_datasets(self, industry: str, use_cases: List[str]) -> List[Dict[str, Any]]:
        """
        Perform a Kaggle dataset search.
        (Currently a placeholder that returns mock results.)
        """
        results = []
        try:
            search_query = f"{industry} {' '.join(use_cases)}"
            results.append({
                "platform": "Kaggle",
                "title": f"{industry} Market Dataset",
                "url": f"{DATASET_SOURCES['kaggle']['base_url']}/search?q={search_query}",
                "description": f"Search results for {search_query} on Kaggle"
            })
        except Exception as e:
            logger.warning(f"Kaggle search failed: {str(e)}")
        return results

    def _search_huggingface_datasets(self, industry: str, use_cases: List[str]) -> List[Dict[str, Any]]:
        """
        Perform a HuggingFace dataset search.
        (Currently a placeholder that returns mock results.)
        """
        results = []
        try:
            search_query = f"{industry} {' '.join(use_cases)}"
            results.append({
                "platform": "HuggingFace",
                "title": f"{industry} HF Datasets",
                "url": f"{DATASET_SOURCES['huggingface']['base_url']}?search={search_query}",
                "description": f"Search results for {search_query} on HuggingFace"
            })
        except Exception as e:
            logger.warning(f"HuggingFace search failed: {str(e)}")
        return results

    def _search_github_repos(self, industry: str, use_cases: List[str]) -> List[Dict[str, Any]]:
        """
        Perform a GitHub repo search.
        (Currently a placeholder that returns mock results.)
        """
        results = []
        try:
            search_query = f"{industry} {' '.join(use_cases)} AI ML"
            results.append({
                "platform": "GitHub",
                "title": f"{industry} AI/ML Repositories",
                "url": f"{DATASET_SOURCES['github']['base_url']}/search?q={search_query}&type=repositories",
                "description": f"Search results for {search_query} on GitHub"
            })
        except Exception as e:
            logger.warning(f"GitHub search failed: {str(e)}")
        return results


# Global instance for use by orchestrator
resource_agent = ResourceAgent()
