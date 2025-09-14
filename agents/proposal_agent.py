"""
Proposal Agent for Multi-Agent Market Research System
Generates the final proposal document based on market research, use cases,
and collected resources.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any  # ✅ FIXED: Added Any
from google.generativeai import client as genai_client
from langchain.llms.base import LLM
from config.settings import PROPOSALS_DIR, INCLUDE_TIMESTAMPS, get_model_config
from utils.file_manager import FileManager
from utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLMWrapper(LLM):
    """
    Wrapper around google-generativeai to fit LangChain's LLM interface.
    """

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Call Gemini via google-generativeai.
        """
        try:
            resp = genai_client.generate_text(
                model=self.model,
                temperature=self.temperature,
                prompt=prompt
            )
            return resp.text
        except Exception as e:
            logger.error(f"Gemini generate_text call failed: {str(e)}")
            raise

    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}


class ProposalAgent:
    """
    Uses Gemini (via google-generativeai) to generate proposals.
    """

    def __init__(self):
        model_config = get_model_config()
        self.file_manager = FileManager()
        # Initialize Gemini wrapper
        self.llm = GeminiLLMWrapper(
            model=model_config["model"],
            temperature=model_config["temperature"]
        )
        logger.info(f"ProposalAgent initialized with Gemini model: {model_config['model']}")

    def _build_prompt(
        self,
        company_info: Dict,
        market_analysis: str,
        use_cases: List[Dict],
        datasets: Optional[List[Dict]] = None
    ) -> str:
        """
        Construct a detailed proposal prompt for Gemini.
        """
        prompt = f"""
You are an AI Market Research Assistant. Your task is to generate a professional,
client-facing proposal for the following company:

Company Information:
-------------------
{company_info}

Market Research & Analysis:
--------------------------
{market_analysis}

Proposed AI/ML/GenAI Use Cases:
------------------------------
{use_cases}

{"Relevant Datasets & Resources:\n" + str(datasets) if datasets else ""}

Please structure the proposal in the following format:
1. Executive Summary
2. Company Overview (1-2 paragraphs)
3. Industry Trends & Key Insights
4. Proposed AI/GenAI Use Cases (detailed with value proposition)
5. Implementation Feasibility (brief)
6. Resource Assets & Dataset References (clickable links if possible)
7. Conclusion & Next Steps

Make the proposal clear, concise, and professional.
Output should be in Markdown format for easy rendering.
"""
        return prompt

    def generate_proposal(
        self,
        company_info: Dict,
        market_analysis: str,
        use_cases: List[Dict],
        datasets: Optional[List[Dict]] = None,
        save: bool = True
    ) -> Optional[Path]:
        """
        Generate the final proposal document using Gemini.
        """
        try:
            logger.info("Generating proposal document with Gemini...")
            prompt = self._build_prompt(company_info, market_analysis, use_cases, datasets)
            proposal_content = self.llm(prompt) if isinstance(self.llm, LLM) else self.llm._call(prompt)

            if not proposal_content or not proposal_content.strip():
                logger.error("Gemini returned an empty response.")
                return None

            if save:
                filename = "final_proposal"
                proposal_path = PROPOSALS_DIR / f"{filename}.md"
                self.file_manager.save_markdown(
                    proposal_content,
                    proposal_path,
                    add_timestamp=INCLUDE_TIMESTAMPS
                )
                logger.info(f"Proposal saved to {proposal_path}")
                return proposal_path
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to generate proposal: {str(e)}", exc_info=True)
            return None


# Global instance
proposal_agent = ProposalAgent()
