"""
Market Analysis Agent for AI/ML use case generation and market trends analysis
"""

import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from agents.base_agent import BaseAgent, AgentValidationMixin
from tools.serper_search import serper_tool
from config.prompts import get_agent_prompt
from config.settings import USE_CASE_CATEGORIES
from utils.logger import get_logger

logger = get_logger(__name__)


class MarketAnalysisAgent(BaseAgent, AgentValidationMixin):
    """Agent responsible for market analysis and AI/ML use case generation"""

    def __init__(self):
        super().__init__(
            agent_name="Market Analysis Agent",
            agent_description="Analyzes market trends and generates AI/ML use cases"
        )
        self.search_tool = serper_tool
        self.use_case_categories = USE_CASE_CATEGORIES or [
            "Process Automation", "Customer Experience", "Predictive Maintenance",
            "Demand Forecasting", "NLP/Conversational AI", "Computer Vision",
            "Analytics & Insights", "Recommendation Systems"
        ]

    def validate_input(self, task_input: Dict[str, Any]) -> tuple:
        """
        Validate market analysis task input

        Args:
            task_input (Dict[str, Any]): Input parameters

        Returns:
            tuple: (is_valid, error_messages)
        """
        required_fields = ["research_findings"]
        is_valid, errors = self.validate_required_fields(task_input, required_fields)

        if not is_valid:
            return False, errors

        # Validate research findings structure
        research_findings = task_input["research_findings"]
        if not isinstance(research_findings, dict):
            errors.append("research_findings must be a dictionary")

        # Check for required fields in research findings
        required_research_fields = ["target", "findings"]
        for field in required_research_fields:
            if field not in research_findings:
                errors.append(f"Missing required field in research_findings: {field}")

        return len(errors) == 0, errors

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute market analysis and use case generation

        Args:
            task_input (Dict[str, Any]): Analysis parameters

        Returns:
            Dict[str, Any]: Analysis results
        """
        research_findings = task_input["research_findings"]
        target = research_findings.get("target", "Unknown")
        industry = task_input.get("industry", research_findings.get("findings", {}).get("industry_sector", "Technology"))
        depth = task_input.get("depth", "standard")

        logger.info(f"Starting market analysis for: {target}")

        results = {
            "target": target,
            "industry": industry,
            "market_trends": {},
            "ai_adoption_analysis": {},
            "use_cases": [],
            "prioritization_matrix": {},
            "implementation_roadmap": {}
        }

        try:
            # Phase 1: Analyze market trends
            market_trends = self._analyze_market_trends(research_findings, industry)
            results["market_trends"] = market_trends

            # Phase 2: AI adoption analysis
            ai_adoption = self._analyze_ai_adoption(research_findings, industry)
            results["ai_adoption_analysis"] = ai_adoption

            # Phase 3: Generate use cases
            use_cases = self._generate_use_cases(research_findings, market_trends, ai_adoption)
            results["use_cases"] = use_cases

            # Phase 4: Create prioritization matrix
            prioritization = self._create_prioritization_matrix(use_cases)
            results["prioritization_matrix"] = prioritization

            # Phase 5: Generate implementation roadmap
            roadmap = self._generate_implementation_roadmap(use_cases, prioritization)
            results["implementation_roadmap"] = roadmap

            logger.info(f"Market analysis completed for: {target}")

        except Exception as e:
            logger.error(f"Market analysis execution failed: {str(e)}")
            raise

        return results

    # -------------------- Trend & Adoption Analysis --------------------

    def _analyze_market_trends(self, research_findings: Dict[str, Any], industry: str) -> Dict[str, Any]:
        """
        Analyze market trends relevant to AI adoption
        """
        target = research_findings.get("target", "Unknown")
        findings = research_findings.get("findings", {})

        # Search for industry-specific AI trends
        try:
            ai_trends_results = self.search_tool.search_ai_use_cases(industry)
            tech_trends_results = self.search_tool.search_technology_trends(industry)
        except Exception as e:
            logger.warning(f"Failed to fetch additional trend data: {str(e)}")
            ai_trends_results = {"results": []}
            tech_trends_results = {"results": []}

        # Generate analysis prompt
        analysis_prompt = f"""
Analyze market trends and AI adoption opportunities for {target} in the {industry} industry.

Research Findings:
- Industry Sector: {findings.get('industry_sector', 'Unknown')}
- Market Size: {findings.get('market_size', 'Unknown')}
- Technology Adoption: {findings.get('technology_adoption', 'Unknown')}
- Main Challenges: {findings.get('main_challenges', [])}
- Growth Opportunities: {findings.get('growth_opportunities', [])}

Additional AI Trends Data:
{self._format_search_results(ai_trends_results.get('results', [])[:5])}

Technology Trends Data:
{self._format_search_results(tech_trends_results.get('results', [])[:5])}

Provide analysis on:
1. Current market trends affecting the industry
2. AI adoption maturity in the sector
3. Competitive landscape for AI implementation
4. Market drivers for digital transformation
5. Regulatory and compliance considerations
6. Technology readiness assessment

Format your response with clear sections and bullet points.
"""

        try:
            analysis_response = self.generate_response(analysis_prompt)

            return {
                "analysis_text": analysis_response,
                "trend_categories": self._extract_trend_categories(analysis_response),
                "ai_maturity_level": self._assess_ai_maturity(analysis_response),
                "market_drivers": self._extract_market_drivers(analysis_response),
                "competitive_pressure": self._assess_competitive_pressure(analysis_response)
            }

        except Exception as e:
            logger.error(f"Market trends analysis failed: {str(e)}")
            return {
                "analysis_text": "Analysis could not be generated",
                "trend_categories": [],
                "ai_maturity_level": "Unknown",
                "market_drivers": [],
                "competitive_pressure": "Medium"
            }

    def _analyze_ai_adoption(self, research_findings: Dict[str, Any], industry: str) -> Dict[str, Any]:
        """
        Analyze AI adoption patterns and opportunities
        """
        findings = research_findings.get("findings", {})

        adoption_prompt = f"""
Analyze AI adoption opportunities for the {industry} industry based on research findings.

Current State:
- AI Readiness: {findings.get('ai_readiness', 'Unknown')}
- Digital Maturity: {findings.get('digital_maturity', 'Unknown')}
- Technology Adoption: {findings.get('technology_adoption', 'Unknown')}

Analyze:
1. Current AI adoption barriers and enablers
2. Most promising AI application areas
3. Required infrastructure and capabilities
4. Change management considerations
5. ROI potential and timeline expectations
6. Risk factors and mitigation strategies

Provide specific recommendations for AI adoption strategy.
"""

        try:
            adoption_response = self.generate_response(adoption_prompt)

            return {
                "adoption_readiness": self._assess_adoption_readiness(adoption_response),
                "key_barriers": self._extract_barriers(adoption_response),
                "enablers": self._extract_enablers(adoption_response),
                "recommended_focus_areas": self._extract_focus_areas(adoption_response),
                "timeline_assessment": self._assess_timeline(adoption_response),
                "risk_level": self._assess_risk_level(adoption_response)
            }

        except Exception as e:
            logger.error(f"AI adoption analysis failed: {str(e)}")
            return {
                "adoption_readiness": "Medium",
                "key_barriers": ["Limited data infrastructure", "Skill gaps", "Change resistance"],
                "enablers": ["Leadership support", "Digital transformation initiatives"],
                "recommended_focus_areas": ["Process automation", "Customer analytics"],
                "timeline_assessment": "6-12 months for initial implementations",
                "risk_level": "Medium"
            }

    # -------------------- Use Case Generation --------------------

    def _generate_use_cases(self, research_findings: Dict[str, Any],
                            market_trends: Dict[str, Any],
                            ai_adoption: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate AI/ML use cases based on analysis
        """
        target = research_findings.get("target", "Unknown")
        findings = research_findings.get("findings", {})

        use_case_prompt = f"""
Generate 7-10 specific AI/ML use cases for {target} based on comprehensive analysis.

Research Context:
- Industry: {findings.get('industry_sector', 'Unknown')}
- Business Model: {findings.get('business_model', 'Unknown')}
- Main Challenges: {findings.get('main_challenges', [])}
- Growth Opportunities: {findings.get('growth_opportunities', [])}

Market Context:
- AI Maturity Level: {market_trends.get('ai_maturity_level', 'Unknown')}
- Market Drivers: {market_trends.get('market_drivers', [])}

AI Adoption Context:
- Readiness Level: {ai_adoption.get('adoption_readiness', 'Unknown')}
- Focus Areas: {ai_adoption.get('recommended_focus_areas', [])}
- Key Barriers: {ai_adoption.get('key_barriers', [])}

For each use case, provide:
1. Use Case Title
2. Category (from: {', '.join(self.use_case_categories)})
3. Problem Statement (specific business problem)
4. Proposed AI/ML Solution (technical approach)
5. Expected Benefits (quantify where possible)
6. Implementation Complexity (Low/Medium/High)
7. Required Data Sources
8. Success Metrics (KPIs)
9. Implementation Timeline
10. Investment Level (Low/Medium/High)

Focus on practical, implementable solutions with clear business value.
Prioritize use cases that address the identified challenges and opportunities.

Format each use case as a JSON-like structure for easy parsing.
"""

        try:
            use_cases_response = self.generate_response(use_case_prompt)

            # Parse use cases from response
            use_cases = self._parse_use_cases(use_cases_response, target)

            # Enrich use cases with additional metadata
            enriched_use_cases = []
            for i, use_case in enumerate(use_cases):
                enriched_use_case = self._enrich_use_case(use_case, i + 1, research_findings)
                enriched_use_cases.append(enriched_use_case)

            return enriched_use_cases

        except Exception as e:
            logger.error(f"Use case generation failed: {str(e)}")
            return self._generate_fallback_use_cases(target, findings)

    # -------------------- Parsing Helpers --------------------

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format a few search results succinctly for prompts"""
        lines = []
        for r in results:
            title = r.get("title") or r.get("displayed_link") or r.get("link") or "result"
            snippet = r.get("snippet", "")
            lines.append(f"- {title}: {snippet[:200]}")
        return "\n".join(lines) if lines else "No external trend results available."

    def _extract_trend_categories(self, analysis_text: str) -> List[str]:
        """Lightweight extraction of trend categories from analysis text"""
        text = (analysis_text or "").lower()
        candidates = ["automation", "personalization", "predictive", "vision", "nlp", "cloud", "edge", "privacy"]
        found = [c for c in candidates if c in text]
        return found or ["automation", "personalization"]

    def _assess_ai_maturity(self, analysis_text: str) -> str:
        """Assess AI maturity level from text"""
        text = (analysis_text or "").lower()
        if any(k in text for k in ["mature", "advanced", "enterprise-wide"]):
            return "High"
        if any(k in text for k in ["pilot", "proof of concept", "early", "nascent"]):
            return "Low"
        return "Medium"

    def _extract_market_drivers(self, analysis_text: str) -> List[str]:
        """Extract market drivers"""
        text = (analysis_text or "").lower()
        drivers = []
        for kw in ["cost reduction", "customer demand", "efficiency", "regulation", "sustainability", "innovation"]:
            if kw in text:
                drivers.append(kw.title())
        return drivers or ["Efficiency", "Customer Demand"]

    def _assess_competitive_pressure(self, analysis_text: str) -> str:
        """Rough assessment of competitive pressure"""
        text = (analysis_text or "").lower()
        if "competitive" in text and "high" in text:
            return "High"
        if "low" in text or "limited competition" in text:
            return "Low"
        return "Medium"

    # -------------------- AI Adoption Heuristics --------------------

    def _assess_adoption_readiness(self, adoption_text: str) -> str:
        text = (adoption_text or "").lower()
        if any(x in text for x in ["strong leadership", "clear strategy", "mature stack"]):
            return "High"
        if any(x in text for x in ["pilot", "poC", "proof of concept", "in progress"]):
            return "Low"
        return "Medium"

    def _extract_barriers(self, adoption_text: str) -> List[str]:
        text = (adoption_text or "").lower()
        barriers = []
        for b in ["data", "skills", "integration", "cost", "privacy", "regulation"]:
            if b in text:
                barriers.append(b.title())
        return barriers or ["Data availability", "Skill gaps"]

    def _extract_enablers(self, adoption_text: str) -> List[str]:
        text = (adoption_text or "").lower()
        enablers = []
        for e in ["leadership", "funding", "cloud", "platform", "partnership"]:
            if e in text:
                enablers.append(e.title())
        return enablers or ["Leadership Support", "Cloud Infrastructure"]

    def _extract_focus_areas(self, adoption_text: str) -> List[str]:
        text = (adoption_text or "").lower()
        focus = []
        for f in ["automation", "customer", "forecast", "nlp", "vision", "analytics", "recommendation"]:
            if f in text:
                focus.append(f.title())
        return focus or ["Process Automation", "Customer Analytics"]

    def _assess_timeline(self, adoption_text: str) -> str:
        text = (adoption_text or "").lower()
        if "6 months" in text or "3-6 months" in text or "3 months" in text:
            return "3-6 months"
        if "12 months" in text or "1 year" in text:
            return "6-12 months"
        return "6-12 months"

    def _assess_risk_level(self, adoption_text: str) -> str:
        text = (adoption_text or "").lower()
        if "high risk" in text or "compliance" in text:
            return "High"
        if "low risk" in text:
            return "Low"
        return "Medium"

    # -------------------- Use-case enrichment & scoring (already present above) --------------------

    # _parse_use_cases, _extract_title, _extract_category, _extract_problem_statement,
    # _extract_solution, _extract_benefits, _extract_complexity, _extract_data_sources,
    # _extract_success_metrics, _extract_timeline, _extract_investment_level,
    # _enrich_use_case, _calculate_priority_score, _assess_feasibility, _assess_roi_potential,
    # _assess_strategic_alignment, _assess_use_case_risk, _identify_dependencies,
    # _estimate_effort, _create_prioritization_matrix, _generate_implementation_roadmap,
    # _estimate_resource_requirements, _estimate_budget_range, _generate_fallback_use_cases
    #
    # These methods were provided earlier in the class body; ensure they remain unchanged.
    # If your BaseAgent already contains parsing helpers or utilities, you can factor them out.

    # -------------------- Fallback use cases --------------------

    def _generate_fallback_use_cases(self, target: str, findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback use cases when LLM generation fails"""
        fallback_cases = [
            {
                "id": "uc_1",
                "title": "Process Automation with RPA",
                "category": "Process Automation",
                "problem_statement": "Manual processes causing inefficiencies",
                "solution": "Robotic Process Automation for routine tasks",
                "benefits": "30-50% reduction in processing time; reduced error rates",
                "complexity": "Medium",
                "priority": "High",
                "timeline": "3-6 months",
                "investment_level": "Medium",
                "data_sources": ["Operational Data", "Logs"],
                "success_metrics": ["Time Savings", "Error Reduction"]
            },
            {
                "id": "uc_2",
                "title": "Customer Analytics Dashboard",
                "category": "Customer Experience",
                "problem_statement": "Limited customer insights and segmentation",
                "solution": "ML-powered customer analytics and segmentation platform",
                "benefits": "Improved retention, targeted marketing, increased CLTV",
                "complexity": "Medium",
                "priority": "Medium",
                "timeline": "4-8 months",
                "investment_level": "Medium",
                "data_sources": ["Transaction Data", "CRM Data"],
                "success_metrics": ["Retention Rate", "Average Order Value"]
            },
            {
                "id": "uc_3",
                "title": "Demand Forecasting",
                "category": "Demand Forecasting",
                "problem_statement": "Poor inventory planning and stockouts",
                "solution": "Time-series forecasting models for demand prediction",
                "benefits": "Reduced stockouts and inventory costs",
                "complexity": "Medium",
                "priority": "High",
                "timeline": "3-6 months",
                "investment_level": "Medium",
                "data_sources": ["Sales History", "Promotions Calendar", "Seasonality Data"],
                "success_metrics": ["Forecast Accuracy", "Stockout Rate"]
            }
        ]

        # Enrich fallback cases minimally
        enriched = []
        for i, uc in enumerate(fallback_cases, 1):
            uc = self._enrich_use_case(uc, i, {"findings": findings})
            enriched.append(uc)
        return enriched


# Create a convenient global instance
market_analysis_agent = MarketAnalysisAgent()
