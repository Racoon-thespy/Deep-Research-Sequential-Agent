"""
Research Agent for industry and company analysis
"""

from typing import Dict, Any, List
from datetime import datetime

from agents.base_agent import BaseAgent, AgentValidationMixin
from tools.serper_search import serper_tool
from config.prompts import get_agent_prompt
from config.settings import SUPPORTED_INDUSTRIES
from utils.logger import get_logger, log_search_query, log_error
from utils.validation import validate_company_name, validate_industry

logger = get_logger(__name__)

class ResearchAgent(BaseAgent, AgentValidationMixin):
    """Agent responsible for industry and company research"""
    
    def __init__(self):
        super().__init__(
            agent_name="Research Agent",
            agent_description="Conducts comprehensive industry and company research using web search"
        )
        self.search_tool = serper_tool
    
    def validate_input(self, task_input: Dict[str, Any]) -> tuple:
        """
        Validate research task input
        
        Args:
            task_input (Dict[str, Any]): Input to validate
            
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        required_fields = ["company_or_industry", "research_type"]
        is_valid, field_errors = self.validate_required_fields(task_input, required_fields)
        errors.extend(field_errors)
        
        # Validate research type
        valid_types = ["company", "industry", "both"]
        research_type = task_input.get("research_type", "").lower()
        if research_type not in valid_types:
            errors.append(f"Research type must be one of: {valid_types}")
        
        # Validate company/industry name
        target = task_input.get("company_or_industry", "")
        if target:
            # For company research, validate company name
            if research_type in ["company", "both"]:
                is_valid_name, name_error = validate_company_name(target)
                if not is_valid_name:
                    errors.append(f"Company name validation: {name_error}")
            
            # For industry research, check if it's a supported industry
            if research_type == "industry":
                if target not in SUPPORTED_INDUSTRIES:
                    errors.append(f"Industry must be one of: {', '.join(SUPPORTED_INDUSTRIES)}")
        
        return len(errors) == 0, errors
    
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research task
        
        Args:
            task_input (Dict[str, Any]): Task parameters
            
        Returns:
            Dict[str, Any]: Research results
        """
        company_or_industry = task_input["company_or_industry"]
        research_type = task_input["research_type"].lower()
        depth = task_input.get("depth", "standard")  # standard, deep, quick
        
        results = {
            "target": company_or_industry,
            "research_type": research_type,
            "depth": depth,
            "findings": {},
            "sources": [],
            "recommendations": []
        }
        
        try:
            if research_type in ["company", "both"]:
                company_findings = self._conduct_company_research(company_or_industry, depth)
                results["findings"]["company_analysis"] = company_findings
            
            if research_type in ["industry", "both"]:
                # For company research, try to identify the industry first
                if research_type == "both":
                    industry = self._identify_company_industry(company_or_industry)
                else:
                    industry = company_or_industry
                
                industry_findings = self._conduct_industry_research(industry, depth)
                results["findings"]["industry_analysis"] = industry_findings
            
            # Generate comprehensive analysis
            results["comprehensive_analysis"] = self._generate_comprehensive_analysis(
                results["findings"], company_or_industry, research_type
            )
            
            # Extract sources from all searches
            results["sources"] = self._extract_all_sources()
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(
                results["findings"], research_type
            )
            
            return results
            
        except Exception as e:
            log_error(f"Research execution failed: {str(e)}", self.agent_name, e)
            raise
    
    def _conduct_company_research(self, company_name: str, depth: str) -> Dict[str, Any]:
        """
        Conduct comprehensive company research
        
        Args:
            company_name (str): Name of the company
            depth (str): Research depth level
            
        Returns:
            Dict[str, Any]: Company research findings
        """
        findings = {
            "company_overview": {},
            "business_model": {},
            "market_position": {},
            "technology_adoption": {},
            "financial_performance": {},
            "recent_developments": []
        }
        
        try:
            # Company overview search
            overview_results = self.search_tool.search(
                f'"{company_name}" company overview business model products services',
                num_results=8
            )
            findings["company_overview"] = self._extract_company_overview(overview_results)
            
            # Market position and competitors
            market_results = self.search_tool.search(
                f'"{company_name}" market position competitors competitive analysis',
                num_results=6
            )
            findings["market_position"] = self._extract_market_position(market_results)
            
            # Technology and digital transformation
            tech_results = self.search_tool.search(
                f'"{company_name}" technology stack digital transformation AI automation',
                num_results=5
            )
            findings["technology_adoption"] = self._extract_technology_info(tech_results)
            
            # Financial performance (if depth is deep)
            if depth == "deep":
                financial_results = self.search_tool.search(
                    f'"{company_name}" financial performance revenue annual report earnings',
                    num_results=4
                )
                findings["financial_performance"] = self._extract_financial_info(financial_results)
            
            # Recent news and developments
            news_results = self.search_tool.search(
                f'"{company_name}" news recent developments 2024 2025',
                num_results=5
            )
            findings["recent_developments"] = self._extract_recent_developments(news_results)
            
        except Exception as e:
            log_error(f"Company research failed for {company_name}: {str(e)}")
            raise
        
        return findings
    
    def _conduct_industry_research(self, industry: str, depth: str) -> Dict[str, Any]:
        """
        Conduct comprehensive industry research
        
        Args:
            industry (str): Industry name
            depth (str): Research depth level
            
        Returns:
            Dict[str, Any]: Industry research findings
        """
        findings = {
            "industry_overview": {},
            "market_trends": {},
            "key_players": [],
            "challenges_opportunities": {},
            "ai_adoption": {},
            "regulatory_environment": {}
        }
        
        try:
            # Industry overview and market size
            overview_results = self.search_tool.search_industry_reports(industry)
            findings["industry_overview"] = self._extract_industry_overview(overview_results)
            
            # Technology trends and AI adoption
            ai_results = self.search_tool.search_ai_use_cases(industry)
            findings["ai_adoption"] = self._extract_ai_adoption_info(ai_results)
            
            # Market trends
            trends_results = self.search_tool.search_technology_trends(industry)
            findings["market_trends"] = self._extract_market_trends(trends_results)
            
            # Key players and competitors
            if depth in ["standard", "deep"]:
                players_results = self.search_tool.search(
                    f'"{industry}" key players leading companies market leaders',
                    num_results=6
                )
                findings["key_players"] = self._extract_key_players(players_results)
            
            # Challenges and opportunities
            challenges_results = self.search_tool.search(
                f'"{industry}" challenges opportunities trends future outlook',
                num_results=5
            )
            findings["challenges_opportunities"] = self._extract_challenges_opportunities(challenges_results)
            
            # Regulatory environment (if deep research)
            if depth == "deep":
                regulatory_results = self.search_tool.search(
                    f'"{industry}" regulations compliance requirements standards',
                    num_results=4
                )
                findings["regulatory_environment"] = self._extract_regulatory_info(regulatory_results)
            
        except Exception as e:
            log_error(f"Industry research failed for {industry}: {str(e)}")
            raise
        
        return findings
    
    def _identify_company_industry(self, company_name: str) -> str:
        """
        Identify the industry of a company
        
        Args:
            company_name (str): Company name
            
        Returns:
            str: Identified industry
        """
        try:
            search_results = self.search_tool.search(
                f'"{company_name}" industry sector business type',
                num_results=3
            )
            
            # Use LLM to identify industry from search results
            context = {"search_results": search_results}
            prompt = f"""
            Based on the search results, identify which industry "{company_name}" belongs to.
            
            Choose from these supported industries: {', '.join(SUPPORTED_INDUSTRIES)}
            
            Return only the industry name that best matches, or "Technology" if unclear.
            """
            
            industry = self.generate_response(prompt, context).strip()
            
            # Validate the response is in our supported industries
            if industry in SUPPORTED_INDUSTRIES:
                return industry
            else:
                # Try to find a close match
                for supported in SUPPORTED_INDUSTRIES:
                    if supported.lower() in industry.lower() or industry.lower() in supported.lower():
                        return supported
                
                # Default fallback
                return "Technology"
                
        except Exception as e:
            log_error(f"Failed to identify industry for {company_name}: {str(e)}")
            return "Technology"  # Safe fallback
    
    def _extract_company_overview(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract company overview from search results"""
        try:
            prompt = get_agent_prompt("research", company_industry="Company Analysis")
            context = {"search_results": search_results}
            
            overview_prompt = f"""
            {prompt}
            
            Based on the search results, extract the following company information:
            1. Business model and core offerings
            2. Target markets and customer base
            3. Company size and geographic presence
            4. Mission, vision, and strategic focus
            5. Key differentiators and competitive advantages
            
            Format as structured data with clear sections.
            """
            
            response = self.generate_response(overview_prompt, context)
            
            return {
                "summary": response,
                "data_quality": self._assess_data_quality(search_results),
                "sources_count": len(search_results.get("results", []))
            }
            
        except Exception as e:
            log_error(f"Failed to extract company overview: {str(e)}")
            return {"summary": "Analysis failed", "error": str(e)}
    
    def _extract_market_position(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market position analysis from search results"""
        try:
            prompt = """
            Analyze the market position based on the search results:
            1. Market share and ranking
            2. Key competitors and competitive landscape
            3. Competitive advantages and differentiators
            4. Market positioning strategy
            5. Strengths and weaknesses relative to competitors
            
            Provide specific insights and data points where available.
            """
            
            context = {"search_results": search_results}
            response = self.generate_response(prompt, context)
            
            return {
                "analysis": response,
                "data_quality": self._assess_data_quality(search_results),
                "sources_count": len(search_results.get("results", []))
            }
            
        except Exception as e:
            log_error(f"Failed to extract market position: {str(e)}")
            return {"analysis": "Analysis failed", "error": str(e)}
    
    def _extract_technology_info(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technology adoption information from search results"""
        try:
            prompt = """
            Analyze technology adoption and digital transformation:
            1. Current technology stack and platforms
            2. Digital transformation initiatives
            3. AI and automation adoption
            4. Innovation investments and R&D
            5. Technology partnerships and integrations
            
            Focus on identifying opportunities for AI/ML implementation.
            """
            
            context = {"search_results": search_results}
            response = self.generate_response(prompt, context)
            
            return {
                "analysis": response,
                "ai_readiness": self._assess_ai_readiness(search_results),
                "data_quality": self._assess_data_quality(search_results)
            }
            
        except Exception as e:
            log_error(f"Failed to extract technology info: {str(e)}")
            return {"analysis": "Analysis failed", "error": str(e)}
    
    def _extract_financial_info(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial performance information"""
        try:
            prompt = """
            Extract financial performance insights:
            1. Revenue and growth trends
            2. Profitability and margins
            3. Investment capacity for new initiatives
            4. Financial stability and resources
            5. Technology investment budget indicators
            
            Focus on indicators relevant to AI/digital transformation investments.
            """
            
            context = {"search_results": search_results}
            response = self.generate_response(prompt, context)
            
            return {
                "analysis": response,
                "investment_capacity": "To be assessed",
                "data_quality": self._assess_data_quality(search_results)
            }
            
        except Exception as e:
            log_error(f"Failed to extract financial info: {str(e)}")
            return {"analysis": "Analysis failed", "error": str(e)}
    
    def _extract_recent_developments(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract recent news and developments"""
        try:
            developments = []
            
            for result in search_results.get("results", [])[:5]:
                development = {
                    "title": result.get("title", ""),
                    "summary": result.get("snippet", ""),
                    "url": result.get("link", ""),
                    "date": result.get("date", ""),
                    "relevance": "high" if any(keyword in result.get("title", "").lower() 
                                            for keyword in ["ai", "technology", "digital", "innovation"]) else "medium"
                }
                developments.append(development)
            
            return developments
            
        except Exception as e:
            log_error(f"Failed to extract recent developments: {str(e)}")
            return []
    
    def _extract_industry_overview(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract industry overview from search results"""
        try:
            prompt = """
            Analyze the industry based on search results:
            1. Industry size and market value
            2. Growth trends and projections
            3. Market segments and subsectors
            4. Geographic distribution and key markets
            5. Industry lifecycle stage and maturity
            
            Provide quantitative data where available.
            """
            
            context = {"search_results": search_results}
            response = self.generate_response(prompt, context)
            
            return {
                "analysis": response,
                "data_quality": self._assess_data_quality(search_results),
                "sources_count": len(search_results.get("results", []))
            }
            
        except Exception as e:
            log_error(f"Failed to extract industry overview: {str(e)}")
            return {"analysis": "Analysis failed", "error": str(e)}
    
    def _extract_ai_adoption_info(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract AI adoption information from search results"""
        try:
            prompt = """
            Analyze AI and ML adoption in this industry:
            1. Current AI use cases and applications
            2. Adoption rate and maturity level
            3. Leading companies in AI implementation
            4. Common AI technologies being used
            5. Barriers and challenges to AI adoption
            6. Future AI trends and opportunities
            
            Focus on practical AI applications and implementation patterns.
            """
            
            context = {"search_results": search_results}
            response = self.generate_response(prompt, context)
            
            return {
                "analysis": response,
                "adoption_level": self._assess_ai_adoption_level(search_results),
                "data_quality": self._assess_data_quality(search_results)
            }
            
        except Exception as e:
            log_error(f"Failed to extract AI adoption info: {str(e)}")
            return {"analysis": "Analysis failed", "error": str(e)}
    
    def _assess_data_quality(self, search_results: Dict[str, Any]) -> str:
        """Assess the quality of search results data"""
        results = search_results.get("results", [])
        
        if not results:
            return "poor"
        
        # Check for authoritative sources
        authoritative_domains = [
            "mckinsey.com", "deloitte.com", "pwc.com", "bcg.com", 
            "gartner.com", "forbes.com", "reuters.com", "bloomberg.com"
        ]
        
        authoritative_count = sum(
            1 for result in results 
            if any(domain in result.get("link", "") for domain in authoritative_domains)
        )
        
        total_results = len(results)
        recent_results = sum(
            1 for result in results 
            if "2024" in result.get("date", "") or "2025" in result.get("date", "")
        )
        
        if authoritative_count >= 2 and recent_results >= 1:
            return "excellent"
        elif authoritative_count >= 1 or recent_results >= 2:
            return "good"
        elif total_results >= 3:
            return "fair"
        else:
            return "poor"
    
    def _assess_ai_readiness(self, search_results: Dict[str, Any]) -> str:
        """Assess AI readiness level based on search results"""
        results_text = " ".join([
            result.get("snippet", "") for result in search_results.get("results", [])
        ]).lower()
        
        ai_indicators = ["artificial intelligence", "machine learning", "automation", "digital transformation", "ai"]
        high_readiness_indicators = ["implementing ai", "ai strategy", "ai adoption", "machine learning models"]
        
        high_indicators = sum(1 for indicator in high_readiness_indicators if indicator in results_text)
        ai_mentions = sum(1 for indicator in ai_indicators if indicator in results_text)
        
        if high_indicators >= 2:
            return "high"
        elif ai_mentions >= 3:
            return "medium"
        elif ai_mentions >= 1:
            return "low"
        else:
            return "unknown"
    
    def _assess_ai_adoption_level(self, search_results: Dict[str, Any]) -> str:
        """Assess industry-wide AI adoption level"""
        results_text = " ".join([
            result.get("snippet", "") for result in search_results.get("results", [])
        ]).lower()
        
        adoption_indicators = {
            "high": ["widespread adoption", "mature implementation", "ai-first", "fully integrated"],
            "medium": ["growing adoption", "pilot programs", "early implementation", "selective use"],
            "low": ["emerging interest", "early stages", "limited adoption", "experimental"]
        }
        
        for level, indicators in adoption_indicators.items():
            if any(indicator in results_text for indicator in indicators):
                return level
        
        return "emerging"
    
    def _generate_comprehensive_analysis(self, findings: Dict[str, Any], 
                                       target: str, research_type: str) -> str:
        """Generate comprehensive analysis combining all findings"""
        try:
            prompt = f"""
            Generate a comprehensive analysis report based on the research findings for {target}.
            
            Research Type: {research_type}
            
            Key areas to cover:
            1. Executive Summary
            2. Key Insights and Strategic Implications
            3. AI/ML Opportunities Assessment
            4. Digital Transformation Readiness
            5. Competitive Position and Market Context
            6. Recommendations for AI Implementation
            
            Make the analysis actionable and focused on AI/ML adoption opportunities.
            """
            
            context = {"findings": findings}
            analysis = self.generate_response(prompt, context)
            
            return analysis
            
        except Exception as e:
            log_error(f"Failed to generate comprehensive analysis: {str(e)}")
            return "Comprehensive analysis generation failed"
    
    def _extract_all_sources(self) -> List[Dict[str, str]]:
        """Extract all sources used in research (placeholder for now)"""
        # This would collect all the URLs and sources from search results
        # For now, return empty list - would be implemented to track all sources
        return []
    
    def _generate_recommendations(self, findings: Dict[str, Any], research_type: str) -> List[str]:
        """Generate actionable recommendations based on findings"""
        try:
            prompt = f"""
            Based on the research findings, generate 5-7 specific, actionable recommendations for AI/ML implementation.
            
            Focus on:
            1. Quick wins and low-hanging fruit
            2. Strategic AI initiatives
            3. Technology infrastructure needs
            4. Skill development requirements
            5. Partnership and vendor considerations
            
            Make each recommendation specific and implementable.
            """
            
            context = {"findings": findings}
            recommendations_text = self.generate_response(prompt, context)
            
            # Parse recommendations into list
            recommendations = [
                rec.strip() for rec in recommendations_text.split('\n') 
                if rec.strip() and (rec.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '-', '•')) or len(rec.strip()) > 10)
            ]
            
            return recommendations[:7]  # Limit to 7 recommendations
            
        except Exception as e:
            log_error(f"Failed to generate recommendations: {str(e)}")
            return ["Failed to generate recommendations"]

# Additional extraction methods for industry research
    def _extract_market_trends(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market trends from search results"""
        try:
            prompt = """
            Analyze current and emerging market trends:
            1. Technology trends and innovations
            2. Consumer behavior changes
            3. Regulatory and compliance trends
            4. Competitive landscape evolution
            5. Future outlook and predictions
            
            Focus on trends relevant to AI/ML adoption.
            """
            
            context = {"search_results": search_results}
            response = self.generate_response(prompt, context)
            
            return {
                "analysis": response,
                "trend_strength": "emerging",  # Could be enhanced with sentiment analysis
                "data_quality": self._assess_data_quality(search_results)
            }
            
        except Exception as e:
            log_error(f"Failed to extract market trends: {str(e)}")
            return {"analysis": "Analysis failed", "error": str(e)}
    
    def _extract_key_players(self, search_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract key industry players from search results"""
        try:
            prompt = """
            Identify the key players in this industry:
            1. Market leaders and their market share
            2. Innovative companies to watch
            3. Technology pioneers and disruptors
            4. Key partnerships and alliances
            
            For each company, provide name, role, and key strengths.
            """
            
            context = {"search_results": search_results}
            response = self.generate_response(prompt, context)
            
            # This would ideally parse the response into structured data
            # For now, return as a single analysis
            return [{"analysis": response, "type": "comprehensive_list"}]
            
        except Exception as e:
            log_error(f"Failed to extract key players: {str(e)}")
            return []
    
    def _extract_challenges_opportunities(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract industry challenges and opportunities"""
        try:
            prompt = """
            Identify key challenges and opportunities:
            
            Challenges:
            1. Market challenges and barriers
            2. Technology limitations
            3. Regulatory and compliance issues
            4. Resource and skill constraints
            
            Opportunities:
            1. Growth opportunities and market gaps
            2. Technology advancement opportunities
            3. AI/ML implementation opportunities
            4. Partnership and collaboration opportunities
            
            Focus on aspects relevant to AI adoption.
            """
            
            context = {"search_results": search_results}
            response = self.generate_response(prompt, context)
            
            return {
                "analysis": response,
                "opportunity_assessment": "high",  # Could be enhanced
                "data_quality": self._assess_data_quality(search_results)
            }
            
        except Exception as e:
            log_error(f"Failed to extract challenges/opportunities: {str(e)}")
            return {"analysis": "Analysis failed", "error": str(e)}
    
    def _extract_regulatory_info(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract regulatory environment information"""
        try:
            prompt = """
            Analyze the regulatory environment:
            1. Key regulations and compliance requirements
            2. Data privacy and security regulations
            3. AI-specific regulations and guidelines
            4. Industry standards and certifications
            5. Upcoming regulatory changes
            
            Focus on regulations that impact AI/ML implementation.
            """
            
            context = {"search_results": search_results}
            response = self.generate_response(prompt, context)
            
            return {
                "analysis": response,
                "compliance_complexity": "medium",  # Could be assessed
                "data_quality": self._assess_data_quality(search_results)
            }
            
        except Exception as e:
            log_error(f"Failed to extract regulatory info: {str(e)}")
            return {"analysis": "Analysis failed", "error": str(e)}