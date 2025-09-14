"""
Main orchestration logic for Multi-Agent Market Research System
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from agents.research_agent import ResearchAgent
from tools.dataset_finder import dataset_finder
from config.settings import validate_config, REPORTS_DIR, DATASETS_DIR, PROPOSALS_DIR
from utils.logger import get_logger, log_system_info, log_error, log_file_operation
from utils.validation import validate_research_input, validate_system_requirements

logger = get_logger(__name__)

class MultiAgentOrchestrator:
    """Main orchestrator for the multi-agent system"""
    
    def __init__(self):
        """Initialize the orchestrator and all agents"""
        self.validate_system()
        self.initialize_agents()
        self.execution_history = []
    
    def validate_system(self):
        """Validate system requirements and configuration"""
        log_system_info("Validating system configuration...")
        
        # Validate configuration
        if not validate_config():
            raise ValueError("System configuration validation failed")
        
        # Validate system requirements
        is_valid, errors = validate_system_requirements()
        if not is_valid:
            error_msg = "System validation failed: " + "; ".join(errors)
            log_error(error_msg)
            raise ValueError(error_msg)
        
        log_system_info("System validation completed successfully")
    
    def initialize_agents(self):
        """Initialize all agents"""
        try:
            log_system_info("Initializing agents...")
            
            # Initialize Research Agent
            self.research_agent = ResearchAgent()
            
            # Initialize other agents (placeholders for now)
            # self.market_analysis_agent = MarketAnalysisAgent()
            # self.resource_agent = ResourceAgent()
            # self.proposal_agent = ProposalAgent()
            
            # Initialize dataset finder
            self.dataset_finder = dataset_finder
            
            log_system_info("All agents initialized successfully")
            
        except Exception as e:
            log_error(f"Agent initialization failed: {str(e)}")
            raise
    
    def execute_research_pipeline(self, company_name: str, industry: str = None, 
                                 description: str = None, depth: str = "standard") -> Dict[str, Any]:
        """
        Execute the complete research pipeline
        
        Args:
            company_name (str): Company name to research
            industry (str): Industry context (optional)
            description (str): Additional description (optional)
            depth (str): Research depth (quick, standard, deep)
            
        Returns:
            Dict[str, Any]: Complete research results
        """
        execution_id = f"exec_{int(time.time())}"
        
        log_system_info(f"Starting research pipeline for: {company_name}")
        
        # Validate inputs
        is_valid, errors = validate_research_input(company_name, industry or "Technology", description)
        if not is_valid:
            return {
                "success": False,
                "errors": errors,
                "execution_id": execution_id
            }
        
        results = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "company_name": company_name,
                "industry": industry,
                "description": description,
                "depth": depth
            },
            "stages": {},
            "success": False
        }
        
        try:
            # Stage 1: Research Agent
            log_system_info("Executing Stage 1: Industry & Company Research")
            research_task = {
                "company_or_industry": company_name,
                "research_type": "both" if industry else "company",
                "depth": depth,
                "description": f"Research {company_name} for AI/ML use case generation"
            }
            
            research_results = self.research_agent.execute_with_logging(research_task)
            results["stages"]["research"] = research_results
            
            if not research_results.get("success"):
                results["error"] = "Research stage failed"
                return results
            
            # Stage 2: Dataset Discovery (simplified for now)
            log_system_info("Executing Stage 2: Dataset Discovery")
            dataset_results = self._execute_dataset_discovery(company_name, industry)
            results["stages"]["datasets"] = dataset_results
            
            # Stage 3: Generate Use Cases (placeholder)
            log_system_info("Executing Stage 3: Use Case Generation")
            use_case_results = self._generate_use_cases_placeholder(research_results, dataset_results)
            results["stages"]["use_cases"] = use_case_results
            
            # Stage 4: Generate Final Report
            log_system_info("Executing Stage 4: Report Generation")
            report_results = self._generate_final_report(results)
            results["stages"]["report"] = report_results
            
            results["success"] = True
            
            # Save execution history
            self._save_execution_history(results)
            
            log_system_info(f"Research pipeline completed successfully: {execution_id}")
            
        except Exception as e:
            log_error(f"Pipeline execution failed: {str(e)}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def _execute_dataset_discovery(self, company_name: str, industry: str = None) -> Dict[str, Any]:
        """
        Execute dataset discovery for the company/industry
        
        Args:
            company_name (str): Company name
            industry (str): Industry context
            
        Returns:
            Dict[str, Any]: Dataset discovery results
        """
        try:
            # Generate search queries for datasets
            search_queries = []
            
            if industry:
                search_queries.append(f"{industry} dataset")
                search_queries.append(f"{industry} machine learning")
            
            search_queries.extend([
                f"{company_name} data",
                "business intelligence dataset",
                "customer analytics data"
            ])
            
            # Search across all platforms
            all_datasets = {}
            
            for query in search_queries[:3]:  # Limit to 3 queries to avoid rate limits
                try:
                    datasets = self.dataset_finder.search_all_platforms(query, max_results_per_platform=3)
                    
                    # Merge results
                    for platform, dataset_list in datasets.items():
                        if platform not in all_datasets:
                            all_datasets[platform] = []
                        all_datasets[platform].extend(dataset_list)
                
                except Exception as e:
                    log_error(f"Dataset search failed for query '{query}': {str(e)}")
                    continue
                
                # Add delay between searches
                time.sleep(2)
            
            # Remove duplicates and limit results
            for platform in all_datasets:
                seen_urls = set()
                unique_datasets = []
                for dataset in all_datasets[platform]:
                    url = dataset.get("url", "")
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_datasets.append(dataset)
                all_datasets[platform] = unique_datasets[:5]  # Top 5 per platform
            
            return {
                "success": True,
                "datasets": all_datasets,
                "total_found": sum(len(datasets) for datasets in all_datasets.values()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"Dataset discovery failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "datasets": {},
                "total_found": 0
            }
    
    def _generate_use_cases_placeholder(self, research_results: Dict[str, Any], 
                                       dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI/ML use cases based on research (placeholder implementation)
        
        Args:
            research_results (Dict[str, Any]): Research stage results
            dataset_results (Dict[str, Any]): Dataset discovery results
            
        Returns:
            Dict[str, Any]: Generated use cases
        """
        try:
            # Extract key information from research
            findings = research_results.get("findings", {})
            target = research_results.get("target", "Unknown")
            
            # Generate use cases using research agent's LLM
            use_case_prompt = f"""
            Based on the research findings for {target}, generate 5-7 relevant AI/ML use cases.
            
            For each use case, provide:
            1. Title
            2. Problem Statement
            3. Proposed AI/ML Solution
            4. Expected Benefits
            5. Implementation Complexity (Low/Medium/High)
            6. Required Data Types
            7. Success Metrics
            
            Focus on practical, implementable solutions that align with the company's industry and current capabilities.
            
            Format as structured JSON for each use case.
            """
            
            context = {
                "research_findings": findings,
                "available_datasets": dataset_results.get("datasets", {}),
                "target": target
            }
            
            use_cases_text = self.research_agent.generate_response(use_case_prompt, context)
            
            # Parse use cases (simplified - would need better parsing in production)
            use_cases = self._parse_use_cases_from_text(use_cases_text)
            
            return {
                "success": True,
                "use_cases": use_cases,
                "count": len(use_cases),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"Use case generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "use_cases": [],
                "count": 0
            }
    
    def _parse_use_cases_from_text(self, use_cases_text: str) -> List[Dict[str, Any]]:
        """
        Parse use cases from generated text (simplified implementation)
        
        Args:
            use_cases_text (str): Generated use cases text
            
        Returns:
            List[Dict[str, Any]]: Parsed use cases
        """
        # This is a simplified parser, will later add a more robust parser
        use_cases = []
        
        # Split by common delimiters and create basic use cases
        sections = use_cases_text.split('\n\n')
        
        for i, section in enumerate(sections[:7], 1):  # Max 7 use cases
            if len(section.strip()) > 50:  # Only process substantial sections
                use_case = {
                    "id": f"uc_{i}",
                    "title": f"AI Use Case {i}",
                    "description": section.strip()[:300] + "...",  # First 300 chars
                    "complexity": "Medium",  # Default
                    "category": "Process Automation",  # Default
                    "priority": "High" if i <= 3 else "Medium",
                    "implementation_time": "3-6 months",
                    "roi_potential": "High" if i <= 2 else "Medium"
                }
                use_cases.append(use_case)
        
        return use_cases
    
    def _generate_final_report(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final comprehensive report
        
        Args:
            execution_results (Dict[str, Any]): Complete execution results
            
        Returns:
            Dict[str, Any]: Report generation results
        """
        try:
            # Generate report content
            report_content = self._create_report_content(execution_results)
            
            # Save report to file
            report_filename = f"market_research_report_{execution_results['execution_id']}.md"
            report_path = REPORTS_DIR / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            log_file_operation("write", str(report_path), "SUCCESS")
            
            # Generate dataset report
            dataset_report = self._create_dataset_report(execution_results)
            dataset_filename = f"datasets_report_{execution_results['execution_id']}.md"
            dataset_path = DATASETS_DIR / dataset_filename
            
            with open(dataset_path, 'w', encoding='utf-8') as f:
                f.write(dataset_report)
            
            log_file_operation("write", str(dataset_path), "SUCCESS")
            
            return {
                "success": True,
                "report_path": str(report_path),
                "dataset_report_path": str(dataset_path),
                "report_size": len(report_content),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"Report generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_report_content(self, execution_results: Dict[str, Any]) -> str:
        """Create markdown report content"""
        
        input_data = execution_results.get("input", {})
        stages = execution_results.get("stages", {})
        
        report_lines = [
            "# Multi-Agent Market Research Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Execution ID:** {execution_results.get('execution_id', 'N/A')}",
            f"**Company/Target:** {input_data.get('company_name', 'N/A')}",
            f"**Industry:** {input_data.get('industry', 'N/A')}",
            f"**Research Depth:** {input_data.get('depth', 'standard')}",
            "\n---\n"
        ]
        
        # Executive Summary
        report_lines.extend([
            "## Executive Summary",
            "\nThis report provides comprehensive market research and AI/ML use case recommendations ",
            f"for {input_data.get('company_name', 'the target organization')}. ",
            "The analysis covers industry trends, competitive positioning, technology adoption opportunities, ",
            "and specific AI implementation recommendations.\n"
        ])
        
        # Research Findings
        research_stage = stages.get("research", {})
        if research_stage.get("success"):
            report_lines.extend([
                "## Research Findings",
                "\n### Industry & Company Analysis"
            ])
            
            findings = research_stage.get("findings", {})
            
            # Add comprehensive analysis if available
            if "comprehensive_analysis" in research_stage:
                report_lines.extend([
                    "\n#### Comprehensive Analysis",
                    f"\n{research_stage['comprehensive_analysis']}\n"
                ])
            
            # Add recommendations if available
            if "recommendations" in research_stage:
                report_lines.extend([
                    "\n#### Key Recommendations",
                    ""
                ])
                for i, rec in enumerate(research_stage["recommendations"][:5], 1):
                    report_lines.append(f"{i}. {rec}")
                report_lines.append("")
        
        # Use Cases
        use_case_stage = stages.get("use_cases", {})
        if use_case_stage.get("success"):
            use_cases = use_case_stage.get("use_cases", [])
            report_lines.extend([
                "## AI/ML Use Case Recommendations",
                f"\nIdentified {len(use_cases)} potential AI/ML use cases:\n"
            ])
            
            for i, use_case in enumerate(use_cases, 1):
                report_lines.extend([
                    f"### {i}. {use_case.get('title', f'Use Case {i}')}",
                    f"\n**Category:** {use_case.get('category', 'N/A')}",
                    f"**Complexity:** {use_case.get('complexity', 'N/A')}",
                    f"**Priority:** {use_case.get('priority', 'N/A')}",
                    f"**Implementation Time:** {use_case.get('implementation_time', 'N/A')}",
                    f"**ROI Potential:** {use_case.get('roi_potential', 'N/A')}",
                    f"\n**Description:**",
                    f"{use_case.get('description', 'No description available')}\n"
                ])
        
        # Implementation Strategy
        report_lines.extend([
            "## Implementation Strategy",
            "\n### Phased Approach",
            "\n**Phase 1 (0-3 months):** Quick wins and foundational setup",
            "- Assess current data infrastructure",
            "- Identify pilot use cases",
            "- Build internal AI capabilities",
            "",
            "**Phase 2 (3-9 months):** Core implementations",
            "- Deploy priority use cases",
            "- Establish data pipelines",
            "- Train staff and stakeholders",
            "",
            "**Phase 3 (9+ months):** Scale and optimize",
            "- Expand successful implementations",
            "- Advanced AI capabilities",
            "- Continuous improvement and optimization\n"
        ])
        
        # Next Steps
        report_lines.extend([
            "## Next Steps",
            "\n1. **Technical Assessment:** Evaluate current technology stack and data readiness",
            "2. **Pilot Selection:** Choose 1-2 high-impact, low-complexity use cases for initial implementation",
            "3. **Team Building:** Identify internal champions and required external expertise",
            "4. **Budget Planning:** Develop detailed budget estimates for priority initiatives",
            "5. **Vendor Evaluation:** Assess potential technology partners and solution providers",
            "\n---",
            "\n*This report was generated by the Multi-Agent Market Research System*",
            f"*For questions or additional analysis, please refer to execution ID: {execution_results.get('execution_id')}*"
        ])
        
        return "\n".join(report_lines)
    
    def _create_dataset_report(self, execution_results: Dict[str, Any]) -> str:
        """Create dataset discovery report"""
        
        dataset_stage = execution_results.get("stages", {}).get("datasets", {})
        datasets = dataset_stage.get("datasets", {})
        
        if not datasets:
            return "# Dataset Discovery Report\n\nNo datasets found during the research process."
        
        # Use the dataset finder's report generation method
        use_case_info = execution_results.get("input", {}).get("company_name", "")
        industry_info = execution_results.get("input", {}).get("industry", "")
        
        return self.dataset_finder.generate_dataset_report(
            datasets, use_case_info, industry_info
        )
    
    def _save_execution_history(self, results: Dict[str, Any]):
        """Save execution history"""
        self.execution_history.append(results)
        
        # Keep only last 50 executions
        if len(self.execution_history) > 50:
            self.execution_history = self.execution_history[-50:]
        
        # Save to file
        try:
            history_file = REPORTS_DIR / "execution_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.execution_history, f, indent=2, default=str)
        except Exception as e:
            log_error(f"Failed to save execution history: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and metrics"""
        return {
            "system_status": "operational",
            "agents_initialized": True,
            "total_executions": len(self.execution_history),
            "successful_executions": sum(1 for ex in self.execution_history if ex.get("success")),
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None,
            "research_agent_status": self.research_agent.get_agent_info(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:] if self.execution_history else []

# Global orchestrator instance
orchestrator = None

def get_orchestrator() -> MultiAgentOrchestrator:
    """Get or create global orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = MultiAgentOrchestrator()
    return orchestrator

def main():
    """Main function for command-line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Market Research System")
    parser.add_argument("--company", required=True, help="Company name to research")
    parser.add_argument("--industry", help="Industry context")
    parser.add_argument("--depth", choices=["quick", "standard", "deep"], 
                       default="standard", help="Research depth")
    parser.add_argument("--description", help="Additional description")
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orch = get_orchestrator()
        
        # Execute research
        results = orch.execute_research_pipeline(
            company_name=args.company,
            industry=args.industry,
            description=args.description,
            depth=args.depth
        )
        
        # Print results
        if results.get("success"):
            print(f" Research completed successfully!")
            print(f"Execution ID: {results['execution_id']}")
            
            if "report" in results["stages"]:
                report_info = results["stages"]["report"]
                if report_info.get("success"):
                    print(f" Report saved to: {report_info['report_path']}")
                    print(f" Dataset report saved to: {report_info['dataset_report_path']}")
        else:
            print(f" Research failed: {results.get('error', 'Unknown error')}")
            if "errors" in results:
                for error in results["errors"]:
                    print(f"  - {error}")
    
    except Exception as e:
        print(f" System error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())