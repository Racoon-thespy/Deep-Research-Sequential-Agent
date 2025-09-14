"""
Serper API integration for web search functionality
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from config.settings import SERPER_API_KEY, MAX_SEARCH_RESULTS, DEFAULT_SEARCH_TIMEOUT, RATE_LIMITS
from utils.logger import get_logger, log_search_query, log_api_call, log_error, log_warning
from utils.validation import validate_search_query, validate_api_key

logger = get_logger(__name__)

class SerperSearchTool:
    """Serper API search tool for web searches"""
    
    def __init__(self):
        self.api_key = SERPER_API_KEY
        self.base_url = "https://google.serper.dev/search"
        self.rate_limit = RATE_LIMITS.get("serper", 100)
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # 1 minute window
        
        # Validate API key
        is_valid, error_msg = validate_api_key(self.api_key, "Serper")
        if not is_valid:
            raise ValueError(f"Serper API configuration error: {error_msg}")
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_request_time > self.rate_limit_window:
            self.request_count = 0
        
        if self.request_count >= self.rate_limit:
            sleep_time = self.rate_limit_window - (current_time - self.last_request_time)
            if sleep_time > 0:
                log_warning(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
        
        self.request_count += 1
        self.last_request_time = current_time
    
    def search(self, query: str, num_results: int = None, 
               search_type: str = "search") -> Dict[str, Any]:
        """
        Perform web search using Serper API
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return
            search_type (str): Type of search (search, news, images)
            
        Returns:
            Dict[str, Any]: Search results
        """
        # Validate query
        is_valid, error_msg = validate_search_query(query)
        if not is_valid:
            raise ValueError(f"Invalid search query: {error_msg}")
        
        # Set default number of results
        if num_results is None:
            num_results = MAX_SEARCH_RESULTS
        
        # Check rate limit
        self._check_rate_limit()
        
        # Prepare request
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": min(num_results, 100)  # Serper max is 100
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=DEFAULT_SEARCH_TIMEOUT
            )
            
            log_api_call("Serper", self.base_url, response.status_code)
            
            if response.status_code == 200:
                results = response.json()
                results_count = len(results.get("organic", []))
                log_search_query(query, "serper", results_count)
                return self._process_results(results, query)
            
            elif response.status_code == 429:
                log_error("Serper API rate limit exceeded")
                raise Exception("Rate limit exceeded. Please try again later.")
            
            else:
                log_error(f"Serper API error: {response.status_code} - {response.text}")
                raise Exception(f"Search failed with status code: {response.status_code}")
                
        except requests.exceptions.Timeout:
            log_error("Serper API request timed out")
            raise Exception("Search request timed out")
        
        except requests.exceptions.RequestException as e:
            log_error(f"Serper API request failed: {str(e)}")
            raise Exception(f"Search request failed: {str(e)}")
    
    def _process_results(self, raw_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process and clean search results
        
        Args:
            raw_results (Dict[str, Any]): Raw API response
            query (str): Original search query
            
        Returns:
            Dict[str, Any]: Processed results
        """
        processed_results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "total_results": 0,
            "results": [],
            "related_searches": [],
            "answer_box": None
        }
        
        # Process organic results
        if "organic" in raw_results:
            for result in raw_results["organic"]:
                processed_result = {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "displayed_link": result.get("displayedLink", ""),
                    "date": result.get("date", ""),
                    "position": result.get("position", 0)
                }
                processed_results["results"].append(processed_result)
            
            processed_results["total_results"] = len(processed_results["results"])
        
        # Process answer box if available
        if "answerBox" in raw_results:
            answer_box = raw_results["answerBox"]
            processed_results["answer_box"] = {
                "title": answer_box.get("title", ""),
                "answer": answer_box.get("answer", ""),
                "link": answer_box.get("link", ""),
                "displayed_link": answer_box.get("displayedLink", "")
            }
        
        # Process related searches
        if "relatedSearches" in raw_results:
            processed_results["related_searches"] = [
                search.get("query", "") for search in raw_results["relatedSearches"]
            ]
        
        return processed_results
    
    def search_industry_reports(self, industry: str, company: str = None) -> Dict[str, Any]:
        """
        Search for industry-specific reports and insights
        
        Args:
            industry (str): Industry name
            company (str): Optional company name
            
        Returns:
            Dict[str, Any]: Search results focused on industry reports
        """
        # Build targeted query for industry reports
        query_parts = []
        
        if company:
            query_parts.append(f'"{company}"')
        
        query_parts.extend([
            f'"{industry} industry"',
            "market research OR industry report OR trends OR analysis",
            "McKinsey OR Deloitte OR PwC OR BCG OR Gartner"
        ])
        
        query = " ".join(query_parts)
        
        return self.search(query, num_results=15)
    
    def search_ai_use_cases(self, industry: str) -> Dict[str, Any]:
        """
        Search for AI/ML use cases in specific industry
        
        Args:
            industry (str): Industry name
            
        Returns:
            Dict[str, Any]: Search results focused on AI use cases
        """
        query = f'"{industry}" "artificial intelligence" OR "machine learning" OR "AI use cases" OR "ML applications" implementation examples'
        
        return self.search(query, num_results=12)
    
    def search_technology_trends(self, industry: str) -> Dict[str, Any]:
        """
        Search for technology trends in industry
        
        Args:
            industry (str): Industry name
            
        Returns:
            Dict[str, Any]: Search results focused on tech trends
        """
        query = f'"{industry}" "digital transformation" OR "technology trends" OR "automation" OR "GenAI" 2024 OR 2025'
        
        return self.search(query, num_results=10)
    
    def search_competitors(self, company: str, industry: str) -> Dict[str, Any]:
        """
        Search for company competitors and market analysis
        
        Args:
            company (str): Company name
            industry (str): Industry name
            
        Returns:
            Dict[str, Any]: Search results focused on competitors
        """
        query = f'"{company}" competitors "{industry}" market share analysis'
        
        return self.search(query, num_results=8)
    
    def multi_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Perform multiple searches with rate limiting
        
        Args:
            queries (List[str]): List of search queries
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        results = []
        
        for i, query in enumerate(queries):
            try:
                result = self.search(query)
                results.append(result)
                
                # Add delay between requests to respect rate limits
                if i < len(queries) - 1:
                    time.sleep(1)  # 1 second delay between requests
                    
            except Exception as e:
                log_error(f"Multi-search failed for query '{query}': {str(e)}")
                # Add empty result to maintain order
                results.append({
                    "query": query,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "results": []
                })
        
        return results
    
    def get_search_suggestions(self, base_query: str, industry: str) -> List[str]:
        """
        Generate search query suggestions for comprehensive research
        
        Args:
            base_query (str): Base search query (usually company name)
            industry (str): Industry context
            
        Returns:
            List[str]: List of suggested search queries
        """
        suggestions = [
            f'"{base_query}" company overview business model',
            f'"{base_query}" "{industry}" market analysis',
            f'"{base_query}" financial performance annual report',
            f'"{base_query}" technology stack digital transformation',
            f'"{base_query}" competitors competitive analysis',
            f'"{industry}" AI adoption trends use cases',
            f'"{industry}" automation machine learning applications',
            f'"{industry}" digital innovation case studies',
            f'"{base_query}" customer experience strategy',
            f'"{base_query}" operational efficiency challenges'
        ]
        
        return suggestions

# Global instance
serper_tool = SerperSearchTool()