"""
Dataset discovery tool for finding relevant datasets from Kaggle, HuggingFace, and GitHub
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from github import Github
from huggingface_hub import HfApi, list_datasets
from kaggle.api.kaggle_api_extended import KaggleApi

from config.settings import (
    KAGGLE_USERNAME, KAGGLE_KEY, HUGGINGFACE_TOKEN, 
    GITHUB_TOKEN, DATASET_SOURCES, RATE_LIMITS
)
from utils.logger import get_logger, log_search_query, log_api_call, log_error, log_warning
from utils.validation import validate_search_query, validate_api_key

logger = get_logger(__name__)

class DatasetFinder:
    """Dataset discovery tool for multiple platforms"""
    
    def __init__(self):
        self.setup_apis()
        self.rate_limits = RATE_LIMITS
        self.last_request_times = {}
    
    def setup_apis(self):
        """Setup API clients for different platforms"""
        # Kaggle API
        self.kaggle_api = None
        if KAGGLE_USERNAME and KAGGLE_KEY:
            try:
                self.kaggle_api = KaggleApi()
                self.kaggle_api.authenticate()
                logger.info("Kaggle API initialized successfully")
            except Exception as e:
                log_warning(f"Kaggle API initialization failed: {str(e)}")
        
        # HuggingFace API
        self.hf_api = None
        if HUGGINGFACE_TOKEN:
            try:
                self.hf_api = HfApi(token=HUGGINGFACE_TOKEN)
                logger.info("HuggingFace API initialized successfully")
            except Exception as e:
                log_warning(f"HuggingFace API initialization failed: {str(e)}")
        else:
            # Use public API without token
            self.hf_api = HfApi()
            logger.info("HuggingFace API initialized (public access)")
        
        # GitHub API
        self.github_api = None
        if GITHUB_TOKEN:
            try:
                self.github_api = Github(GITHUB_TOKEN)
                logger.info("GitHub API initialized successfully")
            except Exception as e:
                log_warning(f"GitHub API initialization failed: {str(e)}")
    
    def _check_rate_limit(self, platform: str):
        """Check and enforce rate limiting for specific platform"""
        current_time = time.time()
        rate_limit = self.rate_limits.get(platform.lower(), 60)
        
        if platform in self.last_request_times:
            time_diff = current_time - self.last_request_times[platform]
            min_interval = 60 / rate_limit  # Convert per-minute to interval
            
            if time_diff < min_interval:
                sleep_time = min_interval - time_diff
                time.sleep(sleep_time)
        
        self.last_request_times[platform] = current_time
    
    def search_kaggle_datasets(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for datasets on Kaggle
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of dataset information
        """
        if not self.kaggle_api:
            log_warning("Kaggle API not available")
            return []
        
        # Validate query
        is_valid, error_msg = validate_search_query(query)
        if not is_valid:
            log_error(f"Invalid Kaggle search query: {error_msg}")
            return []
        
        try:
            self._check_rate_limit("kaggle")
            
            # Search datasets
            datasets = self.kaggle_api.dataset_list(search=query, max_size=max_results)
            
            results = []
            for dataset in datasets:
                dataset_info = {
                    "title": dataset.title,
                    "url": f"https://www.kaggle.com/datasets/{dataset.ref}",
                    "description": dataset.subtitle or "No description available",
                    "author": dataset.creatorName,
                    "size": dataset.totalBytes,
                    "download_count": dataset.downloadCount,
                    "vote_count": dataset.voteCount,
                    "usability_rating": dataset.usabilityRating,
                    "last_updated": dataset.lastUpdated.isoformat() if dataset.lastUpdated else None,
                    "file_count": len(dataset.files) if hasattr(dataset, 'files') else 0,
                    "source": "kaggle",
                    "tags": getattr(dataset, 'tags', []),
                    "license": getattr(dataset, 'licenseName', 'Unknown')
                }
                results.append(dataset_info)
            
            log_search_query(query, "kaggle", len(results))
            return results
            
        except Exception as e:
            log_error(f"Kaggle search failed: {str(e)}")
            return []
    
    def search_huggingface_datasets(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for datasets on HuggingFace
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of dataset information
        """
        if not self.hf_api:
            log_warning("HuggingFace API not available")
            return []
        
        # Validate query
        is_valid, error_msg = validate_search_query(query)
        if not is_valid:
            log_error(f"Invalid HuggingFace search query: {error_msg}")
            return []
        
        try:
            self._check_rate_limit("huggingface")
            
            # Search datasets
            datasets = list_datasets(
                search=query,
                limit=max_results,
                full=True
            )
            
            results = []
            for dataset in datasets:
                dataset_info = {
                    "title": dataset.id,
                    "url": f"https://huggingface.co/datasets/{dataset.id}",
                    "description": getattr(dataset, 'description', 'No description available'),
                    "author": dataset.author if hasattr(dataset, 'author') else dataset.id.split('/')[0],
                    "downloads": getattr(dataset, 'downloads', 0),
                    "likes": getattr(dataset, 'likes', 0),
                    "created_at": getattr(dataset, 'created_at', None),
                    "last_modified": getattr(dataset, 'last_modified', None),
                    "source": "huggingface",
                    "tags": getattr(dataset, 'tags', []),
                    "task_categories": getattr(dataset, 'task_categories', []),
                    "language": getattr(dataset, 'language', []),
                    "license": getattr(dataset, 'license', 'Unknown')
                }
                results.append(dataset_info)
            
            log_search_query(query, "huggingface", len(results))
            return results
            
        except Exception as e:
            log_error(f"HuggingFace search failed: {str(e)}")
            return []
    
    def search_github_datasets(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for datasets and data repositories on GitHub
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of repository information
        """
        if not self.github_api:
            log_warning("GitHub API not available - using public search")
            return self._github_public_search(query, max_results)
        
        # Validate query
        is_valid, error_msg = validate_search_query(query)
        if not is_valid:
            log_error(f"Invalid GitHub search query: {error_msg}")
            return []
        
        try:
            self._check_rate_limit("github")
            
            # Enhanced search query for datasets
            search_query = f"{query} dataset OR data OR csv OR json filename:*.csv OR filename:*.json"
            
            # Search repositories
            repos = self.github_api.search_repositories(
                query=search_query,
                sort="stars",
                order="desc"
            )
            
            results = []
            count = 0
            for repo in repos:
                if count >= max_results:
                    break
                
                # Filter for repositories likely to contain datasets
                if self._is_likely_dataset_repo(repo):
                    repo_info = {
                        "title": repo.name,
                        "url": repo.html_url,
                        "description": repo.description or "No description available",
                        "author": repo.owner.login,
                        "stars": repo.stargazers_count,
                        "forks": repo.forks_count,
                        "size": repo.size,
                        "language": repo.language,
                        "created_at": repo.created_at.isoformat() if repo.created_at else None,
                        "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                        "source": "github",
                        "topics": repo.get_topics(),
                        "license": repo.license.name if repo.license else "Unknown",
                        "has_wiki": repo.has_wiki,
                        "has_pages": repo.has_pages
                    }
                    results.append(repo_info)
                    count += 1
            
            log_search_query(query, "github", len(results))
            return results
            
        except Exception as e:
            log_error(f"GitHub search failed: {str(e)}")
            return []
    
    def _github_public_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Fallback public GitHub search using web scraping
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of repository information
        """
        try:
            # Use GitHub's public search API (limited but no auth required)
            search_url = "https://api.github.com/search/repositories"
            params = {
                "q": f"{query} dataset OR data",
                "sort": "stars",
                "order": "desc",
                "per_page": min(max_results, 30)
            }
            
            response = requests.get(search_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for repo in data.get("items", []):
                    repo_info = {
                        "title": repo["name"],
                        "url": repo["html_url"],
                        "description": repo["description"] or "No description available",
                        "author": repo["owner"]["login"],
                        "stars": repo["stargazers_count"],
                        "forks": repo["forks_count"],
                        "size": repo["size"],
                        "language": repo["language"],
                        "created_at": repo["created_at"],
                        "updated_at": repo["updated_at"],
                        "source": "github",
                        "topics": repo.get("topics", []),
                        "license": repo["license"]["name"] if repo.get("license") else "Unknown"
                    }
                    results.append(repo_info)
                
                log_search_query(query, "github-public", len(results))
                return results
            
        except Exception as e:
            log_error(f"GitHub public search failed: {str(e)}")
        
        return []
    
    def _is_likely_dataset_repo(self, repo) -> bool:
        """
        Check if repository is likely to contain datasets
        
        Args:
            repo: GitHub repository object
            
        Returns:
            bool: True if likely contains datasets
        """
        # Keywords that indicate dataset repositories
        dataset_keywords = [
            'dataset', 'data', 'csv', 'json', 'database', 'corpus',
            'collection', 'benchmark', 'samples', 'examples'
        ]
        
        # Check name and description
        repo_text = f"{repo.name} {repo.description or ''}".lower()
        
        # Check if any dataset keywords are present
        has_keywords = any(keyword in repo_text for keyword in dataset_keywords)
        
        # Additional filters
        is_not_too_large = repo.size < 1000000  # Not too large (likely code repo)
        is_not_too_small = repo.size > 100      # Not too small (likely empty)
        has_reasonable_stars = repo.stargazers_count >= 1  # Some community interest
        
        return has_keywords and is_not_too_large and is_not_too_small and has_reasonable_stars
    
    def search_all_platforms(self, query: str, max_results_per_platform: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for datasets across all platforms
        
        Args:
            query (str): Search query
            max_results_per_platform (int): Max results per platform
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Results grouped by platform
        """
        results = {
            "kaggle": [],
            "huggingface": [],
            "github": []
        }
        
        # Search Kaggle
        try:
            results["kaggle"] = self.search_kaggle_datasets(query, max_results_per_platform)
        except Exception as e:
            log_error(f"Kaggle search failed in multi-platform search: {str(e)}")
        
        # Search HuggingFace
        try:
            results["huggingface"] = self.search_huggingface_datasets(query, max_results_per_platform)
        except Exception as e:
            log_error(f"HuggingFace search failed in multi-platform search: {str(e)}")
        
        # Search GitHub
        try:
            results["github"] = self.search_github_datasets(query, max_results_per_platform)
        except Exception as e:
            log_error(f"GitHub search failed in multi-platform search: {str(e)}")
        
        return results
    
    def get_use_case_specific_datasets(self, use_case: str, industry: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find datasets specific to a use case and industry
        
        Args:
            use_case (str): Use case description or type
            industry (str): Industry context
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Relevant datasets by platform
        """
        # Generate targeted search queries
        queries = self._generate_targeted_queries(use_case, industry)
        
        all_results = {
            "kaggle": [],
            "huggingface": [],
            "github": []
        }
        
        for query in queries:
            platform_results = self.search_all_platforms(query, max_results_per_platform=3)
            
            # Merge results
            for platform, datasets in platform_results.items():
                all_results[platform].extend(datasets)
        
        # Remove duplicates and rank by relevance
        for platform in all_results:
            all_results[platform] = self._deduplicate_and_rank(
                all_results[platform], use_case, industry
            )
        
        return all_results
    
    def _generate_targeted_queries(self, use_case: str, industry: str) -> List[str]:
        """
        Generate targeted search queries for specific use cases
        
        Args:
            use_case (str): Use case description
            industry (str): Industry context
            
        Returns:
            List[str]: List of targeted queries
        """
        queries = []
        
        # Base queries
        queries.append(f"{industry} {use_case}")
        queries.append(f"{industry} dataset")
        
        # Use case specific queries
        use_case_lower = use_case.lower()
        
        if "sentiment" in use_case_lower or "nlp" in use_case_lower:
            queries.extend([
                f"{industry} sentiment analysis",
                f"{industry} text data",
                f"{industry} customer reviews"
            ])
        
        elif "predictive" in use_case_lower or "forecasting" in use_case_lower:
            queries.extend([
                f"{industry} time series",
                f"{industry} historical data",
                f"{industry} forecast"
            ])
        
        elif "image" in use_case_lower or "vision" in use_case_lower:
            queries.extend([
                f"{industry} images",
                f"{industry} computer vision",
                f"{industry} visual data"
            ])
        
        elif "recommendation" in use_case_lower:
            queries.extend([
                f"{industry} recommendation",
                f"{industry} user behavior",
                f"{industry} collaborative filtering"
            ])
        
        elif "fraud" in use_case_lower or "anomaly" in use_case_lower:
            queries.extend([
                f"{industry} fraud detection",
                f"{industry} anomaly detection",
                f"{industry} transactions"
            ])
        
        # Generic ML datasets
        queries.append(f"{industry} machine learning")
        
        return queries[:5]  # Limit to top 5 queries
    
    def _deduplicate_and_rank(self, datasets: List[Dict[str, Any]], 
                             use_case: str, industry: str) -> List[Dict[str, Any]]:
        """
        Remove duplicates and rank datasets by relevance
        
        Args:
            datasets (List[Dict[str, Any]]): List of datasets
            use_case (str): Use case context
            industry (str): Industry context
            
        Returns:
            List[Dict[str, Any]]: Deduplicated and ranked datasets
        """
        if not datasets:
            return []
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_datasets = []
        
        for dataset in datasets:
            url = dataset.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_datasets.append(dataset)
        
        # Rank by relevance score
        for dataset in unique_datasets:
            dataset["relevance_score"] = self._calculate_relevance_score(
                dataset, use_case, industry
            )
        
        # Sort by relevance score (descending)
        unique_datasets.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return unique_datasets[:10]  # Return top 10
    
    def _calculate_relevance_score(self, dataset: Dict[str, Any], 
                                  use_case: str, industry: str) -> float:
        """
        Calculate relevance score for a dataset
        
        Args:
            dataset (Dict[str, Any]): Dataset information
            use_case (str): Use case context
            industry (str): Industry context
            
        Returns:
            float: Relevance score (0-100)
        """
        score = 0.0
        
        # Get searchable text from dataset
        text = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
        use_case_lower = use_case.lower()
        industry_lower = industry.lower()
        
        # Industry relevance (0-30 points)
        if industry_lower in text:
            score += 30
        elif any(word in text for word in industry_lower.split()):
            score += 15
        
        # Use case relevance (0-30 points)
        if use_case_lower in text:
            score += 30
        elif any(word in text for word in use_case_lower.split()):
            score += 15
        
        # Quality indicators (0-40 points)
        source = dataset.get("source", "")
        
        if source == "kaggle":
            # Kaggle-specific scoring
            downloads = dataset.get("download_count", 0)
            votes = dataset.get("vote_count", 0)
            usability = dataset.get("usability_rating", 0)
            
            score += min(downloads / 1000, 15)  # Max 15 points for downloads
            score += min(votes / 10, 10)        # Max 10 points for votes
            score += usability * 3               # Max 30 points for usability (0-10 scale)
        
        elif source == "huggingface":
            # HuggingFace-specific scoring
            downloads = dataset.get("downloads", 0)
            likes = dataset.get("likes", 0)
            
            score += min(downloads / 1000, 20)  # Max 20 points for downloads
            score += min(likes / 5, 20)         # Max 20 points for likes
        
        elif source == "github":
            # GitHub-specific scoring
            stars = dataset.get("stars", 0)
            forks = dataset.get("forks", 0)
            
            score += min(stars / 10, 25)        # Max 25 points for stars
            score += min(forks / 5, 15)         # Max 15 points for forks
        
        return min(score, 100)  # Cap at 100
    
    def generate_dataset_report(self, datasets_by_platform: Dict[str, List[Dict[str, Any]]], 
                               use_case: str = None, industry: str = None) -> str:
        """
        Generate a markdown report of found datasets
        
        Args:
            datasets_by_platform (Dict[str, List[Dict[str, Any]]]): Datasets grouped by platform
            use_case (str): Use case context
            industry (str): Industry context
            
        Returns:
            str: Markdown formatted report
        """
        report_lines = []
        
        # Header
        report_lines.append("# Dataset Discovery Report")
        report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if use_case:
            report_lines.append(f"**Use Case:** {use_case}")
        if industry:
            report_lines.append(f"**Industry:** {industry}")
        
        report_lines.append("\n---\n")
        
        # Summary
        total_datasets = sum(len(datasets) for datasets in datasets_by_platform.values())
        report_lines.append(f"## Summary\n")
        report_lines.append(f"Found **{total_datasets} datasets** across multiple platforms:\n")
        
        for platform, datasets in datasets_by_platform.items():
            if datasets:
                report_lines.append(f"- **{platform.title()}:** {len(datasets)} datasets")
        
        report_lines.append("\n---\n")
        
        # Detailed results by platform
        for platform, datasets in datasets_by_platform.items():
            if not datasets:
                continue
            
            report_lines.append(f"## {platform.title()} Datasets\n")
            
            for i, dataset in enumerate(datasets[:10], 1):  # Limit to top 10
                title = dataset.get("title", "Unknown")
                url = dataset.get("url", "#")
                description = dataset.get("description", "No description available")
                author = dataset.get("author", "Unknown")
                
                report_lines.append(f"### {i}. [{title}]({url})\n")
                report_lines.append(f"**Author:** {author}  ")
                
                # Platform-specific metrics
                if platform == "kaggle":
                    downloads = dataset.get("download_count", 0)
                    usability = dataset.get("usability_rating", 0)
                    report_lines.append(f"**Downloads:** {downloads:,} | **Usability:** {usability}/10  ")
                
                elif platform == "huggingface":
                    downloads = dataset.get("downloads", 0)
                    likes = dataset.get("likes", 0)
                    report_lines.append(f"**Downloads:** {downloads:,} | **Likes:** {likes}  ")
                
                elif platform == "github":
                    stars = dataset.get("stars", 0)
                    forks = dataset.get("forks", 0)
                    language = dataset.get("language", "N/A")
                    report_lines.append(f"**Stars:** {stars:,} | **Forks:** {forks} | **Language:** {language}  ")
                
                if "relevance_score" in dataset:
                    score = dataset["relevance_score"]
                    report_lines.append(f"**Relevance Score:** {score:.1f}/100  ")
                
                report_lines.append(f"\n{description[:200]}{'...' if len(description) > 200 else ''}\n\n")
        
        report_lines.append("---\n")
        report_lines.append("*Report generated by Multi-Agent Market Research System*")
        
        return "\n".join(report_lines)

# Global instance
dataset_finder = DatasetFinder()