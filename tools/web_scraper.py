"""
Tool: Web Scraper
Fetches and parses webpages for relevant text snippets.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class WebScraperTool:
    """
    Tool for fetching and extracting text from web pages.
    """

    def __init__(self, user_agent: Optional[str] = None):
        self.headers = {
            "User-Agent": user_agent or "Mozilla/5.0 (compatible; MarketResearchBot/1.0)"
        }

    def run(self, url: str, max_chars: int = 2000) -> Optional[Dict[str, str]]:
        """
        Fetch and extract text from a webpage.

        Args:
            url (str): Target URL.
            max_chars (int): Max length of extracted text.

        Returns:
            Optional[Dict[str, str]]: Dict with title, snippet, and URL.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Fetched URL: {url}")

            soup = BeautifulSoup(response.text, "lxml")
            title = soup.title.string.strip() if soup.title else url

            for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
                tag.decompose()

            text = " ".join(soup.stripped_strings)
            text = text[:max_chars] + "..." if len(text) > max_chars else text

            return {"title": title, "snippet": text, "url": url}

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            return None


# Expose as a ready-to-use tool
web_scraper_tool = WebScraperTool()
