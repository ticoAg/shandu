"""
Search module for Shandu research system.
Provides functionality for searching the web using various search engines.
"""
import os
import asyncio
import time
import random
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from urllib.parse import quote_plus
import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from googlesearch import search as google_search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to get USER_AGENT from environment, otherwise use a generic one
USER_AGENT = os.environ.get('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

@dataclass
class SearchResult:
    """Class to store search results."""
    url: str
    title: str
    snippet: str
    source: str
    
    def __str__(self) -> str:
        """String representation of search result."""
        return f"Title: {self.title}\nURL: {self.url}\nSnippet: {self.snippet}\nSource: {self.source}"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source
        }

class UnifiedSearcher:
    """Unified search engine that can use multiple search engines."""
    
    def __init__(self, max_results: int = 10):
        """
        Initialize the unified searcher.
        
        Args:
            max_results: Maximum number of results to return per engine
        """
        self.max_results = max_results
        self.user_agent = USER_AGENT
        self.default_engine = "google"  # Set a default engine
        
        # Try to use fake_useragent if available
        try:
            ua = UserAgent()
            self.user_agent = ua.random
        except Exception as e:
            logger.warning(f"Could not generate random user agent: {e}. Using default.")
    
    async def search(self, query: str, engines: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Search for a query using multiple engines.
        
        Args:
            query: Query to search for
            engines: List of engines to use (google, duckduckgo, bing, etc.)
            
        Returns:
            List of search results
        """
        if engines is None:
            engines = ["google"]
        
        # Ensure engines is a list
        if isinstance(engines, str):
            engines = [engines]
        
        # Create tasks for each engine
        tasks = []
        for engine in engines:
            if engine.lower() == "google":
                tasks.append(self._search_google(query))
            elif engine.lower() == "duckduckgo":
                tasks.append(self._search_duckduckgo(query))
            elif engine.lower() == "bing":
                tasks.append(self._search_bing(query))
            elif engine.lower() == "wikipedia":
                tasks.append(self._search_wikipedia(query))
            else:
                logger.warning(f"Unknown search engine: {engine}")
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter out exceptions
        all_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error during search: {result}")
            else:
                all_results.extend(result)
        
        # Shuffle results to mix engines
        random.shuffle(all_results)
        
        # Limit to max_results
        return all_results[:self.max_results]
    
    async def _search_google(self, query: str) -> List[SearchResult]:
        """
        Search Google for a query.
        
        Args:
            query: Query to search for
            
        Returns:
            List of search results
        """
        try:
            # Use googlesearch-python library
            results = []
            for j in google_search(query, num_results=self.max_results):
                # Create a search result
                result = SearchResult(
                    url=j,
                    title=j,  # We don't have titles from this library
                    snippet="",  # We don't have snippets from this library
                    source="Google"
                )
                results.append(result)
            
            # Try to get titles and snippets
            if results:
                await self._enrich_google_results(results, query)
            
            return results
        except Exception as e:
            logger.error(f"Error during Google search: {e}")
            return []
    
    async def _enrich_google_results(self, results: List[SearchResult], query: str) -> None:
        """
        Enrich Google search results with titles and snippets.
        
        Args:
            results: List of search results to enrich
            query: Original query
        """
        try:
            # Create a session
            async with aiohttp.ClientSession() as session:
                # Get the Google search page
                url = f"https://www.google.com/search?q={quote_plus(query)}"
                headers = {"User-Agent": self.user_agent}
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"Google search returned status code {response.status}")
                        return
                    
                    # Parse the HTML
                    html = await response.text()
                    soup = BeautifulSoup(html, features="html.parser")
                    
                    # Find search results
                    search_divs = soup.find_all("div", class_="g")
                    
                    # Extract titles and snippets
                    for i, div in enumerate(search_divs):
                        if i >= len(results):
                            break
                        
                        # Find title
                        title_elem = div.find("h3")
                        if title_elem:
                            results[i].title = title_elem.text.strip()
                        
                        # Find snippet
                        snippet_elem = div.find("div", class_="VwiC3b")
                        if snippet_elem:
                            results[i].snippet = snippet_elem.text.strip()
        except Exception as e:
            logger.error(f"Error enriching Google results: {e}")
    
    async def _search_duckduckgo(self, query: str) -> List[SearchResult]:
        """
        Search DuckDuckGo for a query.
        
        Args:
            query: Query to search for
            
        Returns:
            List of search results
        """
        try:
            # Create a session
            async with aiohttp.ClientSession() as session:
                # Get the DuckDuckGo search page
                url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                headers = {"User-Agent": self.user_agent}
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"DuckDuckGo search returned status code {response.status}")
                        return []
                    
                    # Parse the HTML
                    html = await response.text()
                    soup = BeautifulSoup(html, features="html.parser")
                    
                    # Find search results
                    results = []
                    for result in soup.find_all("div", class_="result"):
                        # Find title
                        title_elem = result.find("a", class_="result__a")
                        if not title_elem:
                            continue
                        
                        title = title_elem.text.strip()
                        
                        # Find URL
                        url = title_elem.get("href", "")
                        if not url:
                            continue
                        
                        # Clean URL
                        if url.startswith("/"):
                            url = "https://duckduckgo.com" + url
                        
                        # Find snippet
                        snippet_elem = result.find("a", class_="result__snippet")
                        snippet = snippet_elem.text.strip() if snippet_elem else ""
                        
                        # Create a search result
                        result = SearchResult(
                            url=url,
                            title=title,
                            snippet=snippet,
                            source="DuckDuckGo"
                        )
                        results.append(result)
                        
                        # Limit to max_results
                        if len(results) >= self.max_results:
                            break
                    
                    return results
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {e}")
            return []
    
    async def _search_bing(self, query: str) -> List[SearchResult]:
        """
        Search Bing for a query.
        
        Args:
            query: Query to search for
            
        Returns:
            List of search results
        """
        try:
            # Create a session
            async with aiohttp.ClientSession() as session:
                # Get the Bing search page
                url = f"https://www.bing.com/search?q={quote_plus(query)}"
                headers = {"User-Agent": self.user_agent}
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"Bing search returned status code {response.status}")
                        return []
                    
                    # Parse the HTML
                    html = await response.text()
                    soup = BeautifulSoup(html, features="html.parser")
                    
                    # Find search results
                    results = []
                    for result in soup.find_all("li", class_="b_algo"):
                        # Find title
                        title_elem = result.find("h2")
                        if not title_elem:
                            continue
                        
                        title = title_elem.text.strip()
                        
                        # Find URL
                        url_elem = title_elem.find("a")
                        if not url_elem:
                            continue
                        
                        url = url_elem.get("href", "")
                        if not url:
                            continue
                        
                        # Find snippet
                        snippet_elem = result.find("div", class_="b_caption")
                        snippet = ""
                        if snippet_elem:
                            p_elem = snippet_elem.find("p")
                            if p_elem:
                                snippet = p_elem.text.strip()
                        
                        # Create a search result
                        result = SearchResult(
                            url=url,
                            title=title,
                            snippet=snippet,
                            source="Bing"
                        )
                        results.append(result)
                        
                        # Limit to max_results
                        if len(results) >= self.max_results:
                            break
                    
                    return results
        except Exception as e:
            logger.error(f"Error during Bing search: {e}")
            return []
    
    async def _search_wikipedia(self, query: str) -> List[SearchResult]:
        """
        Search Wikipedia for a query.
        
        Args:
            query: Query to search for
            
        Returns:
            List of search results
        """
        try:
            # Create a session
            async with aiohttp.ClientSession() as session:
                # Get the Wikipedia search API
                url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(query)}&limit={self.max_results}&namespace=0&format=json"
                headers = {"User-Agent": self.user_agent}
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"Wikipedia search returned status code {response.status}")
                        return []
                    
                    # Parse the JSON
                    data = await response.json()
                    
                    # Extract results
                    results = []
                    for i in range(len(data[1])):
                        title = data[1][i]
                        snippet = data[2][i]
                        url = data[3][i]
                        
                        # Create a search result
                        result = SearchResult(
                            url=url,
                            title=title,
                            snippet=snippet,
                            source="Wikipedia"
                        )
                        results.append(result)
                    
                    return results
        except Exception as e:
            logger.error(f"Error during Wikipedia search: {e}")
            return []
    
    def search_sync(self, query: str, engines: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Synchronous version of search.
        
        Args:
            query: Query to search for
            engines: List of engines to use
            
        Returns:
            List of search results
        """
        return asyncio.run(self.search(query, engines))
