"""
Search module for Shandu research system.
Provides functionality for searching the web using various search engines.
"""
import os
import asyncio
import time
import random
import json
from typing import List, Dict, Any, Optional, Union, Set
from functools import lru_cache
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

# Cache settings
CACHE_ENABLED = True
CACHE_DIR = os.path.expanduser("~/.shandu/cache/search")
CACHE_TTL = 86400  # 24 hours in seconds

# Create cache directory if it doesn't exist
if CACHE_ENABLED and not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create cache directory: {e}")
        CACHE_ENABLED = False

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
    """Unified search engine that can use multiple search engines with improved parallelism and caching."""
    
    def __init__(self, max_results: int = 10, cache_enabled: bool = CACHE_ENABLED, cache_ttl: int = CACHE_TTL):
        """
        Initialize the unified searcher.
        
        Args:
            max_results: Maximum number of results to return per engine
            cache_enabled: Whether to use caching for search results
            cache_ttl: Time-to-live for cached content in seconds
        """
        self.max_results = max_results
        self.user_agent = USER_AGENT
        self.default_engine = "google"  # Set a default engine
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.in_progress_queries: Set[str] = set()  # Track queries being processed to prevent duplicates
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent engine searches to 5
        
        # Try to use fake_useragent if available
        try:
            ua = UserAgent()
            self.user_agent = ua.random
        except Exception as e:
            logger.warning(f"Could not generate random user agent: {e}. Using default.")
    
    async def _check_cache(self, query: str, engine: str) -> Optional[List[SearchResult]]:
        """Check if search results are available in cache and not expired."""
        if not self.cache_enabled:
            return None
            
        cache_key = f"{engine}_{query}".replace(" ", "_").replace("/", "_").replace(".", "_")
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            # Check if cache is expired
            if time.time() - os.path.getmtime(cache_path) > self.cache_ttl:
                return None
                
            # Load cached content
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to SearchResult objects
            results = []
            for item in data:
                results.append(SearchResult(
                    url=item["url"],
                    title=item["title"],
                    snippet=item["snippet"],
                    source=item["source"]
                ))
            return results
        except Exception as e:
            logger.warning(f"Error loading cache for {query} on {engine}: {e}")
            return None
    
    async def _save_to_cache(self, query: str, engine: str, results: List[SearchResult]) -> bool:
        """Save search results to cache."""
        if not self.cache_enabled or not results:
            return False
            
        cache_key = f"{engine}_{query}".replace(" ", "_").replace("/", "_").replace(".", "_")
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        try:
            # Create a serializable representation
            data = [result.to_dict() for result in results]
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            return True
        except Exception as e:
            logger.warning(f"Error saving cache for {query} on {engine}: {e}")
            return False
    
    async def search(self, query: str, engines: Optional[List[str]] = None, force_refresh: bool = False) -> List[SearchResult]:
        """
        Search for a query using multiple engines.
        
        Args:
            query: Query to search for
            engines: List of engines to use (google, duckduckgo, bing, etc.)
            force_refresh: Whether to ignore cache and force fresh searches
        
        Returns:
            List of search results
        """
        if engines is None:
            engines = ["google"]
        
        # Ensure engines is a list
        if isinstance(engines, str):
            engines = [engines]
        
        # Use set to ensure unique engine names (case insensitive)
        unique_engines = set(engine.lower() for engine in engines)
        
        # Create tasks for each engine with proper concurrency control
        tasks = []
        for engine in unique_engines:
            # Skip unsupported engines
            if engine not in ["google", "duckduckgo", "bing", "wikipedia"]:
                logger.warning(f"Unknown search engine: {engine}")
                continue
                
            # First check cache unless forcing refresh
            if not force_refresh:
                cached_results = await self._check_cache(query, engine)
                if cached_results:
                    logger.info(f"Using cached results for {query} on {engine}")
                    tasks.append(asyncio.create_task(asyncio.sleep(0, result=cached_results)))
                    continue
            
            # Execute search with appropriate engine method
            if engine == "google":
                tasks.append(self._search_with_retry(self._search_google, query))
            elif engine == "duckduckgo":
                tasks.append(self._search_with_retry(self._search_duckduckgo, query))
            elif engine == "bing":
                tasks.append(self._search_with_retry(self._search_bing, query))
            elif engine == "wikipedia":
                tasks.append(self._search_with_retry(self._search_wikipedia, query))
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter out exceptions
        all_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error during search: {result}")
            else:
                all_results.extend(result)
        
        # Process results: remove duplicates, shuffle, and limit
        unique_urls = set()
        unique_results = []
        
        for result in all_results:
            if result.url not in unique_urls:
                unique_urls.add(result.url)
                unique_results.append(result)
                
        # Shuffle results to mix engines
        random.shuffle(unique_results)
        
        # Limit to max_results
        return unique_results[:self.max_results]
    
    async def _search_with_retry(self, search_function, query: str, max_retries: int = 2) -> List[SearchResult]:
        """Wrapper that adds retry logic to search functions."""
        retries = 0
        engine_name = search_function.__name__.replace("_search_", "")
        
        while retries <= max_retries:
            try:
                # Add a small random delay to avoid rate limiting
                if retries > 0:
                    delay = random.uniform(1.0, 3.0) * retries
                    await asyncio.sleep(delay)
                    
                # Acquire semaphore to limit concurrent searches
                async with self.semaphore:
                    # Add jitter for concurrent requests
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                    
                    # Execute the search
                    results = await search_function(query)
                    
                    # Cache successful results
                    if results:
                        await self._save_to_cache(query, engine_name, results)
                        
                    return results
                    
            except Exception as e:
                logger.warning(f"Search attempt {retries + 1} failed for {engine_name}: {e}")
                retries += 1
                if retries > max_retries:
                    logger.error(f"All {max_retries + 1} attempts failed for {engine_name} search: {e}")
                    return []
    
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
            google_results = list(google_search(query, num_results=self.max_results))
            for j in google_results:
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
            raise  # Re-raise for retry mechanism
    
    async def _enrich_google_results(self, results: List[SearchResult], query: str) -> None:
        """
        Enrich Google search results with titles and snippets.
        
        Args:
            results: List of search results to enrich
            query: Original query
        """
        try:
            # Create a session
            timeout = aiohttp.ClientTimeout(total=15)  # 15 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                            
        except asyncio.TimeoutError:
            logger.warning("Timeout while enriching Google results")
        except Exception as e:
            logger.error(f"Error enriching Google results: {e}")
            # We don't raise here since this is supplementary information
    
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
            timeout = aiohttp.ClientTimeout(total=15)  # 15 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Get the DuckDuckGo search page
                url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                headers = {"User-Agent": self.user_agent}
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"DuckDuckGo search returned status code {response.status}")
                        raise ValueError(f"DuckDuckGo search returned status code {response.status}")
                    
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
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout during DuckDuckGo search for query: {query}")
            raise
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {e}")
            raise  # Re-raise for retry mechanism
    
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
            timeout = aiohttp.ClientTimeout(total=15)  # 15 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Get the Bing search page
                url = f"https://www.bing.com/search?q={quote_plus(query)}"
                headers = {"User-Agent": self.user_agent}
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"Bing search returned status code {response.status}")
                        raise ValueError(f"Bing search returned status code {response.status}")
                    
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
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout during Bing search for query: {query}")
            raise  
        except Exception as e:
            logger.error(f"Error during Bing search: {e}")
            raise  # Re-raise for retry mechanism
    
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
            timeout = aiohttp.ClientTimeout(total=15)  # 15 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Get the Wikipedia search API
                url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(query)}&limit={self.max_results}&namespace=0&format=json"
                headers = {"User-Agent": self.user_agent}
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"Wikipedia search returned status code {response.status}")
                        raise ValueError(f"Wikipedia search returned status code {response.status}")
                    
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
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout during Wikipedia search for query: {query}")
            raise
        except Exception as e:
            logger.error(f"Error during Wikipedia search: {e}")
            raise  # Re-raise for retry mechanism
    
    def search_sync(self, query: str, engines: Optional[List[str]] = None, force_refresh: bool = False) -> List[SearchResult]:
        """
        Synchronous version of search.
        
        Args:
            query: Query to search for
            engines: List of engines to use
            force_refresh: Whether to ignore cache and force fresh searches
        
        Returns:
            List of search results
        """
        return asyncio.run(self.search(query, engines, force_refresh))
    
    @lru_cache(maxsize=128)
    def _get_formatted_query(self, query: str, engine: str) -> str:
        """Create a cache-friendly formatted query string."""
        return f"{engine.lower()}:{query.lower()}"
