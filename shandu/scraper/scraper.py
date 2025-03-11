"""
Web scraping module for Shandu research system.
Provides functionality for scraping web pages with WebBaseLoader integration.
"""
import os
import re
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to get USER_AGENT from environment, otherwise use a generic one
USER_AGENT = os.environ.get('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

@dataclass
class ScrapedContent:
    """Class to store scraped content from a web page."""
    url: str
    title: str
    text: str
    html: str
    content_type: str
    metadata: Dict[str, Any]
    error: Optional[str] = None
    
    def is_successful(self) -> bool:
        """Check if scraping was successful."""
        return self.error is None and len(self.text) > 0

class WebScraper:
    """Web scraper for extracting content from web pages using WebBaseLoader."""
    
    def __init__(self, proxy: Optional[str] = None, timeout: int = 30):
        """
        Initialize the web scraper.
        
        Args:
            proxy: Optional proxy URL to use for requests
            timeout: Timeout for requests in seconds
        """
        self.proxy = proxy
        self.timeout = timeout
        self.user_agent = USER_AGENT
        
        # Try to use fake_useragent if available
        try:
            ua = UserAgent()
            self.user_agent = ua.random
        except Exception as e:
            logger.warning(f"Could not generate random user agent: {e}. Using default.")
    
    async def scrape_url(self, url: str, dynamic: bool = False) -> ScrapedContent:
        """
        Scrape content from a URL using WebBaseLoader.
        
        Args:
            url: URL to scrape
            dynamic: Whether to use dynamic rendering (for JavaScript-heavy sites)
            
        Returns:
            ScrapedContent object with the scraped content
        """
        logger.info(f"Scraping URL: {url}")
        
        # Check if URL is valid
        if not url.startswith(('http://', 'https://')):
            return ScrapedContent(
                url=url,
                title="",
                text="",
                html="",
                content_type="",
                metadata={},
                error="Invalid URL format"
            )
        
        try:
            # Configure WebBaseLoader with appropriate settings
            requests_kwargs = {
                "headers": {"User-Agent": self.user_agent},
                "timeout": self.timeout,
                "verify": True  # SSL verification
            }
            
            if self.proxy:
                requests_kwargs["proxies"] = {
                    "http": self.proxy,
                    "https": self.proxy
                }
            
            # Use WebBaseLoader for scraping
            loader = WebBaseLoader(
                web_path=url,
                requests_kwargs=requests_kwargs,
                # Remove the features parameter from bs_kwargs to avoid duplicate argument error
                bs_kwargs={},  # BeautifulSoup already gets features parameter internally
                raise_for_status=True,
                continue_on_failure=False,
                autoset_encoding=True,
                trust_env=True
            )
            
            # If dynamic rendering is requested, use Playwright
            if dynamic:
                try:
                    from playwright.async_api import async_playwright
                    
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(headless=True)
                        page = await browser.new_page(user_agent=self.user_agent)
                        
                        # Set timeout for navigation
                        page.set_default_timeout(self.timeout * 1000)
                        
                        # Navigate to the URL
                        await page.goto(url, wait_until="networkidle")
                        
                        # Get the page content
                        html_content = await page.content()
                        
                        # Get the page title
                        title = await page.title()
                        
                        # Close the browser
                        await browser.close()
                        
                        # Parse the HTML with BeautifulSoup
                        soup = BeautifulSoup(html_content, "html.parser")
                        
                        # Extract metadata
                        metadata = self._extract_metadata(soup, url)
                        
                        # Extract main content
                        main_content = self._extract_main_content(soup)
                        
                        return ScrapedContent(
                            url=url,
                            title=title,
                            text=main_content,
                            html=html_content,
                            content_type="text/html",
                            metadata=metadata
                        )
                except ImportError:
                    logger.warning("Playwright not installed. Falling back to WebBaseLoader.")
                except Exception as e:
                    logger.error(f"Error during dynamic rendering: {e}. Falling back to WebBaseLoader.")
            
            # Load the document using WebBaseLoader
            documents = await asyncio.to_thread(loader.load)
            
            if not documents:
                return ScrapedContent(
                    url=url,
                    title="",
                    text="",
                    html="",
                    content_type="",
                    metadata={},
                    error="No content found"
                )
            
            # Get the first document (there should only be one for a single URL)
            document = documents[0]
            
            # Extract metadata from the document
            metadata = document.metadata
            
            # Get the page content
            text_content = document.page_content
            with open('example.txt', 'a') as file:
                file.write(f'Text content: {text_content}\n')
            # Split long content for better processing
            if len(text_content) > 10000:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=10000,
                    chunk_overlap=200,
                    length_function=len,
                )
                chunks = text_splitter.split_text(text_content)
                text_content = "\n\n".join(chunks[:3])  # Use first 3 chunks
                with open('example.txt', 'a') as file:
                    file.write(f'Long: {text_content}\n')

            # Get HTML content if available
            html_content = ""
            if hasattr(loader, "_html_content") and loader._html_content:
                html_content = loader._html_content
            
            # Get title from metadata or extract from HTML
            title = metadata.get("title", "")
            if not title and html_content:
                soup = BeautifulSoup(html_content, "html.parser")
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.text.strip()
            
            # Get content type
            content_type = metadata.get("content-type", "text/html")
            
            return ScrapedContent(
                url=url,
                title=title,
                text=text_content,
                html=html_content,
                content_type=content_type,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return ScrapedContent(
                url=url,
                title="",
                text="",
                html="",
                content_type="",
                metadata={},
                error=str(e)
            )
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract metadata from a BeautifulSoup object."""
        metadata = {
            "url": url,
            "domain": urlparse(url).netloc
        }
        
        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.text.strip()
        
        # Extract meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", ""))
            content = meta.get("content", "")
            if name and content:
                metadata[name.lower()] = content
        
        return metadata
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from a BeautifulSoup object."""
        # Try to find main content containers
        main_tags = soup.find_all(["main", "article", "div", "section"], class_=lambda c: c and any(x in str(c).lower() for x in ["content", "main", "article", "body"]))
        
        if main_tags:
            # Use the largest content container
            main_tag = max(main_tags, key=lambda tag: len(tag.get_text()))
            content = main_tag.get_text(separator="\n", strip=True)
            
            # Clean up the content
            content = re.sub(r'\n{3,}', '\n\n', content)
            return content
        
        # If no main content container found, use the body
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)
        
        # If no body found, use the entire HTML
        return soup.get_text(separator="\n", strip=True)
    
    async def scrape_urls(self, urls: List[str], dynamic: bool = False) -> List[ScrapedContent]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            dynamic: Whether to use dynamic rendering
            
        Returns:
            List of ScrapedContent objects
        """
        tasks = [self.scrape_url(url, dynamic) for url in urls]
        return await asyncio.gather(*tasks)

# Structured output models for scraping
class ScrapingResult(BaseModel):
    """Structured output for scraping results."""
    url: str = Field(description="URL of the scraped page")
    title: str = Field(description="Title of the page")
    content: str = Field(description="Extracted content from the page")
    success: bool = Field(description="Whether scraping was successful")
    error: Optional[str] = Field(description="Error message if scraping failed", default=None)
