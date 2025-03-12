from typing import List, Dict, Optional, Any, Union
import asyncio
from dataclasses import dataclass
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .search import UnifiedSearcher, SearchResult
from ..config import config
from ..scraper import WebScraper, ScrapedContent

@dataclass
class AISearchResult:
    """Container for AI-enhanced search results with enriched output."""
    query: str
    summary: str
    sources: List[Dict[str, Any]]
    timestamp: datetime = datetime.now()
    
    def to_markdown(self) -> str:
        """Convert to markdown format with improved readability."""
        timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        md = [
            f"# Search Results for: {self.query}",
            f"*Generated on: {timestamp_str}*",
            "## Summary",
            self.summary,
            "## Sources"
        ]
        for i, source in enumerate(self.sources, 1):
            title = source.get('title', 'Untitled')
            url = source.get('url', '')
            snippet = source.get('snippet', '')
            source_type = source.get('source', 'Unknown')
            md.append(f"### {i}. {title}")
            if url:
                md.append(f"- **URL:** [{url}]({url})")
            if source_type:
                md.append(f"- **Source:** {source_type}")
            if snippet:
                md.append(f"- **Snippet:** {snippet}")
            md.append("")
        return "\n".join(md)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "query": self.query,
            "summary": self.summary,
            "sources": self.sources,
            "timestamp": self.timestamp.isoformat()
        }

class AISearcher:
    """
    AI-powered search functionality.
    Combines search results with AI analysis for any type of query.
    Enhanced with scraping capability, detailed outputs, and source citations.
    """
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        searcher: Optional[UnifiedSearcher] = None,
        scraper: Optional[WebScraper] = None,
        max_results: int = 10,
        max_pages_to_scrape: int = 3
    ):
        api_base = config.get("api", "base_url")
        api_key = config.get("api", "api_key")
        model = config.get("api", "model")
        self.llm = llm or ChatOpenAI(
            base_url=api_base,
            api_key=api_key,
            model=model,
            temperature=0.4,
            max_tokens=8192
        )
        self.searcher = searcher or UnifiedSearcher(max_results=max_results)
        self.scraper = scraper or WebScraper()
        self.max_results = max_results
        self.max_pages_to_scrape = max_pages_to_scrape
    
    async def search(
        self, 
        query: str,
        engines: Optional[List[str]] = None,
        detailed: bool = False,
        enable_scraping: bool = True
    ) -> AISearchResult:
        """
        Perform AI-enhanced search with detailed outputs and source citations.
        
        Args:
            query: Search query (can be about any topic)
            engines: List of search engines to use
            detailed: Whether to generate a detailed analysis
            enable_scraping: Whether to scrape content from top results
            
        Returns:
            AISearchResult object with a comprehensive summary and cited sources
        """
        timestamp = datetime.now()
        search_results = await self.searcher.search(query, engines)
        
        # Collect all sources
        sources = []
        for result in search_results:
            if isinstance(result, SearchResult):
                result_dict = result.to_dict()
                sources.append(result_dict)
            elif isinstance(result, dict):
                sources.append(result)
        
        # Scrape additional content if enabled
        if enable_scraping:
            urls_to_scrape = []
            for source in sources:
                if source.get('url') and len(urls_to_scrape) < self.max_pages_to_scrape:
                    urls_to_scrape.append(source['url'])
            if urls_to_scrape:
                print(f"Scraping {len(urls_to_scrape)} pages for deeper insights...")
                scraped_results = await self.scraper.scrape_urls(urls_to_scrape, dynamic=True)
                for scraped in scraped_results:
                    if hasattr(scraped, 'is_successful') and scraped.is_successful():
                        try:
                            main_content = scraped.text
                            if hasattr(self.scraper, 'extract_main_content'):
                                main_content = await self.scraper.extract_main_content(scraped)
                            if "unexpected error" in main_content.lower():
                                continue
                            preview = main_content[:500] + ("...(truncated)" if len(main_content) > 1500 else "")
                            sources.append({
                                "title": scraped.title,
                                "url": scraped.url,
                                "snippet": preview,
                                "source": "Scraped Content"
                            })
                        except Exception as e:
                            print(f"Error processing scraped content from {scraped.url}: {e}")
        
        # Prepare sources with improved citation format
        aggregated_text = ""
        for i, source in enumerate(sources, 1):
            # Extract domain from URL
            url = source.get('url', '')
            domain = url.split("//")[1].split("/")[0] if "//" in url else "Unknown Source"
            # Capitalize first letter of domain for a more professional look
            domain_name = domain.split('.')[0].capitalize() if '.' in domain else domain
            
            aggregated_text += (
                f"[{i}] {domain_name}\n"
                f"Title: {source.get('title', 'Untitled')}\n"
                f"URL: {url}\n"
                f"Snippet: {source.get('snippet', '')}\n\n"
            )
        
        current_date = timestamp.strftime('%Y-%m-%d')
        if detailed:
            detail_instruction = (
                "Provide a detailed and comprehensive analysis, including in-depth explanations, "
                "specific examples, relevant background information, and any additional insights "
                "that enhance understanding of the topic."
            )
        else:
            detail_instruction = "Provide a concise yet informative summary, focusing on the key points and essential information."
        
        final_prompt = f"""You are Shandu, an expert analyst. Based on the following sources retrieved on {current_date} for the query "{query}", {detail_instruction}

- If the query is a question, answer it directly with a thorough explanation.
- If it's a topic, provide a well-rounded overview with supporting details.
- Use bullet points or numbered lists to organize information clearly.
- If there are conflicting views or uncertainties, discuss them explicitly.
- When providing information, cite the source by using the number in square brackets, like [1], to indicate where the information was sourced.
- ONLY use the citation numbers provided in the sources below.
- DO NOT include years or dates in your citations, just use the bracketed number like [1].
- Ensure the response is engaging, detailed, and written in plain text suitable for all readers.

Sources:

{aggregated_text}
"""
        
        final_output = await self.llm.ainvoke(final_prompt)
        
        return AISearchResult(
            query=query,
            summary=final_output.content.strip(),
            sources=sources,
            timestamp=timestamp
        )
    
    def search_sync(
        self, 
        query: str,
        engines: Optional[List[str]] = None,
        detailed: bool = False,
        enable_scraping: bool = True
    ) -> AISearchResult:
        """Synchronous version of the search method."""
        return asyncio.run(self.search(query, engines, detailed, enable_scraping))
