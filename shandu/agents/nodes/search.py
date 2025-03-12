"""
Search node for research graph.
"""
import asyncio
import time
import random
from typing import List, Dict, Any, Optional
from rich.console import Console
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from ..processors.content_processor import AgentState, is_relevant_url, process_scraped_item, analyze_content
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback, is_shutdown_requested
from ...search.search import SearchResult

console = Console()

# Structured output model for search results
class SearchResultAnalysis(BaseModel):
    """Structured output for search result analysis."""
    relevant_urls: list[str] = Field(
        description="List of URLs that are relevant to the query",
        min_items=0
    )
    analysis: str = Field(
        description="Analysis of the search results"
    )

async def search_node(llm, searcher, scraper, progress_callback, state: AgentState) -> AgentState:
    """
    Search for information based on the current subqueries.
    
    Args:
        llm: Language model to use
        searcher: Search engine to use
        scraper: Web scraper to use
        progress_callback: Callback function for progress updates
        state: Current agent state
        
    Returns:
        Updated agent state
    """
    if is_shutdown_requested():
        state["status"] = "Shutdown requested, skipping search"
        log_chain_of_thought(state, "Shutdown requested, skipping search")
        return state
    
    state["status"] = f"Searching for information (Depth {state['current_depth']})"
    
    breadth = state["breadth"]
    if len(state["subqueries"]) > 0:
        recent_queries = state["subqueries"][-breadth:]
    else:
        recent_queries = [state["query"]]
    
    for query_idx, query in enumerate(recent_queries):
        if is_shutdown_requested():
            log_chain_of_thought(state, f"Shutdown requested, stopping search after {query_idx} queries")
            break
            
        console.print(f"Executing {searcher.default_engine} search for: {query}")
        state["status"] = f"Searching for: {query}"
        
        # Search for the query
        try:
            search_results = await searcher.search(query, engines=[searcher.default_engine])
        except Exception as e:
            console.print(f"[red]Error during search: {e}[/]")
            log_chain_of_thought(state, f"Error during search for '{query}': {str(e)}")
            continue
        
        # Filter relevant URLs
        relevant_urls = []
        for result in search_results:
            console.print(f"{searcher.default_engine} search result type: {type(result)}")
            console.print(f"{searcher.default_engine} search result attributes: {result.__dict__}")
            
            is_relevant = await is_relevant_url(llm, result.url, result.title, result.snippet, query)
            
            if is_relevant:
                relevant_urls.append(result)
                
                # Add to sources
                state["sources"].append({
                    "url": result.url,
                    "title": result.title,
                    "snippet": result.snippet,
                    "source": result.source,
                    "query": query
                })
        
        if not relevant_urls:
            log_chain_of_thought(state, f"No relevant URLs found for '{query}'")
            continue
        
        relevant_urls = relevant_urls[:5]
        
        # Scrape the relevant URLs
        urls_to_scrape = [result.url for result in relevant_urls]
        
        batch_size = 3
        
        batch_size = max(1, batch_size)
        
        if len(urls_to_scrape) > 0 and batch_size > 0:
            num_batches = (len(urls_to_scrape) + batch_size - 1) // batch_size
        else:
            num_batches = 0
        
        scraped_contents = []
        
        # Process each batch
        for i in range(0, len(urls_to_scrape), batch_size):
            if is_shutdown_requested():
                log_chain_of_thought(state, f"Shutdown requested, stopping scraping after {len(scraped_contents)} URLs")
                break
                
            batch = urls_to_scrape[i:i+batch_size]
            
            # Scrape the batch
            batch_results = await scraper.scrape_urls(batch)
            scraped_contents.extend(batch_results)
            
            # Add a small delay between batches
            if i + batch_size < len(urls_to_scrape):
                await asyncio.sleep(random.uniform(1.0, 2.0))
        
        # Process each scraped item
        processed_items = []
        for item in scraped_contents:
            if not item.is_successful():
                continue
                
            console.print(f"Analyzing page: {item.title}")
            console.print(f"URL: {item.url}")
            
            content_preview = item.text[:200] + "..." if len(item.text) > 200 else item.text
            console.print(f"Extracted Content: {content_preview}")
            
            processed_item = await process_scraped_item(llm, item, query, item.text)
            processed_items.append(processed_item)
        
        if not processed_items:
            log_chain_of_thought(state, f"No content could be extracted from URLs for '{query}'")
            continue
        
        combined_content = "\n\n".join([f"Source: {item['item'].url}\nTitle: {item['item'].title}\nContent: {item['content']}" for item in processed_items])
        
        analysis = await analyze_content(llm, query, combined_content)
        
        state["content_analysis"].append({
            "query": query,
            "sources": [item["item"].url for item in processed_items],
            "analysis": analysis
        })
        
        state["findings"] += f"\n\n## Analysis for: {query}\n\n{analysis}\n\n"
        
        log_chain_of_thought(state, f"Analyzed content for query: {query}")
        if progress_callback:
            await _call_progress_callback(progress_callback, state)
    
    state["current_depth"] += 1
    log_chain_of_thought(state, f"Completed depth {state['current_depth']} of {state['depth']}")
    
    return state
