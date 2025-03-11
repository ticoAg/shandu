"""Report generation nodes."""
import re
import time
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from ..processors.content_processor import AgentState
from ..processors.report_generator import (
    generate_title, 
    extract_themes, 
    generate_initial_report,
    enhance_report,
    expand_key_sections
)
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback, is_shutdown_requested

console = Console()

# Structured output models for report generation
class ReportSection(BaseModel):
    """Structured output for a report section."""
    title: str = Field(description="Title of the section")
    content: str = Field(description="Content of the section")

class FinalReport(BaseModel):
    """Structured output for the final report."""
    title: str = Field(description="Title of the report")
    sections: list[ReportSection] = Field(
        description="List of report sections",
        min_items=1
    )
    references: list[str] = Field(
        description="List of references in the report",
        min_items=0
    )

async def generate_initial_report_node(llm, include_objective, progress_callback, state: AgentState) -> AgentState:
    """Generate the initial report."""
    state["status"] = "Generating initial report"
    console.print("[bold blue]Generating initial comprehensive report with dynamic structure...[/]")

    current_date = state["current_date"]
    
    # Generate a professional title for the report
    report_title = await generate_title(llm, state['query'])
    console.print(f"[bold green]Generated title: {report_title}[/]")
    
    # Extract themes using our processor function
    extracted_themes = await extract_themes(llm, state['findings'])
    
    # Generate the initial report using our processor functions
    initial_report = await generate_initial_report(
        llm,
        state['query'],
        state['findings'],
        extracted_themes,
        report_title,
        state['selected_sources'],
        state.get('formatted_citations', ''),
        current_date,
        state['detail_level'],
        include_objective
    )
    
    # Store the themes for later expansion steps
    state["identified_themes"] = extracted_themes
    state["initial_report"] = initial_report
    log_chain_of_thought(state, "Generated initial comprehensive report with dynamic structure based on content themes")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state

async def enhance_report_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Enhance the report."""
    # Check if shutdown was requested
    if is_shutdown_requested():
        state["status"] = "Shutdown requested, skipping report enhancement"
        log_chain_of_thought(state, "Shutdown requested, skipping report enhancement")
        return state
        
    state["status"] = "Enhancing report with additional detail"
    console.print("[bold blue]Enhancing report with additional depth and detail...[/]")

    # Use the enhance_report function from processors
    enhanced_report = await enhance_report(
        llm, 
        state["initial_report"], 
        state["current_date"], 
        state.get("formatted_citations", ""),
        state.get("selected_sources", []),
        state["sources"]
    )
    
    state["enhanced_report"] = enhanced_report
    log_chain_of_thought(state, "Enhanced report with additional depth and detail")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state

async def expand_key_sections_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Expand key sections of the report."""
    # Check if shutdown was requested
    if is_shutdown_requested():
        state["status"] = "Shutdown requested, skipping section expansion"
        log_chain_of_thought(state, "Shutdown requested, skipping section expansion")
        return state
        
    state["status"] = "Expanding key sections for maximum depth"
    console.print("[bold blue]Expanding key sections for maximum depth using multi-step synthesis...[/]")

    # Use the expand_key_sections function from processors
    expanded_report = await expand_key_sections(
        llm,
        state["enhanced_report"],
        state["identified_themes"],
        state["current_date"]
    )
    
    state["final_report"] = expanded_report
    log_chain_of_thought(state, "Expanded key sections using multi-step synthesis")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state

async def report_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Finalize the report."""
    state["status"] = "Finalizing report"
    console.print("[bold blue]Research complete. Finalizing report...[/]")

    # Check if we have any report
    has_report = False
    if "final_report" in state and state["final_report"]:
        final_report = state["final_report"]
        has_report = True
    elif "enhanced_report" in state and state["enhanced_report"]:
        final_report = state["enhanced_report"]
        has_report = True
    elif "initial_report" in state and state["initial_report"]:
        final_report = state["initial_report"]
        has_report = True
    
    # If we have a report but it's broken or too short, regenerate it
    if has_report and (len(final_report.strip()) < 1000):
        console.print("[bold yellow]Existing report appears broken or incomplete. Regenerating...[/]")
        has_report = False
        
    # If we don't have a report, regenerate initial, enhanced, and expanded reports
    if not has_report:
        console.print("[bold yellow]No valid report found. Regenerating report from scratch...[/]")
        
        # Generate a professional title for the report
        report_title = await generate_title(llm, state['query'])
        console.print(f"[bold green]Generated title: {report_title}[/]")
        
        # Extract themes from findings
        extracted_themes = await extract_themes(llm, state['findings'])
        
        # Generate an initial report
        initial_report = await generate_initial_report(
            llm,
            state['query'],
            state['findings'],
            extracted_themes,
            report_title,
            state['selected_sources'],
            state.get('formatted_citations', ''),
            state['current_date'],
            state['detail_level'],
            False # Don't include objective in fallback
        )
        
        # Store the initial report
        state["initial_report"] = initial_report
        
        # Enhance the report
        enhanced_report = await enhance_report(
            llm,
            initial_report,
            state['current_date'],
            state.get('formatted_citations', ''),
            state.get('selected_sources', []),
            state['sources']
        )
        
        # Store the enhanced report
        state["enhanced_report"] = enhanced_report
        
        # Expand key sections
        final_report = await expand_key_sections(
            llm,
            enhanced_report,
            extracted_themes,
            state['current_date']
        )
        
        # Get the sources that were actually analyzed and used in the research
        used_source_urls = []
        for analysis in state["content_analysis"]:
            if "sources" in analysis and isinstance(analysis["sources"], list):
                for url in analysis["sources"]:
                    if url not in used_source_urls:
                        used_source_urls.append(url)
        
        # If we don't have enough used sources, also grab from selected_sources
        if len(used_source_urls) < 5 and "selected_sources" in state:
            for url in state["selected_sources"]:
                if url not in used_source_urls:
                    used_source_urls.append(url)
                    if len(used_source_urls) >= 15:
                        break
        
        # Extract information from sources for the report
        sources_info = []
        for url in used_source_urls[:20]:  # Limit to 20 sources
            source_meta = next((s for s in state["sources"] if s.get("url") == url), {})
            sources_info.append({
                "url": url,
                "title": source_meta.get("title", ""),
                "snippet": source_meta.get("snippet", "")
            })

    # Apply comprehensive cleanup of artifacts and unwanted sections
    final_report = re.sub(r'Completed:.*?\n', '', final_report)
    final_report = re.sub(r'Here are.*?(search queries|queries to investigate).*?\n', '', final_report)
    final_report = re.sub(r'Generated search queries:.*?\n', '', final_report)
    final_report = re.sub(r'\*Generated on:.*?\*', '', final_report)
    
    # Remove entire Research Framework sections (from start to first actual content section)
    if "Research Framework:" in final_report or "# Research Framework:" in final_report:
        # Find the start of Research Framework section
        framework_matches = re.search(r'(?:^|\n)(?:#\s*)?Research Framework:.*?(?=\n#|\n\*\*|\Z)', final_report, re.DOTALL)
        if framework_matches:
            framework_section = framework_matches.group(0)
            final_report = final_report.replace(framework_section, '')
    
    # Remove "Based on our discussion" title if it exists
    final_report = re.sub(r'^(?:#\s*)?Based on our discussion,.*?\n', '', final_report, flags=re.MULTILINE)
    
    # Also try to catch Objective sections and other framework components
    final_report = re.sub(r'^Objective:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Key Aspects to Focus On:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Constraints and Preferences:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Areas to Explore in Depth:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Preferred Sources, Perspectives, or Approaches:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Scope, Boundaries, and Context:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    
    # Also remove any remaining individual problem framework lines
    final_report = re.sub(r'^Research Framework:.*?\n', '', final_report, flags=re.MULTILINE)
    final_report = re.sub(r'^Key Findings:.*?\n', '', final_report, flags=re.MULTILINE)
    final_report = re.sub(r'^Key aspects to focus on:.*?\n', '', final_report, flags=re.MULTILINE)
    
    # Ensure the report has the correctly formatted title at the beginning
    report_title = await generate_title(llm, state['query'])
    
    # Check if the report already starts with a title (# Something)
    title_match = re.match(r'^#\s+.*?\n', final_report)
    if title_match:
        # Replace existing title with our generated one
        final_report = re.sub(r'^#\s+.*?\n', f'# {report_title}\n', final_report, count=1)
    else:
        # Add our title at the beginning
        final_report = f'# {report_title}\n\n{final_report}'
        
    # Check for malformed references and fix them if needed
    if "References" in final_report:
        # Extract references section
        references_match = re.search(r'#+\s*References.*?(?=#+\s+|\Z)', final_report, re.DOTALL)
        if references_match:
            references_section = references_match.group(0)
            
            # Check if references are properly formatted
            if not re.search(r'\[\d+\]', references_section):
                console.print("[yellow]Fixing improperly formatted references...[/]")
                
                # Replace problematic references section with properly formatted one
                if state.get("formatted_citations"):
                    new_references = f"# References\n\n{state['formatted_citations']}\n"
                    final_report = final_report.replace(references_section, new_references)
                else:
                    # Generate basic references if formatted_citations isn't available
                    basic_references = []
                    for i, url in enumerate(state.get("selected_sources", []), 1):
                        source_meta = next((s for s in state["sources"] if s.get("url") == url), {})
                        title = source_meta.get("title", "Untitled")
                        source = source_meta.get("source", url.split("//")[1].split("/")[0] if "//" in url else "Unknown Source")
                        date = source_meta.get("date", "n.d.")
                        
                        citation = f"[{i}] *{source}*, \"{title}\", {date}, {url}"
                        basic_references.append(citation)
                    
                    new_references = f"# References\n\n" + "\n".join(basic_references) + "\n"
                    final_report = final_report.replace(references_section, new_references)

    elapsed_time = time.time() - state["start_time"]
    minutes, seconds = divmod(int(elapsed_time), 60)

    state["messages"].append(AIMessage(content="Research complete. Generating final report..."))
    state["findings"] = final_report
    state["status"] = "Complete"

    log_chain_of_thought(state, f"Generated final report after {minutes}m {seconds}s")
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state
