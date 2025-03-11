"""Report generation utilities with structured output."""
import os
from typing import List, Dict, Optional, Any, Union
import re
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Structured output models
class ReportTitle(BaseModel):
    """Structured output for report title generation."""
    title: str = Field(description="A concise, professional title for the research report")
    
class ThemeExtraction(BaseModel):
    """Structured output for theme extraction."""
    themes: List[str] = Field(description="List of key themes identified in the research")
    descriptions: List[str] = Field(description="Brief descriptions of each theme")
    
class InitialReport(BaseModel):
    """Structured output for initial report generation."""
    executive_summary: str = Field(description="Executive summary of the research findings")
    introduction: str = Field(description="Introduction to the research topic")
    sections: List[Dict[str, str]] = Field(description="List of report sections with titles and content")
    conclusion: str = Field(description="Conclusion summarizing the key findings")
    
class EnhancedReport(BaseModel):
    """Structured output for enhanced report generation."""
    enhanced_sections: List[Dict[str, str]] = Field(description="Enhanced sections with additional details")
    
class ExpandedSection(BaseModel):
    """Structured output for section expansion."""
    section_title: str = Field(description="Title of the section being expanded")
    expanded_content: str = Field(description="Expanded content for the section")

async def generate_title(llm: ChatOpenAI, query: str) -> str:
    """Generate a professional title for the report using structured output."""
    # Use structured output for title generation
    structured_llm = llm.with_structured_output(ReportTitle, method="function_calling")
    
    # Use a direct system prompt without template variables
    system_prompt = """You are creating a professional, concise title for a research report.
    
    CRITICAL REQUIREMENTS - YOUR TITLE MUST:
    1. Be EXTREMELY CONCISE (8 words maximum)
    2. Be DESCRIPTIVE of the main topic
    3. Be PROFESSIONAL in tone
    4. NEVER start with words like "Evaluating", "Analyzing", "Assessment", "Study", "Investigation", etc.
    5. NEVER contain generic phrases like "A Comprehensive Overview", "An In-depth Look", etc.
    6. NEVER use the word "report", "research", "analysis", or similar meta-terms
    7. NEVER include prefixes like "Topic:", "Subject:", etc.
    8. NEVER be in question format
    9. NEVER be in sentence format - should be a noun phrase
    
    EXAMPLES OF GOOD TITLES:
    "NVIDIA RTX 5000 Series Gaming Performance"
    "Quantum Computing Applications in Cryptography"
    "Clean Energy Transition: Global Market Trends"
    """
    
    # Create a user message with the query directly included
    user_message = f"Create a professional, concise title (8 words max) for research about: {query}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_message)
    ])
    
    try:
        # Direct non-structured approach to avoid errors
        direct_prompt = f"""Create a professional, concise title (8 words max) for this research topic: {query}
        
        The title should be:
        - Extremely concise (8 words max)
        - Descriptive of the main topic
        - Professional in tone
        - NOT use words like "Evaluating", "Analyzing", "Assessment", "Report", etc.
        - NOT use generic phrases like "A Comprehensive Overview"
        - Be a noun phrase, not a question or sentence
        
        Return ONLY the title, nothing else. No quotation marks, no explanations.
        """
        
        # Use a direct, simplified approach
        simple_llm = llm.with_config({"temperature": 0.1})
        response = await simple_llm.ainvoke(direct_prompt)
        
        # Clean up any formatting
        clean_title = response.content.replace("Title:", "").replace("\"", "").strip()
        return clean_title
    except Exception as e:
        print(f"Error in structured title generation: {str(e)}. Using simpler approach.")
        current_file = os.path.basename(__file__)
        with open('example.txt', 'a') as file:
            # Append the current file's name and some text
            file.write(f'This line was written by: {current_file}\n')
            file.write(f'Error {e}.\n')
        # Fallback to non-structured approach
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", "Create a professional, concise title (8 words max) for a research report."),
            ("user", f"Topic: {query}")
        ])
        
        simple_llm = llm.with_config({"temperature": 0.1})
        simple_chain = simple_prompt | simple_llm
        title = await simple_chain.ainvoke({})
        
        # Clean up any formatting
        clean_title = title.content.replace("Title: ", "").replace("\"", "").strip()
        
        return clean_title

async def extract_themes(llm: ChatOpenAI, findings: str) -> str:
    """Extract key themes from research findings using structured output."""
    # Use structured output for theme extraction
    structured_llm = llm.with_structured_output(ThemeExtraction, method="function_calling")
    
    # Use a direct system prompt without template variables
    system_prompt = """You are analyzing research findings to identify key themes for a report structure.
    Extract 4-7 major themes from the content that would make logical report sections.
    These themes should emerge naturally from the content rather than following a predetermined structure.
    For each theme, provide a brief description of what content would be included."""
    
    # Create a user message with the findings directly included
    user_message = f"Analyze these research findings and extract 4-7 key themes that should be used as main sections in a report:\n\n{findings}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_message)
    ])
    
    try:
        # Direct non-structured approach to avoid errors
        direct_prompt = f"""Extract 4-7 key themes from these research findings that would make logical report sections.
        
        Research findings:
        {findings[:3000]}
        
        Format your response as:
        
        ## Theme 1
        Description of theme 1
        
        ## Theme 2
        Description of theme 2
        
        And so on. Each theme should have a clear, concise title and a brief description.
        
        Do not include any other text or explanations outside of the themes and descriptions.
        """
        
        # Use direct approach
        simple_llm = llm.with_config({"temperature": 0.3})
        response = await simple_llm.ainvoke(direct_prompt)
        
        # Return the formatted themes
        return response.content
    except Exception as e:
        print(f"Error in structured theme extraction: {str(e)}. Using simpler approach.")
        
        # Fallback to non-structured approach
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract 4-7 key themes from research findings that would make logical report sections.
            Format your response as:
            
            ## Theme 1
            Description of theme 1
            
            ## Theme 2
            Description of theme 2
            
            And so on."""),
            ("user", f"Extract themes from these findings:\n\n{findings[:10000]}")
        ])
        
        simple_llm = llm.with_config({"temperature": 0.3})
        simple_chain = simple_prompt | simple_llm
        themes = await simple_chain.ainvoke({})
        
        return themes.content

async def format_citations(llm: ChatOpenAI, selected_sources: List[str], sources: List[Dict[str, Any]]) -> str:
    """Format citations for selected sources."""
    if not selected_sources:
        return ""
    
    # Format the sources for citation
    sources_text = ""
    for i, url in enumerate(selected_sources, 1):
        # Find the source metadata if available
        source_meta = {}
        for source in sources:
            if source.get("url") == url:
                source_meta = source
                break
        
        # Add source information to the text
        sources_text += f"Source {i}:\nURL: {url}\n"
        if source_meta.get("title"):
            sources_text += f"Title: {source_meta.get('title')}\n"
        if source_meta.get("source"):
            sources_text += f"Publication: {source_meta.get('source')}\n"
        if source_meta.get("date"):
            sources_text += f"Date: {source_meta.get('date')}\n"
        sources_text += "\n"
    
    # Use an enhanced citation formatter prompt with direct system message
    system_prompt = """Format the following source information into properly numbered citations for a research report.
    
    FORMATTING REQUIREMENTS:
    1. Each citation MUST start with [n] where n is the citation number
    2. Each citation MUST include the following elements (when available):
       - Publication name or website name in italics
       - Author(s) with full names when available
       - Title of the article/page in quotes
       - Publication date in format: YYYY-MM-DD
       - URL
    
    EXAMPLE PROPER CITATIONS:
    [1] *TechReview*, John Smith, "Advances in GPU Architecture", 2024-01-15, https://techreview.com/articles/gpu-advances
    [2] *ArXiv*, Zhang et al., "Neural Network Performance Optimization", 2023-11-30, https://arxiv.org/papers/nn-optimization
    
    MISSING INFORMATION:
    - If publication name is missing, use the domain name from the URL
    - If author is missing, omit this element
    - If date is missing, use "n.d." (no date)
    
    FORMAT ALL CITATIONS IN A CONSISTENT STYLE.
    Number citations sequentially starting from [1].
    Place each citation on a new line.
    """
    
    # Create a user message with the sources directly included
    user_message = f"Format these sources into proper citations:\n\n{sources_text}"
    
    citation_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_message)
    ])
    
    formatter_chain = citation_prompt | llm
    formatted_citations = (await formatter_chain.ainvoke({})).content
    
    # Verify citations have proper format [n] at the beginning
    if not re.search(r'\[\d+\]', formatted_citations):
        # Add fallback formatting if needed
        citations = []
        for i, url in enumerate(selected_sources, 1):
            source_meta = next((s for s in sources if s.get("url") == url), {})
            title = source_meta.get("title", "Untitled")
            source = source_meta.get("source", url.split("//")[1].split("/")[0] if "//" in url else "Unknown Source")
            date = source_meta.get("date", "n.d.")
            
            citation = f"[{i}] *{source}*, \"{title}\", {date}, {url}"
            citations.append(citation)
        
        formatted_citations = "\n".join(citations)
    
    return formatted_citations

async def generate_initial_report(
    llm: ChatOpenAI,
    query: str,
    findings: str,
    extracted_themes: str,
    report_title: str,
    selected_sources: List[str],
    formatted_citations: str,
    current_date: str,
    detail_level: str,
    include_objective: bool
) -> str:
    """Generate the initial report draft using structured output."""
    # Use structured output for initial report generation
    structured_llm = llm.with_structured_output(InitialReport, method="function_calling")
    
    # Add objective instruction based on include_objective flag
    objective_instruction = ""
    if not include_objective:
        objective_instruction = "\n\nIMPORTANT: DO NOT include an \"Objective\" section at the beginning of the report. Let your content and analysis naturally determine the structure."
    
    # Use a direct system prompt without template variables
    system_prompt = f"""You are generating a comprehensive research report based on extensive research.
    
    REPORT REQUIREMENTS:
    - The report should be thorough, detailed, and professionally formatted in Markdown.
    - Include headers, subheaders, and formatting for readability.
    - The level of detail should be {detail_level.upper()}.
    - Base the report ENTIRELY on the provided research findings.
    - As of {current_date}, incorporate the most up-to-date information available.
    - Include proper citations to sources throughout the text using [n] format.
    - Create a dynamic structure based on the content themes rather than a rigid template.{objective_instruction}
    """
    
    # Create a user message with all variables directly included
    user_message = f"""Create an extensive, in-depth research report on this topic.

Title: {report_title}
Analyzed Findings: {findings[:5000]}
Number of sources: {len(selected_sources)}
Key themes identified in the research: 
{extracted_themes}

Organize your report around these key themes that naturally emerged from the research.
Create a dynamic, organic structure that best presents the findings, rather than forcing content into predetermined sections.
Ensure comprehensive coverage while maintaining a logical flow between topics.

Your report must be extensive, detailed, and grounded in the research. Include all relevant data, examples, and insights found in the research.
Use proper citations to the sources throughout.

IMPORTANT: Begin your report with the exact title provided: "{report_title}" - do not modify or rephrase it."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_message)
    ])
    
    # Format the selected sources for the report
    sources_text = "\n\nSOURCES ANALYZED IN DETAIL:\n"
    if formatted_citations:
        sources_text += formatted_citations
    else:
        for i, url in enumerate(selected_sources, 1):
            sources_text += f"{i}. {url}\n"
    
    # Augment findings with selected source information
    augmented_findings = findings + sources_text
    
    # Generate the initial detailed report
    try:
        # Direct non-structured approach to avoid errors
        direct_prompt = f"""Create a comprehensive research report on this topic.

Title: {report_title}
Analyzed Findings: {findings[:5000]}
Number of sources: {len(selected_sources)}
Key themes identified in the research: 
{extracted_themes}

REPORT REQUIREMENTS:
- The report should be thorough, detailed, and professionally formatted in Markdown.
- Include headers, subheaders, and formatting for readability.
- The level of detail should be {detail_level.upper()}.
- Base the report ENTIRELY on the provided research findings.
- As of {current_date}, incorporate the most up-to-date information available.
- Include proper citations to sources throughout the text using [n] format.
- Create a dynamic structure based on the content themes rather than a rigid template.

FORMAT:
- Start with the title as a level 1 heading: "# {report_title}"
- Include an executive summary
- Include an introduction
- Include sections based on the key themes
- Include a conclusion
- Include references section

DO NOT include a "Research Framework" or "Objective" section at the beginning.
"""
        
        # Use direct approach with higher token limit
        report_llm = llm.with_config({"max_tokens": 16384, "temperature": 0.2})
        response = await report_llm.ainvoke(direct_prompt)
        
        return response.content
        
    except Exception as e:
        print(f"Error in structured report generation: {str(e)}. Using simpler approach.")
        current_file = os.path.basename(__file__)
        with open('example.txt', 'a') as file:
            # Append the current file's name and some text
            file.write(f'This line was written by: {current_file}\n')
            file.write(f'Error {e}.\n')
        # Fallback to simpler report generation without structured output
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Generate a comprehensive research report based on the provided findings.
            The report should be well-structured with clear sections and proper citations.
            Current date: {current_date}"""),
            ("user", f"Title: {report_title}\n\nFindings: {augmented_findings[:10000]}")
        ])
        
        simple_llm = llm.with_config({"max_tokens": 8192})
        simple_chain = simple_prompt | simple_llm
        report = await simple_chain.ainvoke({})
        
        return report.content

async def enhance_report(
    llm: ChatOpenAI, 
    initial_report: str, 
    current_date: str, 
    formatted_citations: str, 
    selected_sources: List, 
    sources: List[Dict]
) -> str:
    """Enhance the report with additional detail using structured output."""
    # Use structured output for report enhancement
    structured_llm = llm.with_structured_output(EnhancedReport, method="function_calling")
    
    # Extract sections from the initial report
    sections = re.findall(r'## ([^\n]+)\n\n([^#]+)', initial_report, re.DOTALL)
    
    if not sections:
        return initial_report
    
    # Create a list of sections for the structured output
    section_list = []
    for title, content in sections:
        section_list.append({"title": title, "content": content.strip()})
    
    # Use a direct system prompt without template variables
    system_prompt = f"""You are enhancing a research report with additional depth, detail and clarity.
    
    Your task is to:
    1. Add more detailed explanations to key concepts
    2. Expand on examples and case studies
    3. Enhance the analysis and interpretation of findings
    4. Improve the overall structure and flow
    5. Add relevant statistics, data points, or evidence from the sources
    6. Ensure proper citation [n] format throughout
    7. Maintain scientific accuracy and up-to-date information (current as of {current_date})
    
    DO NOT add information not supported by the research. Focus on enhancing what's already there.
    """
    
    # Create a user message with the report directly included
    user_message = f"""Enhance this research report with additional depth and detail:

{initial_report[:10000]}

Make it more comprehensive, rigorous, and valuable to readers while maintaining scientific accuracy.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_message)
    ])
    
    try:
        # Direct non-structured approach
        direct_prompt = f"""Enhance this research report with additional depth and detail:

{initial_report[:10000]}

Your task is to:
1. Add more detailed explanations to key concepts
2. Expand on examples and case studies
3. Enhance the analysis and interpretation of findings
4. Improve the overall structure and flow
5. Add relevant statistics, data points, or evidence 
6. Ensure proper citation [n] format throughout
7. Maintain scientific accuracy and up-to-date information (current as of {current_date})

Return the complete enhanced report, maintaining the original title and structure but with expanded content.
"""
        
        # Use direct approach with high token limit
        enhance_llm = llm.with_config({"max_tokens": 16384, "temperature": 0.3})
        response = await enhance_llm.ainvoke(direct_prompt)
        
        # Extract the title from the initial report to ensure consistency
        title_match = re.match(r'# ([^\n]+)', initial_report)
        if title_match and not response.content.strip().startswith("# "):
            title = title_match.group(1)
            return f"# {title}\n\n{response.content}"
        
        return response.content
        
    except Exception as e:
        current_file = os.path.basename(__file__)
        with open('example.txt', 'a') as file:
            # Append the current file's name and some text
            file.write(f'This line was written by: {current_file}\n')
            file.write(f'Error {e}.\n')
        print(f"Error in structured report enhancement: {str(e)}. Returning initial report.")
        return initial_report

async def expand_key_sections(
    llm: ChatOpenAI, 
    report: str, 
    identified_themes: str, 
    current_date: str
) -> str:
    """Expand key sections of the report using structured output."""
    # Make sure we have a properly formatted report to start with
    if not report or len(report.strip()) < 1000:
        return report
    
    # Extract sections from the report
    sections = re.findall(r'## ([^\n]+)\n\n([^#]+)', report, re.DOTALL)
    
    if not sections:
        return report
    
    # Select 2-3 most important sections to expand
    important_sections = []
    for title, content in sections:
        # Skip executive summary, introduction, conclusion, and references
        if title.lower() in ["executive summary", "introduction", "conclusion", "references"]:
            continue
        important_sections.append((title, content))
    
    # Limit to 3 sections maximum
    important_sections = important_sections[:3]
    
    if not important_sections:
        return report
    
    # Use structured output for section expansion
    structured_llm = llm.with_structured_output(ExpandedSection, method="function_calling")
    
    # Expand each section
    expanded_report = report
    for title, content in important_sections:
        # Use a direct system prompt without template variables
        system_prompt = f"""You are expanding a key section of a research report with additional depth and detail.
        
        EXPANSION REQUIREMENTS:
        1. Triple the length and detail of the section while maintaining accuracy
        2. Add specific examples, case studies, or data points to support claims
        3. Include additional context and background information
        4. Add nuance, caveats, and alternative perspectives
        5. Use proper citation format [n] throughout
        6. Maintain the existing section structure but add subsections if appropriate
        7. Ensure all information is accurate as of {current_date}
        """
        
        # Create a user message with the section directly included
        user_message = f"""Expand this section with much greater depth and detail:

## {title}

{content}

Make it substantially more comprehensive while maintaining accuracy and relevance.
Keep the original section heading but expand everything underneath it.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_message)
        ])
        
        try:
            # Direct non-structured approach
            direct_prompt = f"""Expand this section of a research report with much greater depth and detail:

## {title}

{content}

EXPANSION REQUIREMENTS:
1. Triple the length and detail of the section while maintaining accuracy
2. Add specific examples, case studies, or data points to support claims
3. Include additional context and background information
4. Add nuance, caveats, and alternative perspectives
5. Use proper citation format [n] throughout
6. Maintain the existing section structure but add subsections if appropriate
7. Ensure all information is accurate as of {current_date}

Return only the expanded section with the original heading.
"""
            
            # Use direct approach
            expand_llm = llm.with_config({"max_tokens": 8192, "temperature": 0.3})
            response = await expand_llm.ainvoke(direct_prompt)
            
            # Format the response to ensure it starts with the section title
            expanded_content = response.content
            if not expanded_content.strip().startswith("## "):
                expanded_content = f"## {title}\n\n{expanded_content}"
            
            # Replace the original section with the expanded one
            section_pattern = re.compile(f"## {re.escape(title)}\n\n[^#]+", re.DOTALL)
            expanded_report = section_pattern.sub(expanded_content, expanded_report)
            
        except Exception as e:
            print(f"Error expanding section '{title}': {str(e)}")
            # Continue with other sections if one fails
    
    return expanded_report
