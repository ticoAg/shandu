"""
Reflection node for research graph.
"""
import os
from rich.console import Console
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from ..processors.content_processor import AgentState
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback
from ...prompts import SYSTEM_PROMPTS, USER_PROMPTS

console = Console()

# Structured output model for reflection
class ResearchReflection(BaseModel):
    """Structured output for research reflection."""
    key_insights: list[str] = Field(
        description="Key insights gained from the research so far",
        min_items=1
    )
    knowledge_gaps: list[str] = Field(
        description="Identified gaps in the current research",
        min_items=1
    )
    next_steps: list[str] = Field(
        description="Recommended next steps for the research",
        min_items=1
    )
    reflection_summary: str = Field(
        description="Overall reflection on the current state of the research"
    )

async def reflect_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Reflect on current findings to identify gaps and opportunities using structured output."""
    state["status"] = "Reflecting on findings"
    console.print("[bold yellow]Reflecting on current findings...[/]")
    
    try:
        # Use a completely direct approach to avoid template issues
        direct_prompt = f"""Analyze the following research findings and provide a detailed reflection. Today's date: {state['current_date']}

Research Findings:
{state['findings'][:3000]}

Your reflection must include these sections clearly labeled:

## Key Insights
- List the most important discoveries and insights from the research
- Evaluate the evidence strength for each insight

## Knowledge Gaps
- Identify specific questions that remain unanswered
- Explain why these gaps are significant

## Next Steps
- Suggest specific areas for deeper investigation
- Recommend research methods to address the knowledge gaps

## Overall Reflection
- Provide a comprehensive assessment of the research progress
- Evaluate the overall quality and reliability of the findings

Format your response with clear section headings and bullet points for clarity.
"""
        # Send the prompt directly to the model
        response = await llm.ainvoke(direct_prompt)
        
        # Process the response directly
        reflection_text = response.content
        
        # Extract key sections manually
        import re
        key_insights = []
        knowledge_gaps = []
        next_steps = []
        reflection_summary = ""
        
        # Extract insights
        insights_section = re.search(r'(?:key insights|insights|key findings)(?:\s*:|\s*\n)([^#]*?)(?:#|$)', reflection_text.lower(), re.IGNORECASE | re.DOTALL)
        if insights_section:
            insights_text = insights_section.group(1).strip()
            key_insights = [line.strip().strip('-*').strip() for line in insights_text.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        # Extract gaps
        gaps_section = re.search(r'(?:knowledge gaps|gaps|questions|unanswered questions)(?:\s*:|\s*\n)([^#]*?)(?:#|$)', reflection_text.lower(), re.IGNORECASE | re.DOTALL)
        if gaps_section:
            gaps_text = gaps_section.group(1).strip()
            knowledge_gaps = [line.strip().strip('-*').strip() for line in gaps_text.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        # Extract next steps
        steps_section = re.search(r'(?:next steps|steps|recommendations|future directions)(?:\s*:|\s*\n)([^#]*?)(?:#|$)', reflection_text.lower(), re.IGNORECASE | re.DOTALL)
        if steps_section:
            steps_text = steps_section.group(1).strip()
            next_steps = [line.strip().strip('-*').strip() for line in steps_text.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        # Extract summary
        summary_section = re.search(r'(?:overall reflection|reflection summary|summary|conclusion)(?:\s*:|\s*\n)([^#]*?)(?:#|$)', reflection_text.lower(), re.IGNORECASE | re.DOTALL)
        if summary_section:
            reflection_summary = summary_section.group(1).strip()
        
        # Ensure we have content for each section
        if not key_insights:
            key_insights = ["Research is progressing on " + state['query']]
        if not knowledge_gaps:
            knowledge_gaps = ["Further details needed on specific aspects"]
        if not next_steps:
            next_steps = ["Continue investigating primary aspects", "Search for more specific examples"]
        if not reflection_summary:
            reflection_summary = "The research is making progress and has uncovered valuable information, but further investigation is needed in key areas."
        
        # Format the reflection into a well-structured text
        formatted_reflection = "## Key Insights\n\n"
        for insight in key_insights:
            formatted_reflection += f"- {insight}\n"
        
        formatted_reflection += "\n## Knowledge Gaps\n\n"
        for gap in knowledge_gaps:
            formatted_reflection += f"- {gap}\n"
        
        formatted_reflection += "\n## Next Steps\n\n"
        for step in next_steps:
            formatted_reflection += f"- {step}\n"
        
        formatted_reflection += f"\n## Overall Reflection\n\n{reflection_summary}\n"
        
        state["messages"].append(HumanMessage(content="Analyzing current findings..."))
        state["messages"].append(AIMessage(content=formatted_reflection))
        state["findings"] += f"\n\n## Reflection on Current Findings\n\n{formatted_reflection}\n\n"
        
    except Exception as e:
        console.print(f"[dim red]Error in structured reflection: {str(e)}. Using simpler approach.[/dim red]")
        current_file = os.path.basename(__file__)
        with open('example.txt', 'a') as file:
            # Append the current file's name and some text
            file.write(f'This line was written by: {current_file}\n')
            file.write(f'Error {e}.\n')

        # Fallback to non-structured approach
        try:
            # Even simpler fallback approach
            response = await llm.ainvoke(f"""Reflect on these research findings:

{state['findings'][:2000]}

Include: 
1. Key insights
2. Knowledge gaps
3. Next steps
4. Overall assessment
""")
            
            reflection_content = response.content
            
            state["messages"].append(HumanMessage(content="Analyzing current findings..."))
            state["messages"].append(AIMessage(content=reflection_content))
            state["findings"] += f"\n\n## Reflection on Current Findings\n\n{reflection_content}\n\n"
        except Exception as e2:
            console.print(f"[dim red]Error in fallback reflection: {str(e2)}. Using minimal reflection.[/dim red]")
            
            # Create a minimal reflection if all else fails
            minimal_reflection = "## Research Reflection\n\nThe research is progressing. Further investigation is needed to develop a more comprehensive understanding of the topic."
            
            state["messages"].append(HumanMessage(content="Analyzing current findings..."))
            state["messages"].append(AIMessage(content=minimal_reflection))
            state["findings"] += f"\n\n## Reflection on Current Findings\n\n{minimal_reflection}\n\n"
    
    log_chain_of_thought(state, "Completed reflection on current findings")
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state
