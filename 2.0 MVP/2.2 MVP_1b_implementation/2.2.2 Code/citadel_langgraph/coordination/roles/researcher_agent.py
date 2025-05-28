
"""
Researcher agent for multi-agent workflows.
"""

import logging
from typing import Dict, List, Optional, Callable, Any, Type, Union

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from citadel_langgraph.nodes.agent_nodes import ReActAgentNode
from citadel_langgraph.state.agent_state import ReActAgentState
from citadel_langgraph.tools.web_search_tool import WebSearchTool


class ResearcherAgent(ReActAgentNode):
    """
    Specialized agent for research tasks.
    
    This agent is optimized for gathering and analyzing information.
    """
    
    def __init__(
        self,
        name: str,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        web_search_tool: Optional[WebSearchTool] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the researcher agent.
        
        Args:
            name: Name of the agent.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            web_search_tool: Web search tool to use.
            additional_tools: Additional tools for the agent.
            logger: Logger instance.
        """
        # Create default web search tool if not provided
        if not web_search_tool:
            web_search_tool = WebSearchTool()
        
        # Create tools list
        tools = [web_search_tool.to_dict()]
        if additional_tools:
            tools.extend(additional_tools)
        
        # Create system message
        system_message = (
            "You are a research specialist agent focused on gathering accurate and comprehensive information. "
            "Your strengths include:\n"
            "1. Finding relevant information from multiple sources\n"
            "2. Evaluating the credibility of sources\n"
            "3. Synthesizing information into clear summaries\n"
            "4. Identifying gaps in available information\n"
            "5. Formulating effective search queries\n\n"
            "When researching, always consider multiple perspectives and verify facts across sources. "
            "Be thorough in your research and provide well-organized findings."
        )
        
        super().__init__(
            name=name,
            llm_manager=llm_manager,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            system_message=system_message,
            tools=tools,
            reasoning_steps=3,
            logger=logger or get_logger(f"citadel.langgraph.agents.{name}"),
        )
    
    def research_topic(self, state: ReActAgentState, topic: str) -> ReActAgentState:
        """
        Research a specific topic.
        
        Args:
            state: Current agent state.
            topic: Topic to research.
            
        Returns:
            Updated agent state with research results.
        """
        # Update state
        updated_state = dict(state)
        
        # Add research task to messages
        from langchain_core.messages import HumanMessage
        messages = state.get("messages", [])
        research_message = HumanMessage(content=f"Research the following topic: {topic}")
        updated_state["messages"] = messages + [research_message]
        
        # Execute the agent
        result_state = self(updated_state)
        
        # Extract research results
        research_results = {
            "topic": topic,
            "findings": result_state.get("thought"),
            "sources": [],
        }
        
        # Extract sources from tool results
        for tool_result in result_state.get("tool_results", []):
            if tool_result.get("tool") == "web_search":
                research_results["sources"].extend(tool_result.get("result", []))
        
        # Add research results to state
        result_state["research_results"] = research_results
        
        return result_state
