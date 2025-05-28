
"""
Critic agent for multi-agent workflows.
"""

import logging
from typing import Dict, List, Optional, Callable, Any, Type, Union

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from citadel_langgraph.nodes.agent_nodes import ReActAgentNode
from citadel_langgraph.nodes.reflection_node import ReflectionNode
from citadel_langgraph.state.agent_state import ReActAgentState


class CriticAgent(ReActAgentNode):
    """
    Specialized agent for evaluating and critiquing.
    
    This agent is optimized for providing feedback and identifying improvements.
    """
    
    def __init__(
        self,
        name: str,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        reflection_node: Optional[ReflectionNode] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the critic agent.
        
        Args:
            name: Name of the agent.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            reflection_node: Reflection node to use.
            tools: Tools for the agent.
            logger: Logger instance.
        """
        # Create reflection node if not provided
        self.reflection_node = reflection_node or ReflectionNode(
            name=f"{name}_reflection",
            llm_manager=llm_manager,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
        )
        
        # Create system message
        system_message = (
            "You are a critical analysis specialist agent focused on evaluating and improving work. "
            "Your strengths include:\n"
            "1. Identifying logical flaws and inconsistencies\n"
            "2. Evaluating the quality and completeness of work\n"
            "3. Providing constructive feedback\n"
            "4. Suggesting specific improvements\n"
            "5. Considering multiple perspectives\n\n"
            "When critiquing, always be constructive and specific. "
            "Focus on how things can be improved rather than just pointing out flaws. "
            "Provide clear, actionable feedback that can lead to meaningful improvements."
        )
        
        super().__init__(
            name=name,
            llm_manager=llm_manager,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            system_message=system_message,
            tools=tools or [],
            reasoning_steps=3,
            logger=logger or get_logger(f"citadel.langgraph.agents.{name}"),
        )
    
    def critique_work(self, state: ReActAgentState, work: str, criteria: Optional[List[str]] = None) -> ReActAgentState:
        """
        Critique a piece of work.
        
        Args:
            state: Current agent state.
            work: Work to critique.
            criteria: Specific criteria to evaluate.
            
        Returns:
            Updated agent state with critique.
        """
        # Update state
        updated_state = dict(state)
        
        # Format criteria
        criteria_str = ""
        if criteria:
            criteria_str = "Evaluate based on the following criteria:\n"
            for criterion in criteria:
                criteria_str += f"- {criterion}\n"
        
        # Add critique task to messages
        from langchain_core.messages import HumanMessage
        messages = state.get("messages", [])
        critique_message = HumanMessage(content=f"Critique the following work:\n\n{work}\n\n{criteria_str}")
        updated_state["messages"] = messages + [critique_message]
        
        # Execute the agent
        result_state = self(updated_state)
        
        # Extract critique results
        critique_results = {
            "work": work,
            "critique": result_state.get("thought"),
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
        }
        
        # Parse the critique to extract strengths, weaknesses, and suggestions
        thought = result_state.get("thought", "")
        
        # Simple parsing logic - in a real implementation, this would be more sophisticated
        lines = thought.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "strength" in line.lower() or "positive" in line.lower():
                current_section = "strengths"
            elif "weakness" in line.lower() or "negative" in line.lower() or "issue" in line.lower():
                current_section = "weaknesses"
            elif "suggestion" in line.lower() or "improvement" in line.lower() or "recommend" in line.lower():
                current_section = "suggestions"
            elif current_section and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                critique_results[current_section].append(line)
        
        # Add critique results to state
        result_state["critique_results"] = critique_results
        
        return result_state
    
    def reflect_on_process(self, state: ReActAgentState) -> ReActAgentState:
        """
        Reflect on the process and identify improvements.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated agent state with reflection.
        """
        # Use the reflection node
        reflection_state = self.reflection_node(state)
        
        # Execute the agent to provide additional insights
        result_state = self(reflection_state)
        
        return result_state
