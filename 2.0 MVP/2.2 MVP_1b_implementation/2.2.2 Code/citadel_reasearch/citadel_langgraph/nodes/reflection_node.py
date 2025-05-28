
"""
Reflection node for agent workflows.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    BaseMessage,
)

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from citadel_langgraph.state.agent_state import ReActAgentState
from citadel_langgraph.nodes.base import BaseNode


class ReflectionNode(BaseNode[ReActAgentState]):
    """
    Node for agent reflection.
    
    This class implements reflection logic in a LangGraph workflow,
    allowing agents to reflect on their actions and improve their reasoning.
    """
    
    def __init__(
        self,
        name: str,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        reflection_template: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the reflection node.
        
        Args:
            name: Name of the node.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            reflection_template: Template for reflection prompts.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        
        # Default reflection template if none provided
        self.reflection_template = reflection_template or (
            "You are an AI assistant reflecting on your previous actions and reasoning.\n\n"
            "Review the following steps you've taken so far:\n\n"
            "{step_history}\n\n"
            "Based on this review, consider:\n"
            "1. What went well in your reasoning and actions?\n"
            "2. What could be improved?\n"
            "3. Are there any patterns or mistakes you notice?\n"
            "4. What alternative approaches could you have taken?\n"
            "5. What lessons can you apply to future steps?\n\n"
            "Provide a thoughtful reflection that will help improve future reasoning and actions."
        )
    
    def __call__(self, state: ReActAgentState) -> ReActAgentState:
        """
        Generate a reflection on the agent's actions.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Generating reflection in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get step history from state
            step_history = state.get("step_history", [])
            
            if not step_history:
                self.logger.info("No step history to reflect on")
                updated_state["reflection"] = "No actions to reflect on yet."
                return updated_state
            
            # Format step history for reflection
            formatted_history = self._format_step_history(step_history)
            
            # Create reflection prompt
            reflection_prompt = self.reflection_template.format(
                step_history=formatted_history
            )
            
            # Generate reflection using LLM
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self.llm_manager.generate(
                        prompt=reflection_prompt,
                        model_name=self.model_name,
                    )
                )
            finally:
                loop.close()
            
            # Update state with reflection
            updated_state["reflection"] = result.text
            
            # Add reflection to reflections history
            reflections = state.get("reflections", [])
            reflections.append({
                "timestamp": state.get("updated_at"),
                "reflection": result.text,
            })
            updated_state["reflections"] = reflections
            
            self.logger.info(f"Reflection generated for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error generating reflection: {str(e)}")
            updated_state["error"] = str(e)
        
        return updated_state
    
    def _format_step_history(self, step_history: List[Dict[str, Any]]) -> str:
        """
        Format step history for reflection.
        
        Args:
            step_history: List of steps taken by the agent.
            
        Returns:
            Formatted step history.
        """
        formatted_steps = []
        
        for i, step in enumerate(step_history):
            step_num = i + 1
            thought = step.get("thought", "")
            action = step.get("action", "")
            action_input = step.get("action_input", {})
            observation = step.get("observation", "")
            
            formatted_step = f"Step {step_num}:\n"
            formatted_step += f"Thought: {thought}\n"
            
            if action:
                formatted_step += f"Action: {action}\n"
                
                if action_input:
                    import json
                    formatted_step += f"Action Input: {json.dumps(action_input)}\n"
                
                if observation:
                    formatted_step += f"Observation: {observation}\n"
            
            formatted_steps.append(formatted_step)
        
        return "\n".join(formatted_steps)
