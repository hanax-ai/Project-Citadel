
"""
Planning node for agent workflows.
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


class PlanningNode(BaseNode[ReActAgentState]):
    """
    Node for agent planning.
    
    This class implements planning logic in a LangGraph workflow,
    allowing agents to plan multi-step tasks before execution.
    """
    
    def __init__(
        self,
        name: str,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        planning_template: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the planning node.
        
        Args:
            name: Name of the node.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            planning_template: Template for planning prompts.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        
        # Default planning template if none provided
        self.planning_template = planning_template or (
            "You are an AI assistant tasked with creating a detailed plan to solve a problem.\n\n"
            "Task: {task}\n\n"
            "Available tools:\n{tools}\n\n"
            "Create a step-by-step plan to accomplish this task. For each step, include:\n"
            "1. The specific action to take\n"
            "2. Which tool to use (if applicable)\n"
            "3. What information you expect to gain\n"
            "4. How this step contributes to the overall solution\n\n"
            "Your plan should be comprehensive, addressing potential challenges and alternative approaches."
        )
    
    def __call__(self, state: ReActAgentState) -> ReActAgentState:
        """
        Generate a plan for the task.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Generating plan in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get task and tools from state
            task = self._extract_task(state)
            tools = state.get("tools", [])
            
            # Format tools for planning
            formatted_tools = self._format_tools(tools)
            
            # Create planning prompt
            planning_prompt = self.planning_template.format(
                task=task,
                tools=formatted_tools
            )
            
            # Generate plan using LLM
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self.llm_manager.generate(
                        prompt=planning_prompt,
                        model_name=self.model_name,
                    )
                )
            finally:
                loop.close()
            
            # Parse the plan
            plan = self._parse_plan(result.text)
            
            # Update state with plan
            updated_state["plan"] = plan
            
            # Add plan to messages
            messages = state.get("messages", [])
            plan_message = AIMessage(content=f"I've created a plan to solve this task:\n\n{result.text}")
            updated_state["messages"] = messages + [plan_message]
            
            self.logger.info(f"Plan generated for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error generating plan: {str(e)}")
            updated_state["error"] = str(e)
        
        return updated_state
    
    def _extract_task(self, state: ReActAgentState) -> str:
        """
        Extract the task description from the state.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Task description.
        """
        # Try to find the task in the messages
        messages = state.get("messages", [])
        for message in messages:
            if hasattr(message, "type") and message.type == "human":
                return message.content
        
        # If no human message found, use a default task
        return "Complete the task based on the available information and tools."
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format tools for planning.
        
        Args:
            tools: List of tools.
            
        Returns:
            Formatted tools.
        """
        if not tools:
            return "No tools available."
        
        formatted_tools = []
        for tool in tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            formatted_tools.append(f"- {name}: {description}")
        
        return "\n".join(formatted_tools)
    
    def _parse_plan(self, plan_text: str) -> List[Dict[str, Any]]:
        """
        Parse the plan text into structured steps.
        
        Args:
            plan_text: The plan text from the LLM.
            
        Returns:
            List of plan steps.
        """
        # Simple parsing logic - in a real implementation, this would be more sophisticated
        lines = plan_text.split("\n")
        steps = []
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new step
            if line.startswith("Step") or line.startswith("1.") or line.startswith("1)"):
                if current_step:
                    steps.append(current_step)
                current_step = {"description": line, "details": []}
            elif current_step:
                current_step["details"].append(line)
        
        # Add the last step
        if current_step:
            steps.append(current_step)
        
        return steps
