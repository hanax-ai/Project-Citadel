
"""
Planner agent for multi-agent workflows.
"""

import logging
from typing import Dict, List, Optional, Callable, Any, Type, Union

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from citadel_langgraph.nodes.agent_nodes import ReActAgentNode
from citadel_langgraph.nodes.planning_node import PlanningNode
from citadel_langgraph.state.agent_state import ReActAgentState


class PlannerAgent(ReActAgentNode):
    """
    Specialized agent for planning tasks.
    
    This agent is optimized for creating detailed plans and strategies.
    """
    
    def __init__(
        self,
        name: str,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        planning_node: Optional[PlanningNode] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the planner agent.
        
        Args:
            name: Name of the agent.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            planning_node: Planning node to use.
            tools: Tools for the agent.
            logger: Logger instance.
        """
        # Create planning node if not provided
        self.planning_node = planning_node or PlanningNode(
            name=f"{name}_planning",
            llm_manager=llm_manager,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
        )
        
        # Create system message
        system_message = (
            "You are a strategic planning specialist agent focused on creating detailed and effective plans. "
            "Your strengths include:\n"
            "1. Breaking down complex tasks into manageable steps\n"
            "2. Identifying dependencies between tasks\n"
            "3. Allocating resources efficiently\n"
            "4. Anticipating potential obstacles and creating contingency plans\n"
            "5. Prioritizing tasks based on importance and urgency\n\n"
            "When planning, always consider the overall goal and ensure that each step contributes meaningfully to it. "
            "Be thorough in your planning and provide clear, actionable steps."
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
    
    def create_plan(self, state: ReActAgentState, task: str) -> ReActAgentState:
        """
        Create a plan for a specific task.
        
        Args:
            state: Current agent state.
            task: Task to plan for.
            
        Returns:
            Updated agent state with plan.
        """
        # Update state
        updated_state = dict(state)
        
        # Add planning task to messages
        from langchain_core.messages import HumanMessage
        messages = state.get("messages", [])
        planning_message = HumanMessage(content=f"Create a detailed plan for the following task: {task}")
        updated_state["messages"] = messages + [planning_message]
        
        # Use the planning node to create a plan
        planning_state = self.planning_node(updated_state)
        
        # Execute the agent to refine the plan
        result_state = self(planning_state)
        
        return result_state
    
    def evaluate_plan(self, state: ReActAgentState, plan: List[Dict[str, Any]]) -> ReActAgentState:
        """
        Evaluate an existing plan.
        
        Args:
            state: Current agent state.
            plan: Plan to evaluate.
            
        Returns:
            Updated agent state with evaluation.
        """
        # Update state
        updated_state = dict(state)
        
        # Format the plan
        import json
        plan_str = json.dumps(plan, indent=2)
        
        # Add evaluation task to messages
        from langchain_core.messages import HumanMessage
        messages = state.get("messages", [])
        evaluation_message = HumanMessage(content=f"Evaluate the following plan and suggest improvements:\n\n{plan_str}")
        updated_state["messages"] = messages + [evaluation_message]
        
        # Execute the agent
        result_state = self(updated_state)
        
        # Extract evaluation results
        evaluation_results = {
            "original_plan": plan,
            "evaluation": result_state.get("thought"),
            "suggestions": [],
        }
        
        # Extract suggestions from thought
        thought = result_state.get("thought", "")
        if "suggest" in thought.lower() or "improve" in thought.lower():
            lines = thought.split("\n")
            suggestions = []
            for line in lines:
                if line.strip().startswith("-") or line.strip().startswith("*"):
                    suggestions.append(line.strip())
            evaluation_results["suggestions"] = suggestions
        
        # Add evaluation results to state
        result_state["evaluation_results"] = evaluation_results
        
        return result_state
