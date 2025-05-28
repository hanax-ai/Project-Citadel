
"""
Executor agent for multi-agent workflows.
"""

import logging
from typing import Dict, List, Optional, Callable, Any, Type, Union

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from citadel_langgraph.nodes.agent_nodes import ReActAgentNode, ToolExecutionNode
from citadel_langgraph.state.agent_state import ReActAgentState
from citadel_langgraph.tools.tool_registry import ToolRegistry


class ExecutorAgent(ReActAgentNode):
    """
    Specialized agent for executing tasks.
    
    This agent is optimized for executing plans and performing actions.
    """
    
    def __init__(
        self,
        name: str,
        tool_registry: ToolRegistry,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        tool_execution_node: Optional[ToolExecutionNode] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the executor agent.
        
        Args:
            name: Name of the agent.
            tool_registry: Tool registry to use.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            tool_execution_node: Tool execution node to use.
            logger: Logger instance.
        """
        # Get tools from registry
        tools = tool_registry.get_tool_descriptions()
        
        # Create tool execution node if not provided
        self.tool_execution_node = tool_execution_node or ToolExecutionNode(
            name=f"{name}_tool_execution",
            tool_registry={
                tool_name: tool_registry.get_tool(tool_name)
                for tool_name in tool_registry.get_all_tools()
            },
        )
        
        # Create system message
        system_message = (
            "You are an execution specialist agent focused on efficiently carrying out tasks and plans. "
            "Your strengths include:\n"
            "1. Following instructions precisely\n"
            "2. Using tools effectively to accomplish tasks\n"
            "3. Adapting to unexpected situations\n"
            "4. Providing clear status updates\n"
            "5. Completing tasks efficiently\n\n"
            "When executing tasks, always follow the plan but be ready to adapt if necessary. "
            "Use the most appropriate tools for each step and provide clear feedback on your progress."
        )
        
        super().__init__(
            name=name,
            llm_manager=llm_manager,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            system_message=system_message,
            tools=tools,
            reasoning_steps=2,
            logger=logger or get_logger(f"citadel.langgraph.agents.{name}"),
        )
        
        self.tool_registry = tool_registry
    
    def execute_plan(self, state: ReActAgentState, plan: List[Dict[str, Any]]) -> ReActAgentState:
        """
        Execute a plan.
        
        Args:
            state: Current agent state.
            plan: Plan to execute.
            
        Returns:
            Updated agent state with execution results.
        """
        # Update state
        updated_state = dict(state)
        
        # Format the plan
        import json
        plan_str = json.dumps(plan, indent=2)
        
        # Add execution task to messages
        from langchain_core.messages import HumanMessage
        messages = state.get("messages", [])
        execution_message = HumanMessage(content=f"Execute the following plan step by step:\n\n{plan_str}")
        updated_state["messages"] = messages + [execution_message]
        
        # Execute each step of the plan
        execution_results = []
        
        for i, step in enumerate(plan):
            # Update current step
            updated_state["current_step"] = f"execute_step_{i+1}"
            
            # Execute the agent to determine the action
            agent_state = self(updated_state)
            
            # If an action was determined, execute it
            if agent_state.get("action"):
                # Execute the tool
                tool_state = self.tool_execution_node(agent_state)
                
                # Add result to execution results
                execution_results.append({
                    "step": i + 1,
                    "description": step.get("description", ""),
                    "action": tool_state.get("action"),
                    "result": tool_state.get("observation"),
                })
                
                # Update state for next step
                updated_state = tool_state
            else:
                # If no action, add to results and continue
                execution_results.append({
                    "step": i + 1,
                    "description": step.get("description", ""),
                    "action": None,
                    "result": "No action taken",
                })
        
        # Add execution results to state
        updated_state["execution_results"] = execution_results
        updated_state["current_step"] = "finish"
        
        return updated_state
    
    def execute_action(self, state: ReActAgentState, action: str, action_input: Dict[str, Any]) -> ReActAgentState:
        """
        Execute a specific action.
        
        Args:
            state: Current agent state.
            action: Action to execute.
            action_input: Input for the action.
            
        Returns:
            Updated agent state with execution result.
        """
        # Update state
        updated_state = dict(state)
        updated_state["action"] = action
        updated_state["action_input"] = action_input
        
        # Execute the tool
        result_state = self.tool_execution_node(updated_state)
        
        return result_state
