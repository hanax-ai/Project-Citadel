
"""
Agent nodes for LangGraph workflows.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union
import uuid

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.language_models import BaseLanguageModel
from datetime import datetime

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from citadel_langgraph.state.base import BaseState
from citadel_langgraph.state.agent_state import (
    AgentState,
    ReActAgentState,
    MultiAgentState,
)
from citadel_langgraph.state.ui_state import (
    UIState,
    UIAgentState,
    add_message_with_events,
    emit_state_update
)
from citadel_langgraph.events.emitter import (
    emit_tool_call_start,
    emit_tool_call_args,
    emit_tool_call_end,
    emit_message_start,
    emit_message_content,
    emit_message_end,
    emit_event
)
from citadel_frontend.protocol.ag_ui import EventType
from .base import BaseNode, FunctionNode


class LLMNode(BaseNode[AgentState]):
    """
    Node for interacting with an LLM.
    
    This class implements LLM interaction logic in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        system_message: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the LLM node.
        
        Args:
            name: Name of the node.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            system_message: System message to use.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        self.system_message = system_message
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Generate a response from the LLM.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Generating LLM response in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get messages from state
            messages = state.get("messages", [])
            
            # Add system message if not present
            if self.system_message and not any(
                isinstance(msg, SystemMessage) for msg in messages
            ):
                messages = [SystemMessage(content=self.system_message)] + messages
            
            # Format messages for the LLM
            formatted_messages = self._format_messages(messages)
            
            # Check if we should emit events
            should_emit_events = isinstance(state, UIState) and state.get("emit_events", True)
            
            # Start message event if needed
            message_id = None
            if should_emit_events:
                message_id = emit_message_start(sender="agent")
            
            # Generate response using LLM
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self.llm_manager.generate(
                        prompt=formatted_messages,
                        model_name=self.model_name,
                    )
                )
            finally:
                loop.close()
            
            # Add AI message to messages
            ai_message = AIMessage(content=result.text)
            
            # Emit message content if needed
            if should_emit_events and message_id:
                emit_message_content(message_id, result.text)
                emit_message_end(message_id)
                
                # Add message with events if it's a UI state
                if isinstance(state, UIState):
                    updated_state = add_message_with_events(state, ai_message, emit_events=False)
                else:
                    updated_state["messages"] = messages + [ai_message]
            else:
                updated_state["messages"] = messages + [ai_message]
            
            self.logger.info(f"LLM response generated for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {str(e)}")
            updated_state["error"] = str(e)
            
            # Emit error event if needed
            if isinstance(state, UIState) and state.get("emit_events", True):
                emit_event(EventType.ERROR, {
                    "errorCode": "llm_error",
                    "message": str(e)
                })
        
        return updated_state
    
    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """
        Format messages for the LLM.
        
        Args:
            messages: List of messages.
            
        Returns:
            Formatted messages.
        """
        formatted_parts = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                formatted_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_parts.append(f"AI: {msg.content}")
            elif isinstance(msg, FunctionMessage):
                formatted_parts.append(f"Function ({msg.name}): {msg.content}")
            elif isinstance(msg, ToolMessage):
                formatted_parts.append(f"Tool ({msg.tool_call_id}): {msg.content}")
            else:
                formatted_parts.append(f"Message: {msg.content}")
        
        return "\n\n".join(formatted_parts)


class ReActAgentNode(BaseNode[ReActAgentState]):
    """
    Node for ReAct agent execution with enhanced step-by-step reasoning.
    
    This class implements ReAct agent logic in a LangGraph workflow with
    improved reasoning capabilities for complex tasks.
    """
    
    def __init__(
        self,
        name: str,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_selection_strategy: Optional[Callable[[ReActAgentState], List[Dict[str, Any]]]] = None,
        reasoning_steps: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ReAct agent node with enhanced reasoning.
        
        Args:
            name: Name of the node.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            system_message: System message to use.
            tools: Tools available to the agent.
            tool_selection_strategy: Strategy for selecting tools based on state.
            reasoning_steps: Number of reasoning steps to perform before action.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        self.system_message = system_message
        self.tools = tools or []
        self.tool_selection_strategy = tool_selection_strategy
        self.reasoning_steps = max(1, reasoning_steps)  # Ensure at least 1 step
    
    def __call__(self, state: ReActAgentState) -> ReActAgentState:
        """
        Execute the ReAct agent with enhanced reasoning.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Executing ReAct agent in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get messages and current step
            messages = state.get("messages", [])
            current_step = state.get("current_step", "initialize")
            
            # Add system message if not present
            if self.system_message and not any(
                isinstance(msg, SystemMessage) for msg in messages
            ):
                messages = [SystemMessage(content=self.system_message)] + messages
            
            # Select tools if a tool selection strategy is provided
            tools_to_use = self.tools
            if self.tool_selection_strategy:
                tools_to_use = self.tool_selection_strategy(state)
                updated_state["selected_tools"] = tools_to_use
            
            # Check if we should emit events
            should_emit_events = isinstance(state, UIState) and state.get("emit_events", True)
            
            # Emit thinking start event if needed
            if should_emit_events:
                emit_event(EventType.THINKING_START, {
                    "nodeId": self.name,
                    "state": {
                        "current_step": current_step,
                        "tools": [tool["name"] for tool in tools_to_use]
                    }
                })
            
            # Perform step-by-step reasoning
            reasoning_steps = []
            current_thought = state.get("thought", "")
            action = None
            action_input = None
            
            for i in range(self.reasoning_steps):
                # Skip reasoning steps if we already have an action
                if i > 0 and action:
                    break
                
                # Format messages and tools for the LLM
                formatted_prompt = self._create_react_prompt(
                    messages, tools_to_use, current_thought, state.get("action"), 
                    state.get("action_input"), state.get("observation"),
                    reasoning_steps=reasoning_steps
                )
                
                # Generate response using LLM
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        self.llm_manager.generate(
                            prompt=formatted_prompt,
                            model_name=self.model_name,
                        )
                    )
                finally:
                    loop.close()
                
                # Parse the result to extract thought, action, and action input
                thought, action, action_input = self._parse_react_output(result.text)
                
                # Add to reasoning steps
                reasoning_steps.append({
                    "step": i + 1,
                    "thought": thought,
                })
                
                # Update current thought for next iteration
                current_thought = thought
            
            # Update state with final thought, action, and action input
            updated_state["thought"] = current_thought
            updated_state["action"] = action
            updated_state["action_input"] = action_input
            updated_state["reasoning_steps"] = reasoning_steps
            
            # Add to step history
            step_history = state.get("step_history", [])
            step_history.append({
                "step": current_step,
                "thought": current_thought,
                "action": action,
                "action_input": action_input,
                "reasoning_steps": reasoning_steps,
            })
            updated_state["step_history"] = step_history
            
            # Update current step
            updated_state["current_step"] = "execute_action" if action else "finish"
            
            # Emit thinking end event if needed
            if should_emit_events:
                emit_event(EventType.THINKING_END, {
                    "nodeId": self.name,
                    "state": {
                        "thought": current_thought,
                        "action": action,
                        "action_input": action_input,
                        "reasoning_steps": reasoning_steps,
                        "current_step": updated_state["current_step"]
                    }
                })
                
                # Emit state update
                if isinstance(state, UIState):
                    emit_state_update(updated_state)
            
            self.logger.info(f"ReAct agent executed for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error executing ReAct agent: {str(e)}")
            updated_state["error"] = str(e)
            
            # Emit error event if needed
            if isinstance(state, UIState) and state.get("emit_events", True):
                emit_event(EventType.ERROR, {
                    "errorCode": "react_agent_error",
                    "message": str(e)
                })
        
        return updated_state
    
    def _create_react_prompt(
        self,
        messages: List[BaseMessage],
        tools: List[Dict[str, Any]],
        thought: Optional[str] = None,
        action: Optional[str] = None,
        action_input: Optional[Dict[str, Any]] = None,
        observation: Optional[str] = None,
        reasoning_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Create a prompt for the ReAct agent with enhanced reasoning.
        
        Args:
            messages: List of messages.
            tools: List of tools.
            thought: Current thought.
            action: Current action.
            action_input: Current action input.
            observation: Current observation.
            reasoning_steps: Previous reasoning steps.
            
        Returns:
            ReAct prompt.
        """
        # Format messages
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                formatted_messages.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_messages.append(f"AI: {msg.content}")
            else:
                formatted_messages.append(f"Message: {msg.content}")
        
        messages_text = "\n\n".join(formatted_messages)
        
        # Format tools
        tools_text = ""
        if tools:
            tools_text = "Available tools:\n"
            for tool in tools:
                tools_text += f"- {tool['name']}: {tool['description']}\n"
                if "parameters" in tool:
                    tools_text += "  Parameters:\n"
                    for param_name, param_info in tool["parameters"].items():
                        param_desc = param_info.get("description", "")
                        param_type = param_info.get("type", "")
                        tools_text += f"    - {param_name} ({param_type}): {param_desc}\n"
        
        # Format previous reasoning steps
        reasoning_text = ""
        if reasoning_steps:
            reasoning_text = "Previous reasoning steps:\n"
            for step in reasoning_steps:
                reasoning_text += f"Step {step['step']}: {step['thought']}\n\n"
        
        # Format ReAct components
        react_text = ""
        if thought:
            react_text += f"Thought: {thought}\n"
        if action:
            react_text += f"Action: {action}\n"
        if action_input:
            import json
            react_text += f"Action Input: {json.dumps(action_input)}\n"
        if observation:
            react_text += f"Observation: {observation}\n"
        
        return (
            "You are a ReAct agent that thinks step by step to solve tasks.\n\n"
            "Follow this format:\n"
            "Thought: your detailed reasoning about what to do next\n"
            "Action: the action to take (must be one of the available tools)\n"
            "Action Input: the input to the action (in JSON format)\n"
            "Observation: the result of the action\n"
            "... (repeat Thought/Action/Action Input/Observation as needed)\n"
            "Thought: I've solved the task\n"
            "Final Answer: the final answer to the task\n\n"
            "When thinking, break down complex problems into smaller steps. Consider:\n"
            "1. What information do you need?\n"
            "2. What tools would help you get that information?\n"
            "3. How will you use the information to solve the task?\n"
            "4. What potential challenges might arise and how will you address them?\n\n"
            f"{tools_text}\n\n"
            f"Previous conversation:\n{messages_text}\n\n"
            f"{reasoning_text}\n"
            f"Current reasoning:\n{react_text}\n"
            "Thought:"
        )
    
    def _parse_react_output(self, output: str) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Parse the output from the ReAct agent.
        
        Args:
            output: Output from the LLM.
            
        Returns:
            Tuple of (thought, action, action_input).
        """
        lines = output.strip().split("\n")
        
        thought = output.strip()
        action = None
        action_input = None
        
        # Look for Action and Action Input
        for i, line in enumerate(lines):
            if line.startswith("Action:"):
                action = line[len("Action:"):].strip()
                
                # Look for Action Input in the next lines
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("Action Input:"):
                        action_input_str = lines[j][len("Action Input:"):].strip()
                        try:
                            import json
                            action_input = json.loads(action_input_str)
                        except json.JSONDecodeError:
                            # If not valid JSON, use as string
                            action_input = {"input": action_input_str}
                        break
                break
            elif line.startswith("Final Answer:"):
                # If we find a Final Answer, there's no action to take
                break
        
        return thought, action, action_input


class ToolExecutionNode(BaseNode[ReActAgentState]):
    """
    Node for executing tools in a ReAct agent workflow.
    
    This class implements tool execution logic in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        tool_registry: Dict[str, Callable],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the tool execution node.
        
        Args:
            name: Name of the node.
            tool_registry: Dictionary mapping tool names to functions.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        self.tool_registry = tool_registry
    
    def __call__(self, state: ReActAgentState) -> ReActAgentState:
        """
        Execute a tool.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Executing tool in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get action and action input
            action = state.get("action")
            action_input = state.get("action_input", {})
            
            if not action:
                self.logger.warning("No action specified")
                updated_state["observation"] = "Error: No action specified"
                updated_state["current_step"] = "react"
                return updated_state
            
            # Check if tool exists
            if action not in self.tool_registry:
                self.logger.warning(f"Unknown tool: {action}")
                updated_state["observation"] = f"Error: Unknown tool '{action}'"
                updated_state["current_step"] = "react"
                return updated_state
            
            # Check if we should emit events
            should_emit_events = isinstance(state, UIState) and state.get("emit_events", True)
            
            # Generate a tool call ID and emit tool call start event if needed
            tool_call_id = None
            if should_emit_events:
                tool_call_id = emit_tool_call_start(action)
                emit_tool_call_args(tool_call_id, action_input)
            
            # Execute the tool
            tool_func = self.tool_registry[action]
            result = tool_func(**action_input)
            
            # Emit tool call end event if needed
            if should_emit_events and tool_call_id:
                emit_tool_call_end(tool_call_id, result)
            
            # Update state with observation
            updated_state["observation"] = str(result)
            
            # Add to tool calls and results
            tool_calls = state.get("tool_calls", [])
            tool_results = state.get("tool_results", [])
            
            tool_calls.append({
                "tool": action,
                "input": action_input,
                "tool_call_id": tool_call_id
            })
            tool_results.append({
                "tool": action,
                "input": action_input,
                "result": str(result),
                "tool_call_id": tool_call_id
            })
            
            updated_state["tool_calls"] = tool_calls
            updated_state["tool_results"] = tool_results
            
            # Update current step
            updated_state["current_step"] = "react"
            
            # Emit state update if needed
            if should_emit_events and isinstance(state, UIState):
                emit_state_update(updated_state)
            
            self.logger.info(f"Tool executed for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error executing tool: {str(e)}")
            updated_state["observation"] = f"Error: {str(e)}"
            updated_state["current_step"] = "react"
            
            # Emit error event if needed
            if isinstance(state, UIState) and state.get("emit_events", True):
                emit_event(EventType.ERROR, {
                    "errorCode": "tool_execution_error",
                    "message": str(e)
                })
        
        return updated_state


class AgentCoordinatorNode(BaseNode[MultiAgentState]):
    """
    Node for coordinating multiple agents.
    
    This class implements agent coordination logic in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        agent_selection_strategy: Callable[[MultiAgentState], str],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the agent coordinator node.
        
        Args:
            name: Name of the node.
            agent_selection_strategy: Function to select the next agent.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        self.agent_selection_strategy = agent_selection_strategy
    
    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """
        Coordinate multiple agents.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Coordinating agents in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Select the next agent
            next_agent = self.agent_selection_strategy(state)
            
            # Update active agent
            updated_state["active_agent"] = next_agent
            
            # Add to execution history
            execution_history = state.get("execution_history", [])
            execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "selected_agent": next_agent,
            })
            updated_state["execution_history"] = execution_history
            
            # Emit agent change event if needed
            if isinstance(state, UIState) and state.get("emit_events", True):
                emit_event(EventType.AGENT_CHANGE, {
                    "agentId": next_agent
                })
                
                # Emit state update
                emit_state_update(updated_state)
            
            self.logger.info(f"Selected agent '{next_agent}' for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error coordinating agents: {str(e)}")
            updated_state["error"] = str(e)
            
            # Emit error event if needed
            if isinstance(state, UIState) and state.get("emit_events", True):
                emit_event(EventType.ERROR, {
                    "errorCode": "agent_coordination_error",
                    "message": str(e)
                })
        
        return updated_state
