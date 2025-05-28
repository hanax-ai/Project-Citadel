
"""
Team coordinator for multi-agent workflows.
"""

import logging
from typing import Dict, List, Optional, Callable, Any, Type, Union
from datetime import datetime

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

from citadel_langgraph.state.agent_state import MultiAgentState, AgentState
from citadel_langgraph.nodes.base import BaseNode


class TeamCoordinator(BaseNode[MultiAgentState]):
    """
    Coordinator for multi-agent teams.
    
    This class manages the coordination of multiple agents in a workflow,
    including agent selection, message passing, and task allocation.
    """
    
    def __init__(
        self,
        name: str,
        agent_configs: Dict[str, Dict[str, Any]],
        agent_selection_strategy: Callable[[MultiAgentState], str],
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the team coordinator.
        
        Args:
            name: Name of the coordinator.
            agent_configs: Dictionary mapping agent IDs to their configurations.
            agent_selection_strategy: Function to select the next agent.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.coordination.{name}")
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        
        self.agent_configs = agent_configs
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
            
            self.logger.info(f"Selected agent '{next_agent}' for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error coordinating agents: {str(e)}")
            updated_state["error"] = str(e)
        
        return updated_state
    
    def send_message(
        self,
        state: MultiAgentState,
        from_agent: str,
        to_agent: str,
        message: str,
        message_type: str = "text",
    ) -> MultiAgentState:
        """
        Send a message from one agent to another.
        
        Args:
            state: Current workflow state.
            from_agent: ID of the sending agent.
            to_agent: ID of the receiving agent.
            message: Message content.
            message_type: Type of message.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Sending message from {from_agent} to {to_agent}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get agent states
            agent_states = state.get("agent_states", {})
            
            # Check if agents exist
            if from_agent not in agent_states:
                raise ValueError(f"Agent '{from_agent}' not found")
            if to_agent not in agent_states:
                raise ValueError(f"Agent '{to_agent}' not found")
            
            # Create message
            message_obj = {
                "from": from_agent,
                "to": to_agent,
                "content": message,
                "type": message_type,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Add to messages
            messages = state.get("messages", [])
            messages.append(message_obj)
            updated_state["messages"] = messages
            
            # Add to agent's messages
            to_agent_state = dict(agent_states.get(to_agent, {}))
            to_agent_messages = to_agent_state.get("messages", [])
            
            # Convert to appropriate message type
            if message_type == "system":
                to_agent_messages.append(SystemMessage(content=message))
            else:
                to_agent_messages.append(AIMessage(
                    content=message,
                    additional_kwargs={"from_agent": from_agent}
                ))
            
            to_agent_state["messages"] = to_agent_messages
            agent_states[to_agent] = to_agent_state
            updated_state["agent_states"] = agent_states
            
            self.logger.info(f"Message sent from {from_agent} to {to_agent}")
            
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            updated_state["error"] = str(e)
        
        return updated_state
    
    def broadcast_message(
        self,
        state: MultiAgentState,
        from_agent: str,
        message: str,
        message_type: str = "text",
        exclude_agents: Optional[List[str]] = None,
    ) -> MultiAgentState:
        """
        Broadcast a message to all agents.
        
        Args:
            state: Current workflow state.
            from_agent: ID of the sending agent.
            message: Message content.
            message_type: Type of message.
            exclude_agents: List of agent IDs to exclude.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Broadcasting message from {from_agent}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get agent states
            agent_states = state.get("agent_states", {})
            
            # Check if sending agent exists
            if from_agent not in agent_states:
                raise ValueError(f"Agent '{from_agent}' not found")
            
            # Determine recipients
            exclude_agents = exclude_agents or []
            recipients = [
                agent_id for agent_id in agent_states.keys()
                if agent_id != from_agent and agent_id not in exclude_agents
            ]
            
            # Send message to each recipient
            for recipient in recipients:
                updated_state = self.send_message(
                    updated_state, from_agent, recipient, message, message_type
                )
            
            self.logger.info(f"Message broadcast from {from_agent} to {len(recipients)} agents")
            
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {str(e)}")
            updated_state["error"] = str(e)
        
        return updated_state
    
    def update_shared_memory(
        self,
        state: MultiAgentState,
        key: str,
        value: Any,
        agent_id: Optional[str] = None,
    ) -> MultiAgentState:
        """
        Update the shared memory.
        
        Args:
            state: Current workflow state.
            key: Memory key.
            value: Memory value.
            agent_id: ID of the agent updating the memory.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Updating shared memory: {key}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get shared memory
            shared_memory = state.get("shared_memory", {})
            
            # Update memory
            shared_memory[key] = value
            
            # Add metadata
            memory_metadata = state.get("memory_metadata", {})
            memory_metadata[key] = {
                "updated_at": datetime.utcnow().isoformat(),
                "updated_by": agent_id,
            }
            
            # Update state
            updated_state["shared_memory"] = shared_memory
            updated_state["memory_metadata"] = memory_metadata
            
            self.logger.info(f"Shared memory updated: {key}")
            
        except Exception as e:
            self.logger.error(f"Error updating shared memory: {str(e)}")
            updated_state["error"] = str(e)
        
        return updated_state
    
    def get_from_shared_memory(
        self,
        state: MultiAgentState,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a value from the shared memory.
        
        Args:
            state: Current workflow state.
            key: Memory key.
            default: Default value if key not found.
            
        Returns:
            Memory value.
        """
        shared_memory = state.get("shared_memory", {})
        return shared_memory.get(key, default)


class RoundRobinStrategy:
    """
    Round-robin agent selection strategy.
    
    This strategy selects agents in a round-robin fashion.
    """
    
    def __init__(self, agent_ids: List[str]):
        """
        Initialize the strategy.
        
        Args:
            agent_ids: List of agent IDs.
        """
        self.agent_ids = agent_ids
    
    def __call__(self, state: MultiAgentState) -> str:
        """
        Select the next agent.
        
        Args:
            state: Current workflow state.
            
        Returns:
            ID of the next agent.
        """
        # Get current agent
        current_agent = state.get("active_agent")
        
        # If no current agent, start with the first one
        if not current_agent:
            return self.agent_ids[0]
        
        # Find the index of the current agent
        try:
            current_index = self.agent_ids.index(current_agent)
        except ValueError:
            # If current agent not found, start with the first one
            return self.agent_ids[0]
        
        # Select the next agent
        next_index = (current_index + 1) % len(self.agent_ids)
        return self.agent_ids[next_index]


class TaskBasedStrategy:
    """
    Task-based agent selection strategy.
    
    This strategy selects agents based on the current task.
    """
    
    def __init__(
        self,
        task_agent_mapping: Dict[str, List[str]],
        default_agent: str,
        llm_manager: Optional[LLMManager] = None,
    ):
        """
        Initialize the strategy.
        
        Args:
            task_agent_mapping: Mapping from task keywords to agent IDs.
            default_agent: Default agent ID.
            llm_manager: LLM manager for task analysis.
        """
        self.task_agent_mapping = task_agent_mapping
        self.default_agent = default_agent
        self.llm_manager = llm_manager
    
    def __call__(self, state: MultiAgentState) -> str:
        """
        Select the next agent based on the task.
        
        Args:
            state: Current workflow state.
            
        Returns:
            ID of the next agent.
        """
        # Get the current task
        task = self._extract_task(state)
        
        # If no task, use default agent
        if not task:
            return self.default_agent
        
        # Select agent based on task keywords
        for keyword, agent_ids in self.task_agent_mapping.items():
            if keyword.lower() in task.lower():
                # Return the first matching agent
                return agent_ids[0]
        
        # If no match, use default agent
        return self.default_agent
    
    def _extract_task(self, state: MultiAgentState) -> str:
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
            if message.get("type") == "human":
                return message.get("content", "")
        
        # If no human message found, check shared memory
        shared_memory = state.get("shared_memory", {})
        return shared_memory.get("task", "")


class DynamicStrategy:
    """
    Dynamic agent selection strategy.
    
    This strategy uses an LLM to select the next agent based on the current state.
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        agent_descriptions: Dict[str, str],
        llm_manager: LLMManager,
        model_name: str = "mistral:latest",
    ):
        """
        Initialize the strategy.
        
        Args:
            agent_ids: List of agent IDs.
            agent_descriptions: Descriptions of each agent.
            llm_manager: LLM manager for agent selection.
            model_name: Name of the model to use.
        """
        self.agent_ids = agent_ids
        self.agent_descriptions = agent_descriptions
        self.llm_manager = llm_manager
        self.model_name = model_name
    
    def __call__(self, state: MultiAgentState) -> str:
        """
        Select the next agent dynamically.
        
        Args:
            state: Current workflow state.
            
        Returns:
            ID of the next agent.
        """
        # Format the current state for the LLM
        state_description = self._format_state(state)
        
        # Format agent descriptions
        agent_descriptions = self._format_agent_descriptions()
        
        # Create prompt for agent selection
        prompt = (
            "You are an AI coordinator tasked with selecting the most appropriate agent "
            "for the current state of a multi-agent workflow.\n\n"
            f"Current state:\n{state_description}\n\n"
            f"Available agents:\n{agent_descriptions}\n\n"
            "Based on the current state and the capabilities of each agent, "
            "which agent should be selected next? Respond with just the agent ID."
        )
        
        # Generate response using LLM
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.llm_manager.generate(
                    prompt=prompt,
                    model_name=self.model_name,
                )
            )
        finally:
            loop.close()
        
        # Parse the result to extract the agent ID
        selected_agent = result.text.strip()
        
        # Validate the selected agent
        if selected_agent in self.agent_ids:
            return selected_agent
        else:
            # If invalid agent, return the first one
            return self.agent_ids[0]
    
    def _format_state(self, state: MultiAgentState) -> str:
        """
        Format the state for the LLM.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Formatted state description.
        """
        # Get relevant state information
        active_agent = state.get("active_agent", "None")
        shared_memory = state.get("shared_memory", {})
        messages = state.get("messages", [])
        
        # Format the state
        state_description = f"Active agent: {active_agent}\n"
        
        # Add shared memory
        state_description += "Shared memory:\n"
        for key, value in shared_memory.items():
            state_description += f"- {key}: {value}\n"
        
        # Add recent messages
        state_description += "Recent messages:\n"
        for message in messages[-5:]:  # Last 5 messages
            from_agent = message.get("from", "Unknown")
            to_agent = message.get("to", "Unknown")
            content = message.get("content", "")
            state_description += f"- {from_agent} -> {to_agent}: {content}\n"
        
        return state_description
    
    def _format_agent_descriptions(self) -> str:
        """
        Format agent descriptions for the LLM.
        
        Returns:
            Formatted agent descriptions.
        """
        descriptions = []
        for agent_id in self.agent_ids:
            description = self.agent_descriptions.get(agent_id, "No description")
            descriptions.append(f"{agent_id}: {description}")
        
        return "\n".join(descriptions)
