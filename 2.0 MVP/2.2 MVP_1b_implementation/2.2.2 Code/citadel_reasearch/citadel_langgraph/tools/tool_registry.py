
"""
Tool registry for managing and selecting tools in agent workflows.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, Type, Union

from citadel_core.logging import get_logger
from citadel_langgraph.state.agent_state import ReActAgentState


class BaseTool(ABC):
    """
    Base class for all tools.
    
    This class defines the interface for tools that can be used by agents.
    """
    
    name: str
    description: str
    
    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """
        Execute the tool with the given arguments.
        
        Args:
            **kwargs: Tool-specific arguments.
            
        Returns:
            The result of the tool execution.
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary representation.
        
        Returns:
            Dictionary representation of the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
        }


class ToolRegistry:
    """
    Registry for managing tools available to agents.
    
    This class provides methods for registering, retrieving, and managing tools.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the tool registry.
        
        Args:
            logger: Logger instance.
        """
        self.logger = logger or get_logger("citadel.langgraph.tools.registry")
        self._tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: The tool to register.
        """
        self.logger.info(f"Registering tool: {tool.name}")
        if tool.name in self._tools:
            self.logger.warning(f"Tool '{tool.name}' already registered. Overwriting.")
        self._tools[tool.name] = tool
    
    def register_tools(self, tools: List[BaseTool]) -> None:
        """
        Register multiple tools in the registry.
        
        Args:
            tools: The tools to register.
        """
        for tool in tools:
            self.register_tool(tool)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: The name of the tool.
            
        Returns:
            The tool if found, None otherwise.
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary mapping tool names to tools.
        """
        return dict(self._tools)
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all registered tools.
        
        Returns:
            List of tool descriptions.
        """
        return [tool.to_dict() for tool in self._tools.values()]
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name.
        
        Args:
            name: The name of the tool.
            **kwargs: Tool-specific arguments.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            ValueError: If the tool is not found.
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry")
        
        self.logger.info(f"Executing tool: {name}")
        try:
            result = tool(**kwargs)
            self.logger.info(f"Tool '{name}' executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error executing tool '{name}': {str(e)}")
            raise


class ToolSelectionStrategy(ABC):
    """
    Base class for tool selection strategies.
    
    This class defines the interface for strategies that select tools for agents.
    """
    
    @abstractmethod
    def select_tools(self, state: ReActAgentState) -> List[Dict[str, Any]]:
        """
        Select tools for an agent based on the current state.
        
        Args:
            state: The current agent state.
            
        Returns:
            List of tool descriptions.
        """
        pass


class AllToolsStrategy(ToolSelectionStrategy):
    """
    Strategy that provides all registered tools to the agent.
    """
    
    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the strategy.
        
        Args:
            tool_registry: The tool registry to use.
        """
        self.tool_registry = tool_registry
    
    def select_tools(self, state: ReActAgentState) -> List[Dict[str, Any]]:
        """
        Select all registered tools.
        
        Args:
            state: The current agent state.
            
        Returns:
            List of all tool descriptions.
        """
        return self.tool_registry.get_tool_descriptions()


class TaskBasedToolStrategy(ToolSelectionStrategy):
    """
    Strategy that selects tools based on the task description.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        task_tool_mapping: Dict[str, List[str]],
        default_tools: Optional[List[str]] = None,
    ):
        """
        Initialize the strategy.
        
        Args:
            tool_registry: The tool registry to use.
            task_tool_mapping: Mapping from task keywords to tool names.
            default_tools: Default tools to always include.
        """
        self.tool_registry = tool_registry
        self.task_tool_mapping = task_tool_mapping
        self.default_tools = default_tools or []
    
    def select_tools(self, state: ReActAgentState) -> List[Dict[str, Any]]:
        """
        Select tools based on the task description.
        
        Args:
            state: The current agent state.
            
        Returns:
            List of selected tool descriptions.
        """
        # Get task description from the first human message
        task_description = ""
        for message in state.get("messages", []):
            if hasattr(message, "type") and message.type == "human":
                task_description = message.content
                break
        
        # Select tools based on task description
        selected_tool_names = set(self.default_tools)
        
        for keyword, tool_names in self.task_tool_mapping.items():
            if keyword.lower() in task_description.lower():
                selected_tool_names.update(tool_names)
        
        # Get tool descriptions
        selected_tools = []
        for name in selected_tool_names:
            tool = self.tool_registry.get_tool(name)
            if tool:
                selected_tools.append(tool.to_dict())
        
        return selected_tools


class DynamicToolStrategy(ToolSelectionStrategy):
    """
    Strategy that dynamically selects tools based on a selection function.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        selection_function: Callable[[ReActAgentState, Dict[str, BaseTool]], List[str]],
    ):
        """
        Initialize the strategy.
        
        Args:
            tool_registry: The tool registry to use.
            selection_function: Function that selects tool names based on state.
        """
        self.tool_registry = tool_registry
        self.selection_function = selection_function
    
    def select_tools(self, state: ReActAgentState) -> List[Dict[str, Any]]:
        """
        Dynamically select tools based on the selection function.
        
        Args:
            state: The current agent state.
            
        Returns:
            List of selected tool descriptions.
        """
        all_tools = self.tool_registry.get_all_tools()
        selected_tool_names = self.selection_function(state, all_tools)
        
        selected_tools = []
        for name in selected_tool_names:
            tool = self.tool_registry.get_tool(name)
            if tool:
                selected_tools.append(tool.to_dict())
        
        return selected_tools
