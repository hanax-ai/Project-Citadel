
"""
Base workflow classes for LangGraph workflows.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union, Set

from langgraph.graph import StateGraph, END

from citadel_core.logging import get_logger
from citadel_langgraph.state.base import BaseState
from citadel_langgraph.state.ui_state import UIState, emit_state_update
from citadel_langgraph.nodes.base import BaseNode
from citadel_langgraph.edges.base import BaseEdge
from citadel_langgraph.events.emitter import (
    start_run, end_run, emit_node_start, emit_node_end
)

# Define a generic type variable for state
S = TypeVar('S', bound=BaseState)


class BaseWorkflow(Generic[S], ABC):
    """
    Base class for all workflows.
    
    This class defines the interface for workflows in LangGraph.
    """
    
    def __init__(
        self,
        name: str,
        state_type: type,
        logger: Optional[logging.Logger] = None,
        emit_events: bool = True,
    ):
        """
        Initialize the workflow.
        
        Args:
            name: Name of the workflow.
            state_type: Type of the state.
            logger: Logger instance.
            emit_events: Whether to emit events during workflow execution.
        """
        self.name = name
        self.state_type = state_type
        self.logger = logger or get_logger(f"citadel.langgraph.workflows.{name}")
        self.emit_events = emit_events
        
        # Initialize the state graph
        self.graph = StateGraph(state_type)
        
        # Initialize nodes and edges
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: Dict[str, BaseEdge] = {}
        
        # Track node connections
        self.connections: Dict[str, Set[str]] = {}
        
        # Entry point
        self.entry_point: Optional[str] = None
        
        # Wrap nodes with event emission if needed
        if self.emit_events:
            self._wrap_nodes_with_events()
    
    def _wrap_nodes_with_events(self) -> None:
        """
        Wrap nodes with event emission.
        
        This method wraps all nodes in the graph with event emission logic.
        """
        # Override the add_node method to wrap nodes with event emission
        original_add_node = self.graph.add_node
        
        def add_node_with_events(name, fn):
            # Create a wrapper function that emits events
            def node_wrapper(state):
                # Emit node start event
                if self.emit_events and isinstance(state, UIState) and state.get("emit_events", True):
                    emit_node_start(name, state)
                
                # Call the original node function
                result = fn(state)
                
                # Emit node end event
                if self.emit_events and isinstance(result, UIState) and result.get("emit_events", True):
                    emit_node_end(name, result)
                    # Also emit a state update
                    emit_state_update(result)
                
                return result
            
            # Add the wrapped node to the graph
            return original_add_node(name, node_wrapper)
        
        # Replace the add_node method
        self.graph.add_node = add_node_with_events
    
    def add_node(self, node: BaseNode[S]) -> None:
        """
        Add a node to the workflow.
        
        Args:
            node: Node to add.
        """
        self.logger.debug(f"Adding node '{node.name}' to workflow '{self.name}'")
        
        # Add node to the graph
        self.graph.add_node(node.name, node)
        
        # Store the node
        self.nodes[node.name] = node
        
        # Initialize connections for this node
        if node.name not in self.connections:
            self.connections[node.name] = set()
    
    def add_edge(self, source: str, target: Union[str, BaseEdge[S]]) -> None:
        """
        Add an edge to the workflow.
        
        Args:
            source: Source node name.
            target: Target node name or edge.
        """
        if isinstance(target, BaseEdge):
            self.logger.debug(f"Adding conditional edge from '{source}' in workflow '{self.name}'")
            
            # Store the edge
            self.edges[target.name] = target
            
            # Add edge to the graph
            self.graph.add_conditional_edges(
                source,
                target,
            )
            
            # We don't know the exact targets for conditional edges
            # They will be determined at runtime
        else:
            self.logger.debug(f"Adding edge from '{source}' to '{target}' in workflow '{self.name}'")
            
            # Add edge to the graph
            self.graph.add_edge(source, target)
            
            # Update connections
            if source not in self.connections:
                self.connections[source] = set()
            self.connections[source].add(target)
    
    def set_entry_point(self, node_name: str) -> None:
        """
        Set the entry point of the workflow.
        
        Args:
            node_name: Name of the entry point node.
        """
        self.logger.debug(f"Setting entry point to '{node_name}' in workflow '{self.name}'")
        
        # Set entry point in the graph
        self.graph.set_entry_point(node_name)
        
        # Store the entry point
        self.entry_point = node_name
    
    def add_end_node(self, node_name: str) -> None:
        """
        Add an end node to the workflow.
        
        Args:
            node_name: Name of the node to connect to END.
        """
        self.logger.debug(f"Adding end node '{node_name}' in workflow '{self.name}'")
        
        # Add edge to END
        self.graph.add_edge(node_name, END)
        
        # Update connections
        if node_name not in self.connections:
            self.connections[node_name] = set()
        self.connections[node_name].add("END")
    
    @abstractmethod
    def build(self) -> None:
        """
        Build the workflow.
        
        This method should add all nodes and edges to the workflow.
        """
        pass
    
    def compile(self) -> Any:
        """
        Compile the workflow.
        
        Returns:
            Compiled workflow.
        """
        self.logger.info(f"Compiling workflow '{self.name}'")
        
        # Build the workflow if not already built
        if not self.nodes:
            self.build()
        
        # Check if entry point is set
        if not self.entry_point:
            raise ValueError(f"Entry point not set for workflow '{self.name}'")
        
        # Compile the graph
        return self.graph.compile()
    
    def run(self, initial_state: Optional[S] = None, **kwargs) -> S:
        """
        Run the workflow.
        
        Args:
            initial_state: Initial state.
            **kwargs: Additional arguments to pass to the workflow.
            
        Returns:
            Final state.
        """
        self.logger.info(f"Running workflow '{self.name}'")
        
        # Start a new run if events are enabled
        run_id = None
        if self.emit_events and (initial_state is None or 
                                (isinstance(initial_state, UIState) and 
                                 initial_state.get("emit_events", True))):
            run_id = start_run()
        
        # Compile the workflow if not already compiled
        compiled = self.compile()
        
        try:
            # Run the workflow
            final_state = compiled.invoke(initial_state or {}, **kwargs)
            
            # Emit final state update if events are enabled
            if self.emit_events and isinstance(final_state, UIState) and final_state.get("emit_events", True):
                emit_state_update(final_state)
            
            return final_state
        finally:
            # End the run if events are enabled
            if run_id and self.emit_events:
                end_run()
    
    def visualize(self, output_path: Optional[str] = None) -> None:
        """
        Visualize the workflow.
        
        Args:
            output_path: Path to save the visualization.
        """
        try:
            import graphviz
            
            # Create a new directed graph
            dot = graphviz.Digraph(comment=f"Workflow: {self.name}")
            
            # Add nodes
            for node_name in self.nodes:
                dot.node(node_name)
            
            # Add END node
            dot.node("END", shape="doublecircle")
            
            # Add edges
            for source, targets in self.connections.items():
                for target in targets:
                    dot.edge(source, target)
            
            # Render the graph
            if output_path:
                dot.render(output_path, format="png", cleanup=True)
            
            return dot
        except ImportError:
            self.logger.warning("graphviz not installed, cannot visualize workflow")
            return None
