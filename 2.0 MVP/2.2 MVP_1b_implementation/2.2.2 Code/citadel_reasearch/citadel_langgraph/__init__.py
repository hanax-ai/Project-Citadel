
"""
LangGraph workflows for Project Citadel.

This package provides stateful agent operations using LangGraph.
It includes components for agent workflows, memory management, feedback loops,
enhanced reasoning mechanisms, tool usage, and multi-agent coordination.
"""

__version__ = "0.1.0"

# Import submodules
from citadel_langgraph import agents
from citadel_langgraph import edges
from citadel_langgraph import nodes
from citadel_langgraph import state
from citadel_langgraph import workflows
from citadel_langgraph import feedback
try:
    from citadel_langgraph import tools
    from citadel_langgraph import coordination
except ImportError:
    # These modules might not be fully initialized yet
    pass
