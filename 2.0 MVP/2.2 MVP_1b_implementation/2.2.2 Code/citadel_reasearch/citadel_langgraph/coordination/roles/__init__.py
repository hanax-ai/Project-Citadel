
"""
Agent roles module for Citadel LangGraph.

This module provides specialized agent roles for multi-agent workflows.
"""

from .researcher_agent import ResearcherAgent
from .planner_agent import PlannerAgent
from .executor_agent import ExecutorAgent
from .critic_agent import CriticAgent

__all__ = [
    "ResearcherAgent",
    "PlannerAgent",
    "ExecutorAgent",
    "CriticAgent",
]
