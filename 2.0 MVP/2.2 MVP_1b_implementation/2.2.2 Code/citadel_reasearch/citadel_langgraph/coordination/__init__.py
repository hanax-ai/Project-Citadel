
"""
Coordination module for Citadel LangGraph.

This module provides components for coordinating multiple agents in LangGraph workflows.
"""

from .team_coordinator import TeamCoordinator
from .roles import (
    ResearcherAgent,
    PlannerAgent,
    ExecutorAgent,
    CriticAgent,
)

__all__ = [
    "TeamCoordinator",
    "ResearcherAgent",
    "PlannerAgent",
    "ExecutorAgent",
    "CriticAgent",
]
