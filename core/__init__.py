"""
AIS Attack Generation System - Core Module

This module contains the core components for AIS attack generation:
- Attack Orchestrator: Multi-stage progressive attack coordination
- Target Selector: MCDA + fuzzy logic target selection
- Physics Engine: MMG constraints and ship dynamics
- COLREGs Validator: Maritime collision avoidance rules
- Auto Labeler: Automatic attack labeling and metadata generation
"""

from .attack_orchestrator import AttackOrchestrator
from .target_selector import TargetSelector
from .physics_engine import PhysicsEngine
from .colregs_validator import COLREGSValidator
from .auto_labeler import AutoLabeler

__version__ = "1.0.0"
__author__ = "Jason Tim Wong"

__all__ = [
    "AttackOrchestrator",
    "TargetSelector", 
    "PhysicsEngine",
    "COLREGSValidator",
    "AutoLabeler",
]
