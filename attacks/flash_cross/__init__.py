"""
Flash Cross Attack (S1) - 4-Stage Progressive Attack Implementation

The Flash Cross attack is the signature attack pattern of our system,
implementing a sophisticated 4-stage approach that mimics real-world
attack behaviors observed in maritime incidents.

Attack Progression:
1. Parallel Following (2 min) - Ghost vessel follows target at safe distance
2. Approach Initiation (30 sec) - Begin closing distance with speed increase  
3. Flash Cross Maneuver (45 sec) - Rapid crossing maneuver to trigger CPA alert
4. Silent Disappearance (30+ sec) - Vanish after causing evasive response

Key Features:
- Physics-constrained ghost vessel movement
- COLREGs-compliant target response simulation
- Real-time CPA/TCPA monitoring
- Adaptive timing based on target behavior
"""

from .flash_cross_attack import FlashCrossAttack
from .stage_executor import StageExecutor
from .ghost_vessel import GhostVessel

__all__ = ["FlashCrossAttack", "StageExecutor", "GhostVessel"]
