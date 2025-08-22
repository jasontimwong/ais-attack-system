"""
AIS Attack Types Module

This module implements 9 distinct AIS attack patterns:

S1: Flash Cross - 4-stage progressive attack with ghost vessel
S2: Zone Violation - Position spoofing with restricted area intrusion  
S3: Ghost Swarm - Coordinated 8-vessel attack formation
S4: Position Offset - 1.5nm systematic position displacement
S5: Port Spoofing - Harbor area disruption and false traffic
S6: Course Disruption - Forced collision avoidance maneuvers
S7: Identity Swap - MMSI identity exchange between vessels
S8: Identity Clone - Vessel identity duplication attack
S9: Identity Whitewashing - Reputation attack through identity manipulation
"""

from .flash_cross import FlashCrossAttack
from .zone_violation import ZoneViolationAttack
from .ghost_swarm import GhostSwarmAttack
from .position_offset import PositionOffsetAttack
from .port_spoofing import PortSpoofingAttack
from .course_disruption import CourseDisruptionAttack
from .identity_swap import IdentitySwapAttack
from .identity_clone import IdentityCloneAttack
from .identity_whitewashing import IdentityWhitewashingAttack

# Attack registry for dynamic loading
ATTACK_REGISTRY = {
    "s1_flash_cross": FlashCrossAttack,
    "s2_zone_violation": ZoneViolationAttack,
    "s3_ghost_swarm": GhostSwarmAttack,
    "s4_position_offset": PositionOffsetAttack,
    "s5_port_spoofing": PortSpoofingAttack,
    "s6_course_disruption": CourseDisruptionAttack,
    "s7_identity_swap": IdentitySwapAttack,
    "s8_identity_clone": IdentityCloneAttack,
    "s9_identity_whitewashing": IdentityWhitewashingAttack,
}

__all__ = [
    "FlashCrossAttack",
    "ZoneViolationAttack", 
    "GhostSwarmAttack",
    "PositionOffsetAttack",
    "PortSpoofingAttack",
    "CourseDisruptionAttack",
    "IdentitySwapAttack",
    "IdentityCloneAttack",
    "IdentityWhitewashingAttack",
    "ATTACK_REGISTRY",
]
