"""
Impact Simulator module for AIS attack generator.

This module simulates vessel reactions to false target attacks by:
1. Building traffic snapshots from attack data
2. Evaluating collision risks using CPA/TCPA
3. Simulating vessel avoidance maneuvers
4. Exporting impact data showing vessel reactions
"""

from .traffic_snapshot_builder import TrafficSnapshotBuilder
from .situation_evaluator import SituationEvaluator
from .maneuver_simulator import ManeuverSimulator
from .impact_exporter import ImpactExporter

__all__ = [
    'TrafficSnapshotBuilder',
    'SituationEvaluator', 
    'ManeuverSimulator',
    'ImpactExporter'
]