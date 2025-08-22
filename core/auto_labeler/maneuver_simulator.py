"""
Maneuver Simulator for Impact Simulator.

Simulates vessel turning and speed changes in response to collision risks.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from .traffic_snapshot_builder import VesselSnapshot, TrafficSnapshot
from .situation_evaluator import ConflictSituation
from .ghost_watcher import GhostThreat

logger = logging.getLogger(__name__)


class ManeuverEvent:
    """Represents a maneuver event with details."""
    
    def __init__(self, vessel_mmsi: int, maneuver_type: str, start_time: datetime, 
                 end_time: datetime, trigger_conflict: ConflictSituation,
                 original_cog: float, target_cog: float, original_sog: float = None, 
                 target_sog: float = None, trigger_type: str = 'vessel_conflict'):
        self.vessel_mmsi = vessel_mmsi
        self.maneuver_type = maneuver_type  # 'turn_starboard', 'turn_port', 'speed_change'
        self.start_time = start_time
        self.end_time = end_time
        self.trigger_conflict = trigger_conflict
        self.original_cog = original_cog
        self.target_cog = target_cog
        self.original_sog = original_sog
        self.target_sog = target_sog
        self.trigger_type = trigger_type  # 'vessel_conflict' or 'ghost_threat'
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'vessel_mmsi': self.vessel_mmsi,
            'maneuver_type': self.maneuver_type,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'original_cog': self.original_cog,
            'target_cog': self.target_cog,
            'original_sog': self.original_sog,
            'target_sog': self.target_sog,
            'trigger_type': self.trigger_type,
            'trigger_conflict': self.trigger_conflict.to_dict() if self.trigger_conflict else None
        }


class ManeuverSimulator:
    """
    Simulates vessel maneuvers in response to collision risks.
    
    Implements standard collision avoidance maneuvers including:
    - Starboard turn (30 degrees)
    - Port turn (30 degrees) 
    - Speed reduction
    """
    
    def __init__(self, max_turn_rate_deg_per_min: float = 15.0, 
                 time_step_seconds: int = 10,
                 standard_turn_degrees: float = 30.0):
        """
        Initialize maneuver simulator.
        
        Args:
            max_turn_rate_deg_per_min: Maximum turn rate in degrees per minute
            time_step_seconds: Time step for simulation
            standard_turn_degrees: Standard avoidance turn angle
        """
        self.max_turn_rate_deg_per_min = max_turn_rate_deg_per_min
        self.time_step_seconds = time_step_seconds
        self.standard_turn_degrees = standard_turn_degrees
        
        # Convert turn rate to degrees per time step
        self.turn_rate_per_step = (max_turn_rate_deg_per_min / 60.0) * time_step_seconds
        
        # Track active maneuvers
        self.active_maneuvers: Dict[int, ManeuverEvent] = {}
        self.completed_maneuvers: List[ManeuverEvent] = []
        
    def simulate_maneuvers(self, snapshots: Dict[datetime, TrafficSnapshot], 
                          conflicts: List[ConflictSituation],
                          ghost_threats: Optional[Dict[int, List[GhostThreat]]] = None) -> Dict[datetime, TrafficSnapshot]:
        """
        Simulate maneuvers for all conflicts and return modified snapshots.
        
        Args:
            snapshots: Original traffic snapshots
            conflicts: List of conflict situations requiring maneuvers
            
        Returns:
            Modified snapshots with maneuvers applied
        """
        # Filter conflicts to only process ghost-tagged conflicts
        ghost_conflicts = [c for c in conflicts if c.threat_tag == 'ghost']
        logger.info(f"Processing {len(ghost_conflicts)} ghost-tagged conflicts out of {len(conflicts)} total")
        
        # Create a copy of snapshots to modify
        modified_snapshots = {}
        for timestamp, snapshot in snapshots.items():
            modified_snapshots[timestamp] = self._copy_snapshot(snapshot)
        
        # Group ghost conflicts by vessel and time
        vessel_conflicts = self._group_conflicts_by_vessel(ghost_conflicts)
        
        # Process ghost threats first (higher priority)
        if ghost_threats:
            logger.info(f"Processing {len(ghost_threats)} vessels threatened by ghosts")
            for vessel_mmsi, threats in ghost_threats.items():
                if not threats:
                    continue
                
                # Find most critical ghost threat
                most_critical_threat = min(threats, key=lambda t: t.cpa_nm)
                
                # Create ghost-triggered maneuver
                maneuver_event = self._create_maneuver_for_ghost_threat(most_critical_threat, modified_snapshots)
                if maneuver_event:
                    self._apply_maneuver_to_snapshots(modified_snapshots, maneuver_event)
                    self.completed_maneuvers.append(maneuver_event)
                    
                    # Remove this vessel from regular conflicts to avoid double maneuvers
                    if vessel_mmsi in vessel_conflicts:
                        del vessel_conflicts[vessel_mmsi]
        
        # Process remaining vessel-to-vessel conflicts
        for vessel_mmsi, vessel_conflicts in vessel_conflicts.items():
            if not vessel_conflicts:
                continue
                
            # Find the most critical conflict for this vessel
            most_critical = min(vessel_conflicts, key=lambda c: c.cpa_distance_nm)
            
            # Create maneuver event
            maneuver_event = self._create_maneuver_for_conflict(most_critical)
            if maneuver_event:
                # Apply maneuver to snapshots
                self._apply_maneuver_to_snapshots(modified_snapshots, maneuver_event)
                self.completed_maneuvers.append(maneuver_event)
        
        logger.info(f"Completed {len(self.completed_maneuvers)} maneuvers")
        return modified_snapshots
    
    def _copy_snapshot(self, snapshot: TrafficSnapshot) -> TrafficSnapshot:
        """Create a deep copy of a snapshot."""
        new_snapshot = TrafficSnapshot(snapshot.timestamp)
        for vessel in snapshot.get_all_vessels():
            new_vessel = VesselSnapshot(
                mmsi=vessel.mmsi,
                timestamp=vessel.timestamp,
                lat=vessel.lat,
                lon=vessel.lon,
                sog=vessel.sog,
                cog=vessel.cog,
                heading=vessel.heading,
                vessel_name=vessel.vessel_name
            )
            new_snapshot.add_vessel(new_vessel)
        return new_snapshot
    
    def _group_conflicts_by_vessel(self, conflicts: List[ConflictSituation]) -> Dict[int, List[ConflictSituation]]:
        """Group conflicts by the give-way vessel."""
        vessel_conflicts = {}
        
        for conflict in conflicts:
            vessel_mmsi = conflict.give_way_vessel.mmsi
            if vessel_mmsi not in vessel_conflicts:
                vessel_conflicts[vessel_mmsi] = []
            vessel_conflicts[vessel_mmsi].append(conflict)
        
        return vessel_conflicts
    
    def _create_maneuver_for_conflict(self, conflict: ConflictSituation) -> Optional[ManeuverEvent]:
        """
        Create appropriate maneuver for a conflict situation.
        
        Args:
            conflict: ConflictSituation requiring maneuver
            
        Returns:
            ManeuverEvent or None
        """
        give_way_vessel = conflict.give_way_vessel
        
        # For MVP, always use starboard turn
        maneuver_type = 'turn_starboard'
        
        # Calculate target course
        original_cog = give_way_vessel.cog
        target_cog = (original_cog + self.standard_turn_degrees) % 360
        
        # Calculate maneuver duration based on turn rate
        degrees_to_turn = self.standard_turn_degrees
        maneuver_duration_seconds = (degrees_to_turn / self.max_turn_rate_deg_per_min) * 60
        
        start_time = conflict.timestamp
        end_time = start_time + timedelta(seconds=maneuver_duration_seconds)
        
        return ManeuverEvent(
            vessel_mmsi=give_way_vessel.mmsi,
            maneuver_type=maneuver_type,
            start_time=start_time,
            end_time=end_time,
            trigger_conflict=conflict,
            original_cog=original_cog,
            target_cog=target_cog
        )
    
    def _apply_maneuver_to_snapshots(self, snapshots: Dict[datetime, TrafficSnapshot], 
                                   maneuver_event: ManeuverEvent):
        """
        Apply maneuver to all relevant snapshots.
        
        Args:
            snapshots: Dictionary of snapshots to modify
            maneuver_event: Maneuver to apply
        """
        vessel_mmsi = maneuver_event.vessel_mmsi
        
        # Find all snapshots that need modification
        for timestamp, snapshot in snapshots.items():
            if timestamp < maneuver_event.start_time:
                continue
                
            vessel = snapshot.get_vessel(vessel_mmsi)
            if not vessel:
                continue
            
            # Apply maneuver based on timing
            if timestamp <= maneuver_event.end_time:
                # During maneuver - interpolate course change
                self._apply_turning_maneuver(vessel, maneuver_event, timestamp)
            else:
                # After maneuver - maintain new course
                self._apply_post_maneuver_course(vessel, maneuver_event, timestamp)
    
    def _apply_turning_maneuver(self, vessel: VesselSnapshot, maneuver_event: ManeuverEvent, 
                               current_time: datetime):
        """
        Apply turning maneuver to vessel at current time.
        
        Args:
            vessel: Vessel to modify
            maneuver_event: Maneuver being applied
            current_time: Current simulation time
        """
        # Calculate progress through maneuver
        maneuver_duration = (maneuver_event.end_time - maneuver_event.start_time).total_seconds()
        elapsed_time = (current_time - maneuver_event.start_time).total_seconds()
        
        if maneuver_duration <= 0:
            progress = 1.0
        else:
            progress = min(1.0, elapsed_time / maneuver_duration)
        
        # Interpolate course change
        course_change = (maneuver_event.target_cog - maneuver_event.original_cog) * progress
        new_cog = (maneuver_event.original_cog + course_change) % 360
        
        # Update vessel course and heading
        vessel.cog = new_cog
        vessel.heading = new_cog
        
        # Update position based on new course
        self._update_vessel_position(vessel, current_time)
    
    def _apply_post_maneuver_course(self, vessel: VesselSnapshot, maneuver_event: ManeuverEvent, 
                                   current_time: datetime):
        """
        Apply post-maneuver course to vessel.
        
        Args:
            vessel: Vessel to modify
            maneuver_event: Completed maneuver
            current_time: Current simulation time
        """
        # Maintain target course
        vessel.cog = maneuver_event.target_cog
        vessel.heading = maneuver_event.target_cog
        
        # Update position
        self._update_vessel_position(vessel, current_time)
    
    def _update_vessel_position(self, vessel: VesselSnapshot, current_time: datetime):
        """
        Update vessel position based on current course and speed.
        
        This is a simplified position update for simulation purposes.
        
        Args:
            vessel: Vessel to update
            current_time: Current simulation time
        """
        # Calculate distance traveled in this time step
        time_step_hours = self.time_step_seconds / 3600.0
        distance_nm = vessel.sog * time_step_hours
        
        if distance_nm > 0:
            # Convert course to radians
            course_rad = math.radians(vessel.cog)
            
            # Calculate lat/lon changes
            # Simplified calculation - not accurate for large distances
            lat_change = (distance_nm / 60.0) * math.cos(course_rad)
            lon_change = (distance_nm / 60.0) * math.sin(course_rad) / math.cos(math.radians(vessel.lat))
            
            # Update position
            vessel.lat += lat_change
            vessel.lon += lon_change
    
    def get_all_maneuver_events(self) -> List[ManeuverEvent]:
        """Get all completed maneuver events."""
        return self.completed_maneuvers.copy()
    
    def get_maneuvers_for_vessel(self, vessel_mmsi: int) -> List[ManeuverEvent]:
        """Get all maneuvers for a specific vessel."""
        return [m for m in self.completed_maneuvers if m.vessel_mmsi == vessel_mmsi]
    
    def _create_maneuver_for_ghost_threat(self, threat: GhostThreat, 
                                         snapshots: Dict[datetime, TrafficSnapshot]) -> Optional[ManeuverEvent]:
        """
        Create maneuver in response to ghost vessel threat.
        
        Args:
            threat: GhostThreat requiring maneuver
            snapshots: Snapshots to find vessel state
            
        Returns:
            ManeuverEvent or None
        """
        # Find vessel snapshot at threat time
        vessel_snapshot = None
        for timestamp, snapshot in sorted(snapshots.items()):
            if timestamp >= threat.timestamp:
                vessel_snapshot = snapshot.get_vessel(threat.real_mmsi)
                if vessel_snapshot:
                    break
        
        if not vessel_snapshot:
            logger.warning(f"Could not find vessel {threat.real_mmsi} for ghost threat maneuver")
            return None
        
        # Always use starboard turn for ghost threats
        maneuver_type = 'turn_starboard'
        
        # Calculate target course
        original_cog = vessel_snapshot.cog
        target_cog = (original_cog + self.standard_turn_degrees) % 360
        
        # Calculate maneuver duration
        degrees_to_turn = self.standard_turn_degrees
        maneuver_duration_seconds = (degrees_to_turn / self.max_turn_rate_deg_per_min) * 60
        
        start_time = threat.timestamp
        end_time = start_time + timedelta(seconds=maneuver_duration_seconds)
        
        # Create a mock conflict for compatibility
        # In a real implementation, we'd have a separate GhostConflict class
        mock_conflict = type('obj', (object,), {
            'vessel1': vessel_snapshot,
            'vessel2': type('obj', (object,), {'mmsi': threat.ghost_mmsi})(),
            'cpa_distance_nm': threat.cpa_nm,
            'tcpa_seconds': threat.tcpa_min * 60,
            'timestamp': threat.timestamp,
            'give_way_vessel': vessel_snapshot,
            'stand_on_vessel': type('obj', (object,), {'mmsi': threat.ghost_mmsi})(),
            'to_dict': lambda: {
                'timestamp': threat.timestamp.isoformat(),
                'vessel1_mmsi': threat.real_mmsi,
                'vessel2_mmsi': threat.ghost_mmsi,
                'cpa_distance_nm': threat.cpa_nm,
                'tcpa_seconds': threat.tcpa_min * 60,
                'give_way_vessel': threat.real_mmsi,
                'stand_on_vessel': threat.ghost_mmsi,
                'is_critical': True,
                'trigger_type': 'ghost_threat'
            }
        })()
        
        return ManeuverEvent(
            vessel_mmsi=threat.real_mmsi,
            maneuver_type=maneuver_type,
            start_time=start_time,
            end_time=end_time,
            trigger_conflict=mock_conflict,
            original_cog=original_cog,
            target_cog=target_cog,
            trigger_type='ghost_threat'
        )