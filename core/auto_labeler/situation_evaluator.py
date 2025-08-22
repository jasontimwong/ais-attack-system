"""
Situation Evaluator for Impact Simulator.

Detects collision risks using CPA (Closest Point of Approach) and 
TCPA (Time to Closest Point of Approach) calculations.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from .traffic_snapshot_builder import VesselSnapshot, TrafficSnapshot

logger = logging.getLogger(__name__)


class ConflictSituation:
    """Represents a potential collision situation between two vessels."""
    
    def __init__(self, vessel1: VesselSnapshot, vessel2: VesselSnapshot, 
                 cpa_distance: float, tcpa_seconds: float, timestamp: datetime,
                 threat_tag: Optional[str] = None):
        self.vessel1 = vessel1
        self.vessel2 = vessel2
        self.cpa_distance_nm = cpa_distance
        self.tcpa_seconds = tcpa_seconds
        self.timestamp = timestamp
        self.threat_tag = threat_tag  # e.g., 'ghost' for ghost vessel threats
        
        # Determine give-way vessel based on rules of the road
        self.give_way_vessel = self._determine_give_way_vessel()
        self.stand_on_vessel = vessel2 if self.give_way_vessel == vessel1 else vessel1
        
    def _determine_give_way_vessel(self) -> VesselSnapshot:
        """
        Determine which vessel should give way based on simplified rules.
        
        For MVP, we'll use a simple heuristic:
        - Vessel with higher speed gives way
        - If speeds are similar, vessel with higher MMSI gives way
        """
        if abs(self.vessel1.sog - self.vessel2.sog) > 1.0:
            return self.vessel1 if self.vessel1.sog > self.vessel2.sog else self.vessel2
        else:
            return self.vessel1 if self.vessel1.mmsi > self.vessel2.mmsi else self.vessel2
    
    def is_critical(self, cpa_threshold_nm: float = 1.0, tcpa_threshold_seconds: float = 900) -> bool:
        """
        Check if this situation requires immediate action.
        
        Args:
            cpa_threshold_nm: CPA distance threshold in nautical miles
            tcpa_threshold_seconds: TCPA time threshold in seconds
            
        Returns:
            True if situation is critical
        """
        return (self.cpa_distance_nm < cpa_threshold_nm and 
                0 < self.tcpa_seconds < tcpa_threshold_seconds)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'vessel1_mmsi': self.vessel1.mmsi,
            'vessel2_mmsi': self.vessel2.mmsi,
            'cpa_distance_nm': self.cpa_distance_nm,
            'tcpa_seconds': self.tcpa_seconds,
            'give_way_vessel': self.give_way_vessel.mmsi,
            'stand_on_vessel': self.stand_on_vessel.mmsi,
            'is_critical': self.is_critical()
        }


class SituationEvaluator:
    """
    Evaluates traffic situations to detect collision risks.
    
    Uses CPA/TCPA calculations to identify when vessels need to take
    evasive action to avoid collisions.
    """
    
    def __init__(self, cpa_threshold_nm: float = 1.0, tcpa_threshold_seconds: float = 900):
        """
        Initialize situation evaluator.
        
        Args:
            cpa_threshold_nm: CPA distance threshold in nautical miles (default: 1.0)
            tcpa_threshold_seconds: TCPA time threshold in seconds (default: 900 = 15 minutes)
        """
        self.cpa_threshold_nm = cpa_threshold_nm
        self.tcpa_threshold_seconds = tcpa_threshold_seconds
        
    def evaluate_snapshot(self, snapshot: TrafficSnapshot) -> List[ConflictSituation]:
        """
        Evaluate all vessel pairs in a snapshot for collision risks.
        
        Args:
            snapshot: TrafficSnapshot containing vessel positions
            
        Returns:
            List of ConflictSituation objects
        """
        conflicts = []
        vessels = snapshot.get_all_vessels()
        
        # Check all vessel pairs
        for i in range(len(vessels)):
            for j in range(i + 1, len(vessels)):
                vessel1 = vessels[i]
                vessel2 = vessels[j]
                
                # Calculate CPA and TCPA
                cpa_distance, tcpa_seconds = self._calculate_cpa_tcpa(vessel1, vessel2)
                
                if cpa_distance is not None and tcpa_seconds is not None:
                    conflict = ConflictSituation(
                        vessel1, vessel2, cpa_distance, tcpa_seconds, snapshot.timestamp
                    )
                    
                    # Only include critical situations
                    if conflict.is_critical(self.cpa_threshold_nm, self.tcpa_threshold_seconds):
                        conflicts.append(conflict)
        
        return conflicts
    
    def _calculate_cpa_tcpa(self, vessel1: VesselSnapshot, vessel2: VesselSnapshot) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate CPA (Closest Point of Approach) and TCPA (Time to CPA).
        
        Args:
            vessel1: First vessel
            vessel2: Second vessel
            
        Returns:
            Tuple of (CPA distance in nm, TCPA in seconds) or (None, None) if no approach
        """
        try:
            # Convert positions to Cartesian coordinates (approximate)
            x1, y1 = self._lat_lon_to_xy(vessel1.lat, vessel1.lon)
            x2, y2 = self._lat_lon_to_xy(vessel2.lat, vessel2.lon)
            
            # Calculate velocity vectors (in nm/hour, then convert to nm/second)
            v1x = vessel1.sog * math.cos(math.radians(vessel1.cog)) / 3600
            v1y = vessel1.sog * math.sin(math.radians(vessel1.cog)) / 3600
            v2x = vessel2.sog * math.cos(math.radians(vessel2.cog)) / 3600
            v2y = vessel2.sog * math.sin(math.radians(vessel2.cog)) / 3600
            
            # Relative position and velocity
            dx = x2 - x1
            dy = y2 - y1
            dvx = v2x - v1x
            dvy = v2y - v1y
            
            # Calculate TCPA
            dv_squared = dvx * dvx + dvy * dvy
            if dv_squared < 1e-10:  # Vessels have same velocity
                # CPA is current distance
                cpa_distance = math.sqrt(dx * dx + dy * dy)
                return cpa_distance, float('inf')
            
            tcpa_seconds = -(dx * dvx + dy * dvy) / dv_squared
            
            # If TCPA is negative, vessels are moving apart
            if tcpa_seconds < 0:
                return None, None
            
            # Calculate CPA distance
            cpa_x = dx + dvx * tcpa_seconds
            cpa_y = dy + dvy * tcpa_seconds
            cpa_distance = math.sqrt(cpa_x * cpa_x + cpa_y * cpa_y)
            
            return cpa_distance, tcpa_seconds
            
        except Exception as e:
            logger.warning(f"Error calculating CPA/TCPA for vessels {vessel1.mmsi}/{vessel2.mmsi}: {e}")
            return None, None
    
    def _lat_lon_to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert lat/lon to approximate Cartesian coordinates.
        
        This is a simplified conversion suitable for local areas.
        For more accurate results over large distances, use proper projections.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Tuple of (x, y) in nautical miles
        """
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # Earth radius in nautical miles
        earth_radius_nm = 3440.065
        
        # Simple equirectangular projection
        x = earth_radius_nm * lon_rad * math.cos(lat_rad)
        y = earth_radius_nm * lat_rad
        
        return x, y
    
    def find_target_vessel_conflicts(self, snapshots: Dict[datetime, TrafficSnapshot], 
                                   target_mmsi: int) -> List[ConflictSituation]:
        """
        Find all conflicts involving a specific target vessel.
        
        Args:
            snapshots: Dictionary of traffic snapshots
            target_mmsi: MMSI of target vessel to analyze
            
        Returns:
            List of ConflictSituation objects involving the target vessel
        """
        target_conflicts = []
        
        for timestamp, snapshot in snapshots.items():
            conflicts = self.evaluate_snapshot(snapshot)
            
            for conflict in conflicts:
                if (conflict.vessel1.mmsi == target_mmsi or 
                    conflict.vessel2.mmsi == target_mmsi):
                    target_conflicts.append(conflict)
        
        return target_conflicts
    
    def get_most_critical_conflict(self, conflicts: List[ConflictSituation]) -> Optional[ConflictSituation]:
        """
        Get the most critical conflict from a list.
        
        Args:
            conflicts: List of ConflictSituation objects
            
        Returns:
            Most critical conflict or None
        """
        if not conflicts:
            return None
        
        # Sort by CPA distance (ascending) then by TCPA (ascending)
        critical_conflicts = sorted(conflicts, 
                                  key=lambda c: (c.cpa_distance_nm, c.tcpa_seconds))
        
        return critical_conflicts[0]