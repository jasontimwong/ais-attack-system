"""
Ghost Watcher module for detecting ghost vessel threats.

Monitors traffic snapshots to identify real vessels threatened by ghost vessels
and marks them for avoidance maneuvers.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
import pandas as pd

from .traffic_snapshot_builder import TrafficSnapshot, VesselSnapshot
from .situation_evaluator import ConflictSituation
from ..metrics.cpa_utils import calc_simple_cpa

logger = logging.getLogger(__name__)


class GhostThreat:
    """Represents a threat from a ghost vessel to a real vessel."""
    
    def __init__(self, ghost_mmsi: int, real_mmsi: int, 
                 cpa_nm: float, tcpa_min: float, timestamp: datetime):
        self.ghost_mmsi = ghost_mmsi
        self.real_mmsi = real_mmsi
        self.cpa_nm = cpa_nm
        self.tcpa_min = tcpa_min
        self.timestamp = timestamp
        self.trigger_type = 'ghost_threat'
    
    def is_critical(self, cpa_threshold_nm: float = 1.0, 
                   tcpa_threshold_min: float = 15.0) -> bool:
        """Check if threat requires immediate action."""
        return (self.cpa_nm < cpa_threshold_nm and 
                0 < self.tcpa_min < tcpa_threshold_min)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'ghost_mmsi': self.ghost_mmsi,
            'real_mmsi': self.real_mmsi,
            'cpa_nm': self.cpa_nm,
            'tcpa_min': self.tcpa_min,
            'timestamp': self.timestamp.isoformat(),
            'trigger_type': self.trigger_type
        }


class GhostWatcher:
    """
    Watches for ghost vessels and identifies threatened real vessels.
    
    Ghost vessels are identified by MMSI patterns or from attack metadata.
    """
    
    def __init__(self, cpa_threshold_nm: float = 1.0, tcpa_threshold_min: float = 15.0,
                 ghost_mmsi_pattern: str = '999*'):
        """
        Initialize ghost watcher.
        
        Args:
            cpa_threshold_nm: CPA threshold for threat detection (nm)
            tcpa_threshold_min: TCPA threshold for threat detection (minutes)
            ghost_mmsi_pattern: Pattern to identify ghost vessels (e.g., '999*')
        """
        self.cpa_threshold_nm = cpa_threshold_nm
        self.tcpa_threshold_min = tcpa_threshold_min
        self.ghost_mmsi_pattern = ghost_mmsi_pattern
        self.known_ghosts: Set[int] = set()
        self.threat_history: List[GhostThreat] = []
        
    def identify_ghosts(self, snapshot: TrafficSnapshot, 
                       attack_metadata: Optional[Dict] = None) -> Set[int]:
        """
        Identify ghost vessels in a traffic snapshot.
        
        Args:
            snapshot: Traffic snapshot to analyze
            attack_metadata: Optional attack metadata containing ghost MMSIs
            
        Returns:
            Set of ghost vessel MMSIs
        """
        ghosts = set()
        
        # From attack metadata
        if attack_metadata:
            for event in attack_metadata.get('events', []):
                if event.get('attack_type') == 'S1':
                    # Try to get ghost MMSI from metrics
                    ghost_mmsi = event.get('metrics', {}).get('ghost_mmsi')
                    if ghost_mmsi:
                        ghosts.add(ghost_mmsi)
        
        # From MMSI pattern
        for vessel in snapshot.get_all_vessels():
            mmsi_str = str(vessel.mmsi)
            if self.ghost_mmsi_pattern.endswith('*'):
                prefix = self.ghost_mmsi_pattern[:-1]
                if mmsi_str.startswith(prefix):
                    ghosts.add(vessel.mmsi)
            elif vessel.mmsi == int(self.ghost_mmsi_pattern):
                ghosts.add(vessel.mmsi)
        
        # Update known ghosts
        self.known_ghosts.update(ghosts)
        
        return ghosts
    
    def find_threatened_vessels(self, snapshot: TrafficSnapshot,
                               ghost_mmsis: Optional[Set[int]] = None) -> Dict[int, List[GhostThreat]]:
        """
        Find real vessels threatened by ghost vessels.
        
        Args:
            snapshot: Traffic snapshot to analyze
            ghost_mmsis: Optional set of ghost MMSIs (uses known_ghosts if not provided)
            
        Returns:
            Dictionary mapping threatened vessel MMSI to list of threats
        """
        if ghost_mmsis is None:
            ghost_mmsis = self.known_ghosts
        
        if not ghost_mmsis:
            return {}
        
        threatened = {}
        all_vessels = snapshot.get_all_vessels()
        
        # Separate ghosts and real vessels
        ghost_vessels = [v for v in all_vessels if v.mmsi in ghost_mmsis]
        real_vessels = [v for v in all_vessels if v.mmsi not in ghost_mmsis]
        
        # Check each ghost-real vessel pair
        for ghost in ghost_vessels:
            for real_vessel in real_vessels:
                # Calculate CPA/TCPA
                cpa_nm, tcpa_min = calc_simple_cpa(
                    ghost.lat, ghost.lon, ghost.sog, ghost.cog,
                    real_vessel.lat, real_vessel.lon, real_vessel.sog, real_vessel.cog
                )
                
                # Create threat if within thresholds
                threat = GhostThreat(
                    ghost.mmsi, real_vessel.mmsi,
                    cpa_nm, tcpa_min, snapshot.timestamp
                )
                
                if threat.is_critical(self.cpa_threshold_nm, self.tcpa_threshold_min):
                    if real_vessel.mmsi not in threatened:
                        threatened[real_vessel.mmsi] = []
                    threatened[real_vessel.mmsi].append(threat)
                    self.threat_history.append(threat)
                    
                    logger.info(f"Ghost threat detected: Ghost {ghost.mmsi} threatens "
                              f"vessel {real_vessel.mmsi} (CPA={cpa_nm:.3f}nm, TCPA={tcpa_min:.1f}min)")
        
        return threatened
    
    def process_snapshot(self, snapshot: TrafficSnapshot,
                        attack_metadata: Optional[Dict] = None) -> Dict[int, List[GhostThreat]]:
        """
        Process a snapshot to identify ghosts and find threatened vessels.
        
        Args:
            snapshot: Traffic snapshot to process
            attack_metadata: Optional attack metadata
            
        Returns:
            Dictionary mapping threatened vessel MMSI to list of threats
        """
        # Identify ghost vessels
        ghosts = self.identify_ghosts(snapshot, attack_metadata)
        
        if ghosts:
            logger.info(f"Identified {len(ghosts)} ghost vessels: {sorted(ghosts)}")
        
        # Find threatened vessels
        threatened = self.find_threatened_vessels(snapshot, ghosts)
        
        return threatened
    
    def create_ghost_conflicts(self, snapshot: TrafficSnapshot, 
                              threatened_vessels: Dict[int, List[GhostThreat]]) -> List[ConflictSituation]:
        """
        Convert ghost threats to ConflictSituation objects with threat_tag='ghost'.
        
        Args:
            snapshot: Current traffic snapshot
            threatened_vessels: Dictionary of threatened vessels and their threats
            
        Returns:
            List of ConflictSituation objects
        """
        conflicts = []
        
        for real_mmsi, threats in threatened_vessels.items():
            real_vessel = snapshot.get_vessel(real_mmsi)
            if not real_vessel:
                continue
                
            for threat in threats:
                ghost_vessel = snapshot.get_vessel(threat.ghost_mmsi)
                if not ghost_vessel:
                    continue
                
                # Create conflict with ghost threat tag
                conflict = ConflictSituation(
                    vessel1=real_vessel,
                    vessel2=ghost_vessel,
                    cpa_distance=threat.cpa_nm,
                    tcpa_seconds=threat.tcpa_min * 60,  # Convert to seconds
                    timestamp=threat.timestamp,
                    threat_tag='ghost'
                )
                conflicts.append(conflict)
                
        return conflicts
    
    def get_most_critical_threat(self, vessel_mmsi: int) -> Optional[GhostThreat]:
        """
        Get the most critical ghost threat for a specific vessel.
        
        Args:
            vessel_mmsi: MMSI of the vessel to check
            
        Returns:
            Most critical threat or None
        """
        vessel_threats = [t for t in self.threat_history if t.real_mmsi == vessel_mmsi]
        
        if not vessel_threats:
            return None
        
        # Sort by CPA (ascending) then TCPA (ascending)
        vessel_threats.sort(key=lambda t: (t.cpa_nm, t.tcpa_min))
        
        return vessel_threats[0]
    
    def get_threat_summary(self) -> Dict:
        """Get summary of all detected threats."""
        
        # Group threats by real vessel
        threats_by_vessel = {}
        for threat in self.threat_history:
            if threat.real_mmsi not in threats_by_vessel:
                threats_by_vessel[threat.real_mmsi] = []
            threats_by_vessel[threat.real_mmsi].append(threat)
        
        return {
            'total_ghosts': len(self.known_ghosts),
            'ghost_mmsis': sorted(list(self.known_ghosts)),
            'threatened_vessels': len(threats_by_vessel),
            'total_threats': len(self.threat_history),
            'critical_threats': sum(1 for t in self.threat_history if t.is_critical()),
            'threats_by_vessel': {
                mmsi: {
                    'threat_count': len(threats),
                    'min_cpa_nm': min(t.cpa_nm for t in threats),
                    'earliest_tcpa_min': min(t.tcpa_min for t in threats)
                }
                for mmsi, threats in threats_by_vessel.items()
            }
        }
    
    def reset(self):
        """Reset the watcher state."""
        self.known_ghosts.clear()
        self.threat_history.clear()