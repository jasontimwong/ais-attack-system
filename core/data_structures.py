"""
Core data structures for AIS records and events.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class AISRecord:
    """Represents a single AIS message/record."""
    timestamp: datetime
    mmsi: int
    lat: float
    lon: float
    sog: float  # Speed Over Ground (knots)
    cog: float  # Course Over Ground (degrees)
    rot: float  # Rate of Turn (degrees/minute)
    nav_status: int  # Navigation status code
    true_heading: Optional[float] = None
    special_manoeuvre: Optional[int] = None
    spare: Optional[int] = None
    raim: Optional[int] = None
    comm_state: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'mmsi': self.mmsi,
            'lat': self.lat,
            'lon': self.lon,
            'sog': self.sog,
            'cog': self.cog,
            'rot': self.rot,
            'nav_status': self.nav_status,
            'true_heading': self.true_heading,
            'special_manoeuvre': self.special_manoeuvre,
            'spare': self.spare,
            'raim': self.raim,
            'comm_state': self.comm_state
        }
    
    def copy(self) -> 'AISRecord':
        """Create a deep copy of this record."""
        return AISRecord(**self.to_dict())


@dataclass
class EventLog:
    """Represents an attack event with metadata and impact metrics."""
    event_id: int
    attack_type: str
    impact_type: str
    start_ts: datetime
    end_ts: datetime
    metrics: Dict[str, Any]
    target_mmsi: Optional[int] = None
    secondary_mmsi: Optional[int] = None  # For identity swap
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event log to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'attack_type': self.attack_type,
            'impact_type': self.impact_type,
            'start_ts': self.start_ts.isoformat(),
            'end_ts': self.end_ts.isoformat(),
            'metrics': self.metrics,
            'target_mmsi': self.target_mmsi,
            'secondary_mmsi': self.secondary_mmsi
        }


@dataclass
class AttackResult:
    """Result returned by attack plugins."""
    event_log: EventLog
    modified_records: List[AISRecord]
    new_records: Optional[List[AISRecord]] = None
    deleted_record_indices: Optional[List[int]] = None