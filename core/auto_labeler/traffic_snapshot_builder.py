"""
Traffic Snapshot Builder for Impact Simulator.

Converts attack.csv into time-stepped snapshots for simulation processing.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VesselSnapshot:
    """Represents a vessel's state at a specific time."""
    
    def __init__(self, mmsi: int, timestamp: datetime, lat: float, lon: float, 
                 sog: float, cog: float, heading: float = None, vessel_name: str = None):
        self.mmsi = mmsi
        self.timestamp = timestamp
        self.lat = lat
        self.lon = lon
        self.sog = sog  # Speed Over Ground (knots)
        self.cog = cog  # Course Over Ground (degrees)
        self.heading = heading if heading is not None else cog
        self.vessel_name = vessel_name or f"VESSEL_{mmsi}"
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'MMSI': self.mmsi,
            'BaseDateTime': self.timestamp.isoformat(),
            'LAT': self.lat,
            'LON': self.lon,
            'SOG': self.sog,
            'COG': self.cog,
            'Heading': self.heading,
            'VesselName': self.vessel_name
        }


class TrafficSnapshot:
    """Represents all vessels at a specific time."""
    
    def __init__(self, timestamp: datetime):
        self.timestamp = timestamp
        self.vessels: Dict[int, VesselSnapshot] = {}
        
    def add_vessel(self, vessel: VesselSnapshot):
        """Add a vessel to this snapshot."""
        self.vessels[vessel.mmsi] = vessel
        
    def get_vessel(self, mmsi: int) -> Optional[VesselSnapshot]:
        """Get a vessel by MMSI."""
        return self.vessels.get(mmsi)
        
    def get_all_vessels(self) -> List[VesselSnapshot]:
        """Get all vessels in this snapshot."""
        return list(self.vessels.values())


class TrafficSnapshotBuilder:
    """
    Builds time-stepped traffic snapshots from attack.csv data.
    
    Processes AIS data into regular time intervals (default 10 seconds) 
    to enable systematic conflict detection and maneuver simulation.
    """
    
    def __init__(self, time_step_seconds: int = 10):
        """
        Initialize the snapshot builder.
        
        Args:
            time_step_seconds: Time interval between snapshots (default: 10s)
        """
        self.time_step_seconds = time_step_seconds
        self.snapshots: Dict[datetime, TrafficSnapshot] = {}
        
    def build_snapshots(self, attack_csv_path: str) -> Dict[datetime, TrafficSnapshot]:
        """
        Build traffic snapshots from attack.csv file.
        
        Args:
            attack_csv_path: Path to attack.csv file
            
        Returns:
            Dictionary mapping timestamp to TrafficSnapshot
        """
        logger.info(f"Building traffic snapshots from {attack_csv_path}")
        
        # Load attack data
        df = pd.read_csv(attack_csv_path)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        
        return self.build_snapshots_from_dataframe(df)
    
    def build_snapshots_from_dataframe(self, df: pd.DataFrame) -> Dict[datetime, TrafficSnapshot]:
        """
        Build traffic snapshots from a DataFrame.
        
        Args:
            df: DataFrame with AIS data
            
        Returns:
            Dictionary mapping timestamp to TrafficSnapshot
        """
        # Ensure BaseDateTime is datetime type
        if 'BaseDateTime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['BaseDateTime']):
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        
        # Sort by timestamp
        df = df.sort_values('BaseDateTime')
        
        # Get time range
        start_time = df['BaseDateTime'].min()
        end_time = df['BaseDateTime'].max()
        
        logger.info(f"Time range: {start_time} to {end_time}")
        
        # Create snapshots at regular intervals
        current_time = start_time
        while current_time <= end_time:
            self._build_snapshot_at_time(df, current_time)
            current_time += timedelta(seconds=self.time_step_seconds)
            
        logger.info(f"Created {len(self.snapshots)} traffic snapshots")
        return self.snapshots
    
    def _build_snapshot_at_time(self, df: pd.DataFrame, target_time: datetime):
        """
        Build a snapshot at a specific time by interpolating vessel positions.
        
        Args:
            df: DataFrame with AIS data
            target_time: Target timestamp for snapshot
        """
        snapshot = TrafficSnapshot(target_time)
        
        # Group by vessel (MMSI)
        for mmsi, vessel_data in df.groupby('MMSI'):
            vessel_data = vessel_data.sort_values('BaseDateTime')
            
            # Find closest records before and after target time
            before_records = vessel_data[vessel_data['BaseDateTime'] <= target_time]
            after_records = vessel_data[vessel_data['BaseDateTime'] > target_time]
            
            if len(before_records) == 0 and len(after_records) == 0:
                continue
                
            # Use interpolation or nearest record
            vessel_state = self._interpolate_vessel_state(
                before_records, after_records, target_time
            )
            
            if vessel_state:
                snapshot.add_vessel(vessel_state)
        
        self.snapshots[target_time] = snapshot
    
    def _interpolate_vessel_state(self, before_records: pd.DataFrame, 
                                after_records: pd.DataFrame, 
                                target_time: datetime) -> Optional[VesselSnapshot]:
        """
        Interpolate vessel state at target time.
        
        Args:
            before_records: Records before target time
            after_records: Records after target time
            target_time: Target timestamp
            
        Returns:
            Interpolated VesselSnapshot or None
        """
        if len(before_records) == 0 and len(after_records) > 0:
            # Use first available record
            record = after_records.iloc[0]
        elif len(after_records) == 0 and len(before_records) > 0:
            # Use last available record
            record = before_records.iloc[-1]
        elif len(before_records) > 0 and len(after_records) > 0:
            # Interpolate between records
            before_record = before_records.iloc[-1]
            after_record = after_records.iloc[0]
            
            # Simple linear interpolation
            time_diff = (after_record['BaseDateTime'] - before_record['BaseDateTime']).total_seconds()
            if time_diff <= 0:
                record = before_record
            else:
                target_offset = (target_time - before_record['BaseDateTime']).total_seconds()
                ratio = target_offset / time_diff
                
                # Interpolate position
                lat = before_record['LAT'] + ratio * (after_record['LAT'] - before_record['LAT'])
                lon = before_record['LON'] + ratio * (after_record['LON'] - before_record['LON'])
                
                # Use more recent values for course/speed
                record = after_record.copy()
                record['LAT'] = lat
                record['LON'] = lon
        else:
            return None
        
        # Create vessel snapshot
        return VesselSnapshot(
            mmsi=int(record['MMSI']),
            timestamp=target_time,
            lat=float(record['LAT']),
            lon=float(record['LON']),
            sog=float(record['SOG']),
            cog=float(record['COG']),
            heading=float(record.get('Heading', record['COG'])),
            vessel_name=str(record.get('VesselName', f"VESSEL_{record['MMSI']}"))
        )
    
    def get_snapshot_at_time(self, timestamp: datetime) -> Optional[TrafficSnapshot]:
        """Get snapshot at specific time."""
        return self.snapshots.get(timestamp)
    
    def get_all_snapshots(self) -> List[TrafficSnapshot]:
        """Get all snapshots sorted by time."""
        return [self.snapshots[ts] for ts in sorted(self.snapshots.keys())]
    
    def get_vessel_trajectory(self, mmsi: int) -> List[VesselSnapshot]:
        """
        Get complete trajectory for a specific vessel.
        
        Args:
            mmsi: Vessel MMSI
            
        Returns:
            List of VesselSnapshot objects in chronological order
        """
        trajectory = []
        for timestamp in sorted(self.snapshots.keys()):
            vessel = self.snapshots[timestamp].get_vessel(mmsi)
            if vessel:
                trajectory.append(vessel)
        return trajectory