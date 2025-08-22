"""
S2 Zone Violation Attack Plugin - VLCC tanker enters prohibited zone.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Point, Polygon

from core.attack_plugin import AttackPlugin

logger = logging.getLogger(__name__)


class S2ZoneViolationPlugin(AttackPlugin):
    """
    Implements zone violation attack where a ghost VLCC tanker
    enters and traverses a prohibited maritime zone.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.attack_config = config.get('attack_config', {})
        self.ghost_config = config.get('ghost_vessel', {})
        self.zone_config = config.get('prohibited_zone', {})
        
        # Create zone polygon
        zone_coords = self.zone_config.get('polygon', [])
        self.zone_polygon = Polygon([(lat, lon) for lat, lon in zone_coords])
        
        # Ghost vessel parameters
        self.ghost_mmsi = self.ghost_config.get('mmsi', 636012345)
        self.vessel_name = self.ghost_config.get('name', 'PHANTOM VOYAGER')
        
        # Movement parameters
        self.speed_range = self.ghost_config.get('speed_range', {'min': 10, 'max': 15})
        self.entry_point = self.ghost_config.get('entry_point', {'lat': 30.0, 'lon': -123.0})
        self.exit_point = self.ghost_config.get('exit_point', {'lat': 30.0, 'lon': -121.0})
        
        # Zone violation timing
        self.zone_entry_time = pd.to_datetime(self.attack_config.get('zone_entry_time')).tz_localize(None)
        self.min_time_in_zone = self.attack_config.get('min_time_in_zone', 20)  # minutes
        self.max_time_in_zone = self.attack_config.get('max_time_in_zone', 120)
        
        logger.info(f"S2 Zone Violation attack initialized for zone: {self.zone_config.get('name')}")
        
    def inject_attack(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject ghost VLCC tanker that violates prohibited zone.
        """
        logger.info(f"Injecting zone violation attack with ghost VLCC {self.ghost_mmsi}")
        
        # Calculate trajectory
        trajectory = self._calculate_zone_crossing_trajectory()
        
        if len(trajectory) == 0:
            logger.error("Failed to generate zone crossing trajectory")
            return baseline_data
            
        # Create ghost vessel AIS messages
        ghost_messages = self._create_ghost_messages(trajectory)
        
        # Combine with baseline
        attack_data = pd.concat([baseline_data, ghost_messages], ignore_index=True)
        attack_data = attack_data.sort_values('BaseDateTime').reset_index(drop=True)
        
        # Log statistics
        time_in_zone = self._calculate_time_in_zone(trajectory)
        logger.info(f"Generated {len(ghost_messages)} ghost messages")
        logger.info(f"Ghost vessel in zone for {time_in_zone:.1f} minutes")
        logger.info(f"Zone entry: {trajectory[0]['entry_time']}")
        logger.info(f"Zone exit: {trajectory[-1]['exit_time'] if trajectory[-1].get('in_zone') else 'Still in zone'}")
        
        return attack_data
    
    def _calculate_zone_crossing_trajectory(self) -> List[Dict]:
        """
        Calculate trajectory that crosses through the prohibited zone.
        """
        trajectory = []
        
        # Start before zone
        current_time = self.zone_entry_time - timedelta(hours=2)
        current_lat = self.entry_point['lat']
        current_lon = self.entry_point['lon']
        
        # Calculate course to exit point
        dlat = self.exit_point['lat'] - self.entry_point['lat']
        dlon = self.exit_point['lon'] - self.entry_point['lon']
        course = np.degrees(np.arctan2(dlon, dlat)) % 360
        
        # Speed (constant for simplicity)
        speed = (self.speed_range['min'] + self.speed_range['max']) / 2
        
        # Time step (3 minutes for Class A AIS)
        time_step = timedelta(seconds=180)
        
        # Generate trajectory
        entered_zone = False
        entry_time = None
        exit_time = None
        
        while current_time < self.zone_entry_time + timedelta(hours=4):
            # Check if in zone
            point = Point(current_lat, current_lon)
            in_zone = self.zone_polygon.contains(point)
            
            # Track entry/exit times
            if in_zone and not entered_zone:
                entered_zone = True
                entry_time = current_time
                logger.info(f"Ghost vessel entered zone at {entry_time}")
            elif not in_zone and entered_zone and exit_time is None:
                exit_time = current_time
                logger.info(f"Ghost vessel exited zone at {exit_time}")
            
            # Add point to trajectory
            trajectory.append({
                'time': current_time,
                'lat': current_lat,
                'lon': current_lon,
                'speed': speed,
                'course': course,
                'in_zone': in_zone,
                'entry_time': entry_time,
                'exit_time': exit_time
            })
            
            # Move vessel
            # Simplified movement (should use proper geodesic calculations)
            distance_nm = speed * (time_step.total_seconds() / 3600)
            dlat_step = distance_nm * np.cos(np.radians(course)) / 60
            dlon_step = distance_nm * np.sin(np.radians(course)) / 60
            
            current_lat += dlat_step
            current_lon += dlon_step
            current_time += time_step
            
            # Stop if far past exit point
            if current_lon > self.exit_point['lon'] + 1.0:
                break
        
        return trajectory
    
    def _create_ghost_messages(self, trajectory: List[Dict]) -> pd.DataFrame:
        """
        Create AIS messages for ghost vessel.
        """
        messages = []
        
        for point in trajectory:
            msg = {
                'MMSI': self.ghost_mmsi,
                'BaseDateTime': point['time'],
                'LAT': point['lat'],
                'LON': point['lon'],
                'SOG': point['speed'],
                'COG': point['course'],
                'Heading': point['course'],  # Same as COG for straight line
                'VesselName': self.vessel_name,
                'IMO': self.ghost_config.get('imo', 'IMO9999999'),
                'CallSign': self.ghost_config.get('callsign', 'A8QZ9'),
                'VesselType': self.ghost_config.get('vessel_type', 80),
                'Status': 0,  # Under way using engine
                'Length': self.ghost_config.get('length', 333),
                'Width': self.ghost_config.get('width', 60),
                'Draft': self.ghost_config.get('draft', 22.5),
                'Cargo': 80,  # Tanker
                'TransceiverClass': 'A'
            }
            messages.append(msg)
        
        return pd.DataFrame(messages)
    
    def _calculate_time_in_zone(self, trajectory: List[Dict]) -> float:
        """
        Calculate total time spent in prohibited zone (minutes).
        """
        time_in_zone = 0
        last_time = None
        
        for point in trajectory:
            if point['in_zone']:
                if last_time is not None:
                    time_in_zone += (point['time'] - last_time).total_seconds() / 60
                last_time = point['time']
            else:
                last_time = None
                
        return time_in_zone
    
    def validate(self, attack_data: pd.DataFrame) -> bool:
        """
        Validate that zone violation meets requirements.
        """
        ghost_data = attack_data[attack_data.MMSI == self.ghost_mmsi]
        
        if len(ghost_data) == 0:
            logger.error("No ghost vessel data found")
            return False
        
        # Check zone violation
        violations = 0
        for _, row in ghost_data.iterrows():
            point = Point(row['LAT'], row['LON'])
            if self.zone_polygon.contains(point):
                violations += 1
        
        if violations == 0:
            logger.error("Ghost vessel did not enter prohibited zone")
            return False
        
        # Calculate dwell time
        time_in_zone = violations * 3  # minutes (3 min intervals)
        if time_in_zone < self.min_time_in_zone:
            logger.error(f"Insufficient time in zone: {time_in_zone} < {self.min_time_in_zone} minutes")
            return False
        
        logger.info(f"Zone violation validated: {time_in_zone} minutes in zone")
        return True
    
    def apply(self, data, config):
        """Required abstract method from AttackPlugin."""
        # This method is not used in our implementation
        # We use inject_attack instead
        pass