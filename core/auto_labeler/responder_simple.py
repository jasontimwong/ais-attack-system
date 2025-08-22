"""
Simple vessel response simulator for drift attack scenario.
Optimized for 12s AIS data and quick CPA calculation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VesselResponder:
    """Simulates vessel collision avoidance responses with cascade effects"""
    
    # COLREGS-based response thresholds
    CPA_THRESHOLD_NM = 0.5  # Trigger maneuver when CPA <= 0.5 nm
    TCPA_THRESHOLD_MIN = 15  # Minutes
    
    # Maneuver parameters
    EMERGENCY_TURN_DEG = 30  # Emergency turn angle
    SPEED_REDUCTION = 0.2  # 20% speed reduction
    
    def __init__(self):
        self.cascade_events = []
        self.affected_vessels = set()
    
    def simulate_responses(self, baseline_df: pd.DataFrame, attack_df: pd.DataFrame,
                          ghost_mmsi: int, attack_start: datetime, attack_end: datetime,
                          cascade_levels: int = 2) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Simulate vessel responses to attack with cascade effects.
        """
        logger.info(f"Simulating vessel responses with {cascade_levels} cascade levels")
        
        # Initialize
        impact_df = attack_df.copy()
        self.cascade_events = []
        self.affected_vessels = {ghost_mmsi}
        
        # Get ghost vessel trajectory
        ghost_data = attack_df[
            (attack_df['MMSI'] == ghost_mmsi) &
            (attack_df['BaseDateTime'] >= attack_start) &
            (attack_df['BaseDateTime'] <= attack_end)
        ]
        
        if len(ghost_data) == 0:
            logger.warning("No ghost vessel data found")
            return impact_df, []
        
        # Find target vessel
        target_mmsi = None
        min_dist = float('inf')
        
        # Check all vessels in the area
        area_vessels = attack_df[
            (attack_df['BaseDateTime'] >= attack_start) &
            (attack_df['BaseDateTime'] <= attack_end) &
            (attack_df['MMSI'] != ghost_mmsi)
        ]['MMSI'].unique()
        
        logger.info(f"Found {len(area_vessels)} vessels in attack window")
        
        # Find vessels that come close to ghost
        for vessel_mmsi in area_vessels[:100]:  # Limit to first 100 vessels for speed
            vessel_data = attack_df[
                (attack_df['MMSI'] == vessel_mmsi) &
                (attack_df['BaseDateTime'] >= attack_start) &
                (attack_df['BaseDateTime'] <= attack_end)
            ]
            
            if len(vessel_data) < 5:  # Need enough data points
                continue
            
            # Quick CPA check
            cpa = self._quick_cpa_check(ghost_data, vessel_data)
            if cpa < min_dist:
                min_dist = cpa
                target_mmsi = vessel_mmsi
        
        logger.info(f"Closest vessel: MMSI {target_mmsi} at {min_dist:.3f} nm")
        
        # If close enough, add to events
        if min_dist < self.CPA_THRESHOLD_NM and target_mmsi:
            event = {
                'mmsi': target_mmsi,
                'threat_mmsi': ghost_mmsi,
                'cpa_nm': min_dist,
                'tcpa_min': 5.0,  # Simplified
                'cpa_time': attack_start + timedelta(minutes=30),
                'reaction_time': attack_start + timedelta(minutes=25),
                'vessel_name': f"Vessel_{target_mmsi}",
                'cascade_level': 1
            }
            
            # Apply response
            impact_df = self._apply_vessel_response(impact_df, event, 1)
            self.cascade_events.append(event)
            self.affected_vessels.add(target_mmsi)
        
        logger.info(f"Total cascade events: {len(self.cascade_events)}")
        return impact_df, self.cascade_events
    
    def _quick_cpa_check(self, vessel1_data: pd.DataFrame, 
                        vessel2_data: pd.DataFrame) -> float:
        """Quick CPA calculation using closest time points"""
        min_distance = float('inf')
        
        # Sample every 5th point for speed
        v1_sample = vessel1_data.iloc[::5]
        
        for _, v1_row in v1_sample.iterrows():
            # Find closest time in vessel2
            time_diffs = abs(vessel2_data['BaseDateTime'] - v1_row['BaseDateTime'])
            if len(time_diffs) == 0:
                continue
            
            nearest_idx = time_diffs.idxmin()
            if time_diffs[nearest_idx] > pd.Timedelta(minutes=2):
                continue
            
            v2_row = vessel2_data.loc[nearest_idx]
            
            # Calculate distance
            distance = self._haversine_distance(
                v1_row['LAT'], v1_row['LON'],
                v2_row['LAT'], v2_row['LON']
            )
            
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _apply_vessel_response(self, df: pd.DataFrame, vessel_info: Dict,
                             cascade_level: int) -> pd.DataFrame:
        """Apply collision avoidance maneuver to vessel"""
        mmsi = vessel_info['mmsi']
        reaction_time = vessel_info['reaction_time']
        
        logger.info(f"Applying response for MMSI {mmsi} at cascade level {cascade_level}")
        
        # Get vessel data after reaction time
        vessel_mask = (df['MMSI'] == mmsi) & (df['BaseDateTime'] >= reaction_time)
        
        if vessel_mask.sum() == 0:
            return df
        
        # Apply course change
        original_cog = df.loc[vessel_mask, 'COG'].iloc[0]
        new_cog = (original_cog + self.EMERGENCY_TURN_DEG) % 360
        df.loc[vessel_mask, 'COG'] = new_cog
        
        if 'Heading' in df.columns:
            df.loc[vessel_mask, 'Heading'] = new_cog.astype(int)
        
        # Apply speed reduction
        original_sog = df.loc[vessel_mask, 'SOG'].iloc[0]
        df.loc[vessel_mask, 'SOG'] = original_sog * (1 - self.SPEED_REDUCTION)
        
        # Update event info
        vessel_info['delta_heading'] = self.EMERGENCY_TURN_DEG
        vessel_info['speed_drop'] = original_sog * self.SPEED_REDUCTION
        vessel_info['initial_speed'] = original_sog
        vessel_info['action'] = 'emergency_turn'
        
        return df
    
    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance in nautical miles"""
        R = 3440.065
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def get_impact_events(self, ghost_mmsi: int, target_mmsi: int,
                         attack_window: Dict) -> Dict:
        """Format cascade events for output"""
        impact_data = {
            "scenario": "s1_drift", 
            "ghost_mmsi": ghost_mmsi,
            "target_mmsi": target_mmsi,
            "attack_window": attack_window,
            "impacted_vessels": len(self.cascade_events),
            "cascade_levels": max([e.get('cascade_level', 1) for e in self.cascade_events]) if self.cascade_events else 0,
            "events": []
        }
        
        # Format events
        for event in self.cascade_events:
            impact_data["events"].append({
                "mmsi": event['mmsi'],
                "vessel_name": event.get('vessel_name', f"Vessel_{event['mmsi']}"),
                "cascade_level": event.get('cascade_level', 1),
                "threat_mmsi": event.get('threat_mmsi'),
                "cpa_nm": round(event['cpa_nm'], 3),
                "cpa_time": str(event['cpa_time']),
                "reaction_time": str(event['reaction_time']),
                "maneuver_time": str(event['reaction_time']),
                "delta_heading": event.get('delta_heading', 0),
                "speed_drop": round(event.get('speed_drop', 0), 1),
                "initial_speed": round(event.get('initial_speed', 0), 1),
                "action": event.get('action', 'unknown')
            })
        
        return impact_data