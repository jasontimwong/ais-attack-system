"""
Vessel response simulator with cascade effect support.
Simulates realistic vessel maneuvers in response to collision threats.
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
    CPA_THRESHOLD_NM = 0.5  # Trigger maneuver when CPA <= 0.5 nm (increased for better detection)
    TCPA_THRESHOLD_MIN = 15  # Minutes (increased window)
    
    # Maneuver parameters
    TURN_RATE_DEG_PER_MIN = 3.0  # Degrees per minute
    SPEED_REDUCTION_RATE = 0.1  # 10% per minute
    EMERGENCY_TURN_DEG = 30  # Emergency turn angle
    
    def __init__(self):
        self.cascade_events = []
        self.affected_vessels = set()
    
    def simulate_responses(self, baseline_df: pd.DataFrame, attack_df: pd.DataFrame,
                          ghost_mmsi: int, attack_start: datetime, attack_end: datetime,
                          cascade_levels: int = 2) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Simulate vessel responses to attack with cascade effects.
        
        Args:
            baseline_df: Original vessel data
            attack_df: Data with ghost vessel injected
            ghost_mmsi: MMSI of ghost vessel
            attack_start: Attack window start
            attack_end: Attack window end
            cascade_levels: Number of cascade levels to simulate
            
        Returns:
            Tuple of (impact_df, cascade_events)
        """
        logger.info(f"Simulating vessel responses with {cascade_levels} cascade levels")
        
        # Initialize impact data
        impact_df = attack_df.copy()
        self.cascade_events = []
        self.affected_vessels = {ghost_mmsi}
        
        # Log initial conditions
        window_vessels = impact_df[
            (impact_df['BaseDateTime'] >= attack_start) & 
            (impact_df['BaseDateTime'] <= attack_end)
        ]['MMSI'].unique()
        logger.info(f"Found {len(window_vessels)} vessels in attack window")
        
        # Simulate each cascade level
        for level in range(cascade_levels):
            logger.info(f"Processing cascade level {level + 1}")
            
            # Find vessels that will react at this level
            reacting_vessels = self._find_reacting_vessels(
                impact_df, self.affected_vessels, attack_start, attack_end
            )
            
            if not reacting_vessels:
                logger.warning(f"No vessels reacting at level {level + 1}. Consider:")
                logger.warning("- Using denser traffic area")
                logger.warning("- Reducing CPA threshold")
                logger.warning("- Adding more ghost vessels")
                break
            
            # Simulate responses for each reacting vessel
            for vessel_info in reacting_vessels:
                impact_df = self._apply_vessel_response(
                    impact_df, vessel_info, cascade_level=level + 1
                )
                self.affected_vessels.add(vessel_info['mmsi'])
                self.cascade_events.append(vessel_info)
        
        logger.info(f"Total cascade events: {len(self.cascade_events)}")
        return impact_df, self.cascade_events
    
    def _find_reacting_vessels(self, df: pd.DataFrame, threat_vessels: set,
                              start_time: datetime, end_time: datetime) -> List[Dict]:
        """Find vessels that need to react to threats"""
        reacting_vessels = []
        
        # Get all vessels in time window
        window_df = df[(df['BaseDateTime'] >= start_time) & 
                      (df['BaseDateTime'] <= end_time)]
        
        # Check each potential reactor
        for mmsi in window_df['MMSI'].unique():
            if mmsi in self.affected_vessels:
                continue
            
            vessel_data = window_df[window_df['MMSI'] == mmsi]
            
            # Check CPA with each threat vessel
            for threat_mmsi in threat_vessels:
                threat_data = window_df[window_df['MMSI'] == threat_mmsi]
                
                if len(threat_data) == 0:
                    continue
                
                # Calculate CPA
                cpa_info = self._calculate_cpa(vessel_data, threat_data)
                
                if cpa_info and self._should_react(cpa_info):
                    vessel_info = {
                        'mmsi': mmsi,
                        'threat_mmsi': threat_mmsi,
                        'cpa_nm': cpa_info['cpa_nm'],
                        'tcpa_min': cpa_info['tcpa_min'],
                        'cpa_time': cpa_info['cpa_time'],
                        'reaction_time': cpa_info['cpa_time'] - timedelta(minutes=5),
                        'vessel_name': vessel_data['VesselName'].iloc[0] if 'VesselName' in vessel_data else f"Vessel_{mmsi}"
                    }
                    reacting_vessels.append(vessel_info)
                    break
        
        return reacting_vessels
    
    def _calculate_cpa(self, vessel1_data: pd.DataFrame, 
                      vessel2_data: pd.DataFrame) -> Optional[Dict]:
        """Calculate closest point of approach between two vessels"""
        min_distance = float('inf')
        cpa_time = None
        cpa_pos1 = None
        cpa_pos2 = None
        
        # Interpolate 12s data to 6s for better CPA calculation
        vessel1_interp = self._interpolate_to_6s(vessel1_data)
        vessel2_interp = self._interpolate_to_6s(vessel2_data)
        
        # Calculate distance at each interpolated time point
        for _, v1_row in vessel1_interp.iterrows():
            v1_time = v1_row['BaseDateTime']
            
            # Find closest time match in vessel2
            time_diffs = abs(vessel2_interp['BaseDateTime'] - v1_time)
            if len(time_diffs) == 0:
                continue
                
            nearest_idx = time_diffs.idxmin()
            v2_row = vessel2_interp.loc[nearest_idx]
            
            # Skip if time difference too large
            if abs((v2_row['BaseDateTime'] - v1_time).total_seconds()) > 60:
                continue
            
            # Calculate distance
            distance = self._haversine_distance(
                v1_row['LAT'], v1_row['LON'],
                v2_row['LAT'], v2_row['LON']
            )
            
            if distance < min_distance:
                min_distance = distance
                cpa_time = v1_time
                cpa_pos1 = (v1_row['LAT'], v1_row['LON'])
                cpa_pos2 = (v2_row['LAT'], v2_row['LON'])
        
        if cpa_time is None:
            return None
        
        # Calculate TCPA (simplified)
        current_time = vessel1_data['BaseDateTime'].min()
        tcpa_min = (cpa_time - current_time).total_seconds() / 60
        
        return {
            'cpa_nm': min_distance,
            'tcpa_min': tcpa_min,
            'cpa_time': cpa_time,
            'pos1': cpa_pos1,
            'pos2': cpa_pos2
        }
    
    def _should_react(self, cpa_info: Dict) -> bool:
        """Determine if vessel should react based on CPA"""
        return (cpa_info['cpa_nm'] < self.CPA_THRESHOLD_NM and 
                cpa_info['tcpa_min'] > 0 and 
                cpa_info['tcpa_min'] < self.TCPA_THRESHOLD_MIN)
    
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
        speed_reduction = original_sog * 0.2  # 20% reduction
        df.loc[vessel_mask, 'SOG'] = original_sog - speed_reduction
        
        # Update vessel info
        vessel_info['delta_heading'] = self.EMERGENCY_TURN_DEG
        vessel_info['speed_drop'] = speed_reduction
        vessel_info['initial_speed'] = original_sog
        vessel_info['action'] = 'emergency_turn'
        vessel_info['cascade_level'] = cascade_level
        
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
    
    def _interpolate_to_6s(self, vessel_data: pd.DataFrame) -> pd.DataFrame:
        """Interpolate 12s AIS data to 6s intervals for better CPA calculation"""
        if len(vessel_data) < 2:
            return vessel_data
        
        # For drift attack with 12s data, skip interpolation to save time
        return vessel_data.sort_values('BaseDateTime').copy()
            # Find surrounding points
            before = vessel_data[vessel_data['BaseDateTime'] <= t]
            after = vessel_data[vessel_data['BaseDateTime'] > t]
            
            if len(before) == 0 or len(after) == 0:
                # Use nearest point
                if len(before) > 0:
                    nearest = before.iloc[-1]
                elif len(after) > 0:
                    nearest = after.iloc[0]
                else:
                    continue
                
                interp_data.append({
                    'BaseDateTime': t,
                    'LAT': nearest['LAT'],
                    'LON': nearest['LON'],
                    'SOG': nearest['SOG'],
                    'COG': nearest['COG'],
                    'MMSI': nearest['MMSI']
                })
            else:
                # Linear interpolation
                p1 = before.iloc[-1]
                p2 = after.iloc[0]
                
                # Time fraction
                total_sec = (p2['BaseDateTime'] - p1['BaseDateTime']).total_seconds()
                elapsed_sec = (t - p1['BaseDateTime']).total_seconds()
                frac = elapsed_sec / total_sec if total_sec > 0 else 0
                
                # Interpolate position
                lat = p1['LAT'] + (p2['LAT'] - p1['LAT']) * frac
                lon = p1['LON'] + (p2['LON'] - p1['LON']) * frac
                sog = p1['SOG'] + (p2['SOG'] - p1['SOG']) * frac
                
                # Interpolate heading (considering wrap-around)
                cog_diff = p2['COG'] - p1['COG']
                if cog_diff > 180:
                    cog_diff -= 360
                elif cog_diff < -180:
                    cog_diff += 360
                cog = (p1['COG'] + cog_diff * frac) % 360
                
                interp_data.append({
                    'BaseDateTime': t,
                    'LAT': lat,
                    'LON': lon,
                    'SOG': sog,
                    'COG': cog,
                    'MMSI': p1['MMSI']
                })
        
        return pd.DataFrame(interp_data)
    
    def get_impact_events(self, ghost_mmsi: int, target_mmsi: int,
                         attack_window: Dict) -> Dict:
        """Format cascade events for output"""
        # Find primary target event
        primary_event = None
        secondary_events = []
        
        for event in self.cascade_events:
            if event['mmsi'] == target_mmsi:
                primary_event = event
            else:
                secondary_events.append(event)
        
        # Format output
        impact_data = {
            "scenario": "s1_cascade",
            "ghost_mmsi": ghost_mmsi,
            "target_mmsi": target_mmsi,
            "attack_window": attack_window,
            "impacted_vessels": len(self.cascade_events),
            "cascade_levels": max([e.get('cascade_level', 1) for e in self.cascade_events]) if self.cascade_events else 0,
            "events": []
        }
        
        # Add primary event first
        if primary_event:
            impact_data["events"].append(self._format_event(primary_event))
        
        # Add secondary events
        for event in secondary_events:
            impact_data["events"].append(self._format_event(event))
        
        return impact_data
    
    def _format_event(self, event: Dict) -> Dict:
        """Format single event for output"""
        return {
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
        }