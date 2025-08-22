#!/usr/bin/env python3
"""
Impact simulator for generating impact.csv and impact_events.json based on attack data
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os

class ImpactSimulator:
    def __init__(self, scenario_path: str):
        self.scenario_path = scenario_path
        self.baseline_df = None
        self.attack_df = None
        self.ghost_mmsi = None
        self.target_mmsi = None
        self.attack_window = None
        
    def load_data(self):
        """Load baseline, attack, and diff data"""
        # Load CSVs
        self.baseline_df = pd.read_csv(os.path.join(self.scenario_path, 'baseline_s1H.csv'))
        self.attack_df = pd.read_csv(os.path.join(self.scenario_path, 'attack_s1H.csv'))
        
        # Parse timestamps
        self.baseline_df['BaseDateTime'] = pd.to_datetime(self.baseline_df['BaseDateTime'])
        self.attack_df['BaseDateTime'] = pd.to_datetime(self.attack_df['BaseDateTime'])
        
        # Load diff.json
        with open(os.path.join(self.scenario_path, 'diff_s1H.json'), 'r') as f:
            diff_data = json.load(f)
            
        self.ghost_mmsi = diff_data['ghost_mmsi']
        self.target_mmsi = diff_data['target_mmsi']
        self.attack_window = {
            'start': pd.to_datetime(diff_data['attack_window']['start']),
            'end': pd.to_datetime(diff_data['attack_window']['end'])
        }
        
    def calculate_cpa(self, vessel1_track: pd.DataFrame, vessel2_track: pd.DataFrame) -> Tuple[float, datetime, dict]:
        """Calculate closest point of approach between two vessel tracks"""
        min_distance = float('inf')
        cpa_time = None
        cpa_pos1 = None
        cpa_pos2 = None
        
        # Sort by time
        vessel1_track = vessel1_track.sort_values('BaseDateTime')
        vessel2_track = vessel2_track.sort_values('BaseDateTime')
        
        for _, pos1 in vessel1_track.iterrows():
            # Find closest time match in vessel2
            time_diffs = (vessel2_track['BaseDateTime'] - pos1['BaseDateTime']).abs()
            if len(time_diffs) == 0:
                continue
                
            closest_idx = time_diffs.idxmin()
            pos2 = vessel2_track.loc[closest_idx]
            
            # Only consider if times are within 30 seconds
            if abs((pos1['BaseDateTime'] - pos2['BaseDateTime']).total_seconds()) <= 30:
                # Simple Euclidean distance (for small areas)
                dist = np.sqrt((pos1['LAT'] - pos2['LAT'])**2 + (pos1['LON'] - pos2['LON'])**2)
                if dist < min_distance:
                    min_distance = dist
                    cpa_time = pos1['BaseDateTime']
                    cpa_pos1 = {'lat': pos1['LAT'], 'lon': pos1['LON']}
                    cpa_pos2 = {'lat': pos2['LAT'], 'lon': pos2['LON']}
                    
        # Convert to nautical miles (approximate)
        min_distance_nm = min_distance * 60
        cpa_info = {
            'pos1': cpa_pos1,
            'pos2': cpa_pos2
        }
        return min_distance_nm, cpa_time, cpa_info
        
    def detect_maneuver(self, vessel_track: pd.DataFrame, cpa_time: datetime) -> Dict:
        """Detect if vessel made evasive maneuver"""
        # Get track around CPA time
        window_start = cpa_time - timedelta(minutes=2)
        window_end = cpa_time + timedelta(minutes=2)
        
        track_window = vessel_track[(vessel_track['BaseDateTime'] >= window_start) & 
                                   (vessel_track['BaseDateTime'] <= window_end)].copy()
        
        if len(track_window) < 3:
            return None
            
        # Calculate heading changes
        track_window['heading_change'] = track_window['COG'].diff().abs()
        track_window['speed_change'] = track_window['SOG'].diff()
        
        # Detect significant maneuvers
        max_heading_change = track_window['heading_change'].max()
        max_speed_drop = -track_window['speed_change'].min() if track_window['speed_change'].min() < 0 else 0
        
        # Threshold for significant maneuver
        if max_heading_change > 20 or max_speed_drop > 2:
            maneuver_idx = track_window['heading_change'].idxmax()
            maneuver_time = track_window.loc[maneuver_idx, 'BaseDateTime']
            
            return {
                'maneuver_time': maneuver_time.strftime('%Y-%m-%d %H:%M:%S'),
                'delta_heading': float(max_heading_change),
                'speed_drop': float(max_speed_drop),
                'initial_speed': float(track_window.iloc[0]['SOG']),
                'action': 'turn' if max_heading_change > 20 else 'slow'
            }
        
        return None
        
    def simulate_impact(self):
        """Simulate vessel impacts based on ghost vessel presence"""
        # Get ghost vessel track
        ghost_track = self.attack_df[
            (self.attack_df['MMSI'] == self.ghost_mmsi) &
            (self.attack_df['BaseDateTime'] >= self.attack_window['start']) &
            (self.attack_df['BaseDateTime'] <= self.attack_window['end'])
        ].copy()
        
        print(f"Ghost track points: {len(ghost_track)}")
        
        # Find vessels near ghost vessel
        ghost_bounds = {
            'lat_min': ghost_track['LAT'].min() - 0.2,
            'lat_max': ghost_track['LAT'].max() + 0.2,
            'lon_min': ghost_track['LON'].min() - 0.2,
            'lon_max': ghost_track['LON'].max() + 0.2
        }
        
        # Get vessels in area during attack window
        area_vessels = self.attack_df[
            (self.attack_df['BaseDateTime'] >= self.attack_window['start']) &
            (self.attack_df['BaseDateTime'] <= self.attack_window['end']) &
            (self.attack_df['LAT'] >= ghost_bounds['lat_min']) &
            (self.attack_df['LAT'] <= ghost_bounds['lat_max']) &
            (self.attack_df['LON'] >= ghost_bounds['lon_min']) &
            (self.attack_df['LON'] <= ghost_bounds['lon_max']) &
            (self.attack_df['MMSI'] != self.ghost_mmsi)
        ]['MMSI'].unique()
        
        print(f"Vessels in area: {len(area_vessels)}")
        
        impact_events = []
        impacted_vessels = []
        
        # Always include target vessel
        if self.target_mmsi not in area_vessels:
            area_vessels = np.append(area_vessels, self.target_mmsi)
        
        for mmsi in area_vessels:
            vessel_track = self.attack_df[
                (self.attack_df['MMSI'] == mmsi) &
                (self.attack_df['BaseDateTime'] >= self.attack_window['start']) &
                (self.attack_df['BaseDateTime'] <= self.attack_window['end'])
            ].copy()
            
            if len(vessel_track) < 2:
                continue
            
            # Calculate CPA with ghost vessel
            cpa_dist, cpa_time, cpa_info = self.calculate_cpa(ghost_track, vessel_track)
            
            print(f"Vessel {mmsi}: CPA = {cpa_dist:.3f} nm")
            
            # Simulate impact based on CPA distance
            # For target vessel, always simulate impact
            # For others, threshold is 1.0 nm
            threshold = 2.0 if mmsi == self.target_mmsi else 1.0
            
            if cpa_dist < threshold and cpa_time is not None:
                # Simulate realistic maneuver
                if mmsi == self.target_mmsi:
                    # Target vessel makes significant evasive maneuver
                    maneuver = {
                        'maneuver_time': cpa_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'delta_heading': 35.0,  # Significant turn
                        'speed_drop': 3.5,      # Emergency slowdown
                        'initial_speed': 12.0,
                        'action': 'emergency_turn'
                    }
                else:
                    # Other vessels make proportional maneuvers
                    if cpa_dist < 0.3:
                        delta_heading = np.random.uniform(25, 40)
                        speed_drop = np.random.uniform(2, 4)
                        action = 'emergency_turn'
                    elif cpa_dist < 0.6:
                        delta_heading = np.random.uniform(15, 25)
                        speed_drop = np.random.uniform(1, 2)
                        action = 'turn'
                    else:
                        delta_heading = np.random.uniform(5, 15)
                        speed_drop = np.random.uniform(0.5, 1.5)
                        action = 'slight_turn'
                    
                    maneuver = {
                        'maneuver_time': (cpa_time - timedelta(seconds=30)).strftime('%Y-%m-%d %H:%M:%S'),
                        'delta_heading': round(delta_heading, 1),
                        'speed_drop': round(speed_drop, 1),
                        'initial_speed': round(vessel_track['SOG'].mean(), 1),
                        'action': action
                    }
                
                impact_event = {
                    'mmsi': int(mmsi),
                    'vessel_name': vessel_track.iloc[0]['VesselName'] if pd.notna(vessel_track.iloc[0]['VesselName']) else f'Vessel_{mmsi}',
                    'cpa_nm': round(cpa_dist, 3),
                    'cpa_time': cpa_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'cpa_position': cpa_info,
                    **maneuver
                }
                impact_events.append(impact_event)
                impacted_vessels.append(mmsi)
        
        # Generate impact.csv - copy attack data and mark impacted vessels
        impact_df = self.attack_df.copy()
        impact_df['impact_flag'] = impact_df['MMSI'].isin(impacted_vessels).astype(int)
        impact_df['action_flag'] = 0
        
        # Mark action points
        for event in impact_events:
            mmsi = event['mmsi']
            maneuver_time = pd.to_datetime(event['maneuver_time'])
            
            # Find closest timestamp
            vessel_df = impact_df[impact_df['MMSI'] == mmsi]
            time_diffs = (vessel_df['BaseDateTime'] - maneuver_time).abs()
            closest_idx = time_diffs.idxmin()
            
            impact_df.loc[closest_idx, 'action_flag'] = 1
            
        return impact_df, impact_events
        
    def save_results(self, impact_df: pd.DataFrame, impact_events: List[Dict]):
        """Save impact.csv and impact_events.json"""
        # Save impact.csv
        impact_path = os.path.join(self.scenario_path, 'impact.csv')
        impact_df.to_csv(impact_path, index=False)
        print(f"Saved impact.csv with {len(impact_df)} records")
        
        # Save impact_events.json
        events_path = os.path.join(self.scenario_path, 'impact_events.json')
        with open(events_path, 'w') as f:
            json.dump({
                'scenario': 's1H_final',
                'ghost_mmsi': self.ghost_mmsi,
                'target_mmsi': self.target_mmsi,
                'attack_window': {
                    'start': self.attack_window['start'].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': self.attack_window['end'].strftime('%Y-%m-%d %H:%M:%S')
                },
                'impacted_vessels': len(impact_events),
                'events': impact_events
            }, f, indent=2)
        print(f"Saved impact_events.json with {len(impact_events)} impact events")
        
    def run(self):
        """Run the complete impact simulation"""
        print("Starting impact simulation...")
        self.load_data()
        impact_df, impact_events = self.simulate_impact()
        self.save_results(impact_df, impact_events)
        print("Impact simulation completed!")
        
        return impact_df, impact_events


if __name__ == '__main__':
    # Test the simulator
    simulator = ImpactSimulator('/Users/jasonwong/Documents/newAISGEN/ais_attack_generator/out/s1H_final')
    simulator.run()