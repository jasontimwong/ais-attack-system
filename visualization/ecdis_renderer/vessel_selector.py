#!/usr/bin/env python3
"""
VesselSelector - Module for selecting vessels within the attack area
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Set, Tuple
from datetime import datetime


class VesselSelector:
    """Select vessels within the attack area and identify impacted vessels"""
    
    def __init__(self, diff_path: str, impact_events_path: str):
        self.diff_data = self._load_json(diff_path)
        self.impact_events = self._load_json(impact_events_path)
        
        self.ghost_mmsi = self.diff_data['ghost_mmsi']
        self.target_mmsi = self.diff_data['target_mmsi']
        self.attack_window = {
            'start': pd.to_datetime(self.diff_data['attack_window']['start']),
            'end': pd.to_datetime(self.diff_data['attack_window']['end'])
        }
        
        # Get impacted vessel MMSIs
        self.impacted_vessels = set()
        for event in self.impact_events.get('events', []):
            self.impacted_vessels.add(event['mmsi'])
            
    def _load_json(self, path: str) -> Dict:
        """Load JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_ghost_bounds(self, ghost_track: pd.DataFrame, buffer: float = 0.15) -> Dict[str, float]:
        """Calculate bounding box around ghost vessel track"""
        bounds = {
            'lat_min': ghost_track['LAT'].min() - buffer,
            'lat_max': ghost_track['LAT'].max() + buffer,
            'lon_min': ghost_track['LON'].min() - buffer,
            'lon_max': ghost_track['LON'].max() + buffer
        }
        return bounds
    
    def select_area_vessels(self, attack_df: pd.DataFrame, 
                           min_vessels: int = 4) -> List[int]:
        """Select vessels within the attack area"""
        
        # Get ghost vessel track
        ghost_track = attack_df[
            (attack_df['MMSI'] == self.ghost_mmsi) &
            (attack_df['BaseDateTime'] >= self.attack_window['start']) &
            (attack_df['BaseDateTime'] <= self.attack_window['end'])
        ]
        
        if ghost_track.empty:
            print("Warning: No ghost vessel track found in attack window")
            return []
        
        # Get bounds
        bounds = self.get_ghost_bounds(ghost_track)
        
        # Find vessels in area during attack window
        area_vessels = attack_df[
            (attack_df['BaseDateTime'] >= self.attack_window['start']) &
            (attack_df['BaseDateTime'] <= self.attack_window['end']) &
            (attack_df['LAT'] >= bounds['lat_min']) &
            (attack_df['LAT'] <= bounds['lat_max']) &
            (attack_df['LON'] >= bounds['lon_min']) &
            (attack_df['LON'] <= bounds['lon_max']) &
            (attack_df['MMSI'] != self.ghost_mmsi)
        ]['MMSI'].unique()
        
        # Always include target vessel
        vessel_set = set(area_vessels)
        vessel_set.add(self.target_mmsi)
        
        # If we need more vessels, expand the search area
        if len(vessel_set) < min_vessels:
            print(f"Expanding search area (found {len(vessel_set)} vessels, need {min_vessels})")
            
            # Progressively expand bounds
            for expansion in [0.3, 0.5, 0.7, 1.0]:
                expanded_bounds = self.get_ghost_bounds(ghost_track, buffer=expansion)
                
                expanded_vessels = attack_df[
                    (attack_df['BaseDateTime'] >= self.attack_window['start']) &
                    (attack_df['BaseDateTime'] <= self.attack_window['end']) &
                    (attack_df['LAT'] >= expanded_bounds['lat_min']) &
                    (attack_df['LAT'] <= expanded_bounds['lat_max']) &
                    (attack_df['LON'] >= expanded_bounds['lon_min']) &
                    (attack_df['LON'] <= expanded_bounds['lon_max']) &
                    (attack_df['MMSI'] != self.ghost_mmsi)
                ]['MMSI'].unique()
                
                vessel_set.update(expanded_vessels)
                
                if len(vessel_set) >= min_vessels:
                    break
        
        return list(vessel_set)
    
    def categorize_vessels(self, vessel_list: List[int]) -> Dict[str, List[int]]:
        """Categorize vessels by their role/impact"""
        categories = {
            'ghost': [self.ghost_mmsi],
            'target': [self.target_mmsi],
            'impacted': [],
            'nearby': []
        }
        
        for mmsi in vessel_list:
            if mmsi == self.ghost_mmsi:
                continue  # Already in ghost category
            elif mmsi == self.target_mmsi:
                continue  # Already in target category
            elif mmsi in self.impacted_vessels:
                categories['impacted'].append(mmsi)
            else:
                categories['nearby'].append(mmsi)
        
        return categories
    
    def get_impact_details(self, mmsi: int) -> Dict:
        """Get impact details for a specific vessel"""
        for event in self.impact_events.get('events', []):
            if event['mmsi'] == mmsi:
                return event
        return None
    
    def summarize_selection(self, categories: Dict[str, List[int]]) -> None:
        """Print summary of selected vessels"""
        print("\nVessel Selection Summary:")
        print(f"  Ghost vessel: {self.ghost_mmsi}")
        print(f"  Target vessel: {self.target_mmsi}")
        print(f"  Impacted vessels: {len(categories['impacted'])}")
        print(f"  Other nearby vessels: {len(categories['nearby'])}")
        print(f"  Total vessels: {sum(len(v) for v in categories.values())}")