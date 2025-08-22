#!/usr/bin/env python3
"""
PrePostSplitter - Module for splitting tracks into pre/post attack segments
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class PrePostSplitter:
    """Split vessel tracks into baseline, attack, and post-maneuver segments"""
    
    def __init__(self, attack_window: Dict[str, datetime]):
        self.attack_start = attack_window['start']
        self.attack_end = attack_window['end']
        
    def split_tracks(self, baseline_df: pd.DataFrame, attack_df: pd.DataFrame, 
                    impact_df: pd.DataFrame, vessel_categories: Dict[str, List[int]]) -> Dict:
        """Split tracks into pre/during/post segments for visualization"""
        
        results = {
            'baseline_tracks': {},      # Original tracks without ghost
            'attack_tracks': {},        # Tracks during attack (with ghost)
            'maneuver_tracks': {}       # Post-maneuver tracks
        }
        
        # Get all vessels to process
        all_vessels = []
        for category, vessels in vessel_categories.items():
            all_vessels.extend(vessels)
        
        # Process each vessel
        for mmsi in all_vessels:
            # Skip ghost vessel in baseline
            if mmsi == vessel_categories['ghost'][0]:
                # Ghost only exists in attack/impact data
                ghost_attack = attack_df[attack_df['MMSI'] == mmsi].copy()
                ghost_impact = impact_df[impact_df['MMSI'] == mmsi].copy()
                
                results['attack_tracks'][mmsi] = ghost_attack
                results['maneuver_tracks'][mmsi] = ghost_impact
                continue
            
            # For real vessels, get tracks from all three datasets
            baseline_track = baseline_df[baseline_df['MMSI'] == mmsi].copy()
            attack_track = attack_df[attack_df['MMSI'] == mmsi].copy()
            impact_track = impact_df[impact_df['MMSI'] == mmsi].copy()
            
            # Store full baseline track (no ghost influence)
            results['baseline_tracks'][mmsi] = baseline_track
            
            # Attack track during window
            results['attack_tracks'][mmsi] = attack_track[
                (attack_track['BaseDateTime'] >= self.attack_start) &
                (attack_track['BaseDateTime'] <= self.attack_end)
            ].copy()
            
            # For impacted vessels, identify maneuver point
            if mmsi in vessel_categories['impacted']:
                # Get maneuver segments from impact data
                maneuver_track = self._extract_maneuver_track(impact_track, mmsi)
                results['maneuver_tracks'][mmsi] = maneuver_track
            else:
                # Non-impacted vessels continue normally
                results['maneuver_tracks'][mmsi] = impact_track[
                    (impact_track['BaseDateTime'] >= self.attack_start) &
                    (impact_track['BaseDateTime'] <= self.attack_end)
                ].copy()
        
        return results
    
    def _extract_maneuver_track(self, impact_track: pd.DataFrame, mmsi: int) -> pd.DataFrame:
        """Extract the portion of track showing maneuver"""
        # Look for action_flag if available
        if 'action_flag' in impact_track.columns:
            action_points = impact_track[impact_track['action_flag'] == 1]
            if not action_points.empty:
                # Get track from slightly before to after maneuver
                maneuver_time = action_points.iloc[0]['BaseDateTime']
                start_time = maneuver_time - timedelta(minutes=1)
                end_time = maneuver_time + timedelta(minutes=3)
                
                return impact_track[
                    (impact_track['BaseDateTime'] >= start_time) &
                    (impact_track['BaseDateTime'] <= end_time)
                ].copy()
        
        # Fallback: return attack window portion
        return impact_track[
            (impact_track['BaseDateTime'] >= self.attack_start) &
            (impact_track['BaseDateTime'] <= self.attack_end)
        ].copy()
    
    def get_track_statistics(self, tracks: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate statistics for tracks"""
        stats = {}
        
        for mmsi, track in tracks.items():
            if track.empty:
                continue
                
            stats[mmsi] = {
                'points': len(track),
                'duration': (track['BaseDateTime'].max() - track['BaseDateTime'].min()).total_seconds(),
                'avg_speed': track['SOG'].mean() if 'SOG' in track.columns else 0,
                'max_speed': track['SOG'].max() if 'SOG' in track.columns else 0,
                'heading_variance': track['COG'].std() if 'COG' in track.columns else 0
            }
        
        return stats
    
    def identify_maneuver_points(self, maneuver_tracks: Dict[str, pd.DataFrame], 
                                impact_events: List[Dict]) -> Dict[int, Dict]:
        """Identify exact maneuver points for each impacted vessel"""
        maneuver_points = {}
        
        for event in impact_events:
            mmsi = event['mmsi']
            if mmsi not in maneuver_tracks or maneuver_tracks[mmsi].empty:
                continue
                
            track = maneuver_tracks[mmsi]
            
            # Find point closest to maneuver time
            maneuver_time = pd.to_datetime(event['maneuver_time'])
            time_diffs = (track['BaseDateTime'] - maneuver_time).abs()
            
            if not time_diffs.empty:
                closest_idx = time_diffs.idxmin()
                maneuver_point = track.loc[closest_idx]
                
                maneuver_points[mmsi] = {
                    'lat': maneuver_point['LAT'],
                    'lon': maneuver_point['LON'],
                    'time': maneuver_point['BaseDateTime'],
                    'heading': maneuver_point['COG'] if 'COG' in track.columns else 0,
                    'speed': maneuver_point['SOG'] if 'SOG' in track.columns else 0,
                    'delta_heading': event.get('delta_heading', 0),
                    'speed_drop': event.get('speed_drop', 0)
                }
        
        return maneuver_points