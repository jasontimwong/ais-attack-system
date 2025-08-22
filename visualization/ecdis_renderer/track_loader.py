#!/usr/bin/env python3
"""
TrackLoader - Module for loading and preprocessing AIS tracks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class TrackLoader:
    """Load and preprocess AIS track data from CSV files"""
    
    def __init__(self):
        self.baseline_df = None
        self.attack_df = None
        self.impact_df = None
        
    def load_tracks(self, baseline_path: str, attack_path: str, impact_path: str) -> Dict[str, pd.DataFrame]:
        """Load all three track files and handle column name variations"""
        print("Loading track data...")
        
        # Load CSVs
        self.baseline_df = self._load_and_normalize(baseline_path, "baseline")
        self.attack_df = self._load_and_normalize(attack_path, "attack")
        self.impact_df = self._load_and_normalize(impact_path, "impact")
        
        return {
            'baseline': self.baseline_df,
            'attack': self.attack_df,
            'impact': self.impact_df
        }
    
    def _load_and_normalize(self, path: str, label: str) -> pd.DataFrame:
        """Load CSV and normalize column names"""
        df = pd.read_csv(path)
        
        # Normalize column names to uppercase
        column_mapping = {}
        for col in df.columns:
            # Common variations
            if col.lower() in ['mmsi', 'imo', 'lat', 'lon', 'sog', 'cog', 'heading']:
                column_mapping[col] = col.upper()
            elif col.lower() in ['basedatetime', 'timestamp', 'time']:
                column_mapping[col] = 'BaseDateTime'
            elif col.lower() in ['vesselname', 'name', 'shipname']:
                column_mapping[col] = 'VesselName'
                
        df = df.rename(columns=column_mapping)
        
        # Parse timestamps
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        
        print(f"  Loaded {label}: {len(df)} records, {df['MMSI'].nunique()} vessels")
        
        return df
    
    def filter_by_window(self, start_time: datetime, end_time: datetime, 
                        buffer_minutes: int = 5) -> Dict[str, pd.DataFrame]:
        """Filter tracks by time window with optional buffer"""
        window_start = start_time - timedelta(minutes=buffer_minutes)
        window_end = end_time + timedelta(minutes=buffer_minutes)
        
        filtered = {}
        for key, df in [('baseline', self.baseline_df), 
                       ('attack', self.attack_df), 
                       ('impact', self.impact_df)]:
            if df is not None:
                filtered[key] = df[
                    (df['BaseDateTime'] >= window_start) & 
                    (df['BaseDateTime'] <= window_end)
                ].copy()
                
        return filtered
    
    def resample_tracks(self, df: pd.DataFrame, interval_seconds: int = 10) -> pd.DataFrame:
        """Resample tracks to regular intervals"""
        resampled_vessels = []
        
        for mmsi in df['MMSI'].unique():
            vessel_df = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')
            
            if len(vessel_df) < 2:
                resampled_vessels.append(vessel_df)
                continue
                
            # Create regular time grid
            time_range = pd.date_range(
                start=vessel_df['BaseDateTime'].min(),
                end=vessel_df['BaseDateTime'].max(),
                freq=f'{interval_seconds}S'
            )
            
            # Interpolate positions
            vessel_df = vessel_df.set_index('BaseDateTime')
            
            # Numeric columns to interpolate
            numeric_cols = ['LAT', 'LON', 'SOG', 'COG', 'Heading']
            numeric_cols = [col for col in numeric_cols if col in vessel_df.columns]
            
            # Create new dataframe with regular intervals
            resampled = pd.DataFrame(index=time_range)
            
            # Interpolate numeric columns
            for col in numeric_cols:
                if col in vessel_df.columns:
                    resampled[col] = vessel_df[col].reindex(time_range).interpolate(method='linear')
            
            # Forward fill other columns
            for col in vessel_df.columns:
                if col not in numeric_cols:
                    resampled[col] = vessel_df[col].reindex(time_range).fillna(method='ffill')
            
            resampled['MMSI'] = mmsi
            resampled = resampled.reset_index().rename(columns={'index': 'BaseDateTime'})
            
            resampled_vessels.append(resampled)
        
        return pd.concat(resampled_vessels, ignore_index=True)
    
    def get_vessel_list(self, mmsi_filter: Optional[List[int]] = None) -> List[int]:
        """Get list of vessels, optionally filtered"""
        all_vessels = set()
        
        for df in [self.baseline_df, self.attack_df, self.impact_df]:
            if df is not None:
                all_vessels.update(df['MMSI'].unique())
        
        if mmsi_filter:
            return [mmsi for mmsi in all_vessels if mmsi in mmsi_filter]
        
        return list(all_vessels)