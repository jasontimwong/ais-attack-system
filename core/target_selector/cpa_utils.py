"""
Unified CPA (Closest Point of Approach) calculation utilities.

This module provides consistent CPA/TCPA calculations used throughout the system
to ensure all components use the same algorithm.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calc_cpa(pA: np.ndarray, vA: np.ndarray, pB: np.ndarray, vB: np.ndarray) -> Tuple[float, float]:
    """
    Calculate CPA and TCPA between two vessels using ECEF coordinates.
    
    Args:
        pA: Position of vessel A in ECEF meters [x, y, z]
        vA: Velocity of vessel A in ECEF m/s [vx, vy, vz]
        pB: Position of vessel B in ECEF meters [x, y, z]
        vB: Velocity of vessel B in ECEF m/s [vx, vy, vz]
        
    Returns:
        Tuple of (cpa_nm, tcpa_min):
        - cpa_nm: Closest point of approach in nautical miles
        - tcpa_min: Time to CPA in minutes (negative if vessels moving apart)
    """
    # Relative position and velocity
    rel_p = pB - pA
    rel_v = vB - vA
    
    # Check if vessels are stationary relative to each other
    rel_v_norm_sq = np.dot(rel_v, rel_v)
    if rel_v_norm_sq < 1e-10:
        # CPA is current distance
        cpa_m = np.linalg.norm(rel_p)
        cpa_nm = cpa_m / 1852.0
        return cpa_nm, 0.0
    
    # Time to CPA in seconds
    tcpa_s = -np.dot(rel_p, rel_v) / rel_v_norm_sq
    
    # If TCPA is negative, vessels are moving apart
    if tcpa_s < 0:
        # Current distance is the CPA
        cpa_m = np.linalg.norm(rel_p)
        cpa_nm = cpa_m / 1852.0
        return cpa_nm, tcpa_s / 60.0
    
    # Calculate CPA distance
    cpa_pos = rel_p + rel_v * tcpa_s
    cpa_m = np.linalg.norm(cpa_pos)
    cpa_nm = cpa_m / 1852.0
    tcpa_min = tcpa_s / 60.0
    
    return cpa_nm, tcpa_min


def lat_lon_to_ecef(lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
    """
    Convert latitude/longitude to ECEF coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters (default: 0 for sea level)
        
    Returns:
        ECEF coordinates [x, y, z] in meters
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis in meters
    e2 = 0.00669437999014  # first eccentricity squared
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    # ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    
    return np.array([x, y, z])


def calc_velocity_ecef(lat: float, lon: float, sog_knots: float, cog_deg: float) -> np.ndarray:
    """
    Calculate velocity vector in ECEF coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        sog_knots: Speed over ground in knots
        cog_deg: Course over ground in degrees (0=North, 90=East)
        
    Returns:
        Velocity vector [vx, vy, vz] in m/s
    """
    # Convert speed to m/s
    speed_ms = sog_knots * 0.514444
    
    # Convert COG to radians (navigation to math convention)
    cog_rad = np.radians(90 - cog_deg)
    
    # Velocity in local tangent plane (ENU - East North Up)
    v_east = speed_ms * np.cos(cog_rad)
    v_north = speed_ms * np.sin(cog_rad)
    v_up = 0.0
    
    # Convert to ECEF using rotation matrix
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Rotation matrix from ENU to ECEF
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    vx = -sin_lon * v_east - sin_lat * cos_lon * v_north + cos_lat * cos_lon * v_up
    vy = cos_lon * v_east - sin_lat * sin_lon * v_north + cos_lat * sin_lon * v_up
    vz = 0 * v_east + cos_lat * v_north + sin_lat * v_up
    
    return np.array([vx, vy, vz])


def calc_min_cpa_track(trackA: pd.DataFrame, trackB: pd.DataFrame, 
                       dt_step: int = 10, time_window_min: float = 30.0) -> Dict[str, float]:
    """
    Calculate minimum CPA between two vessel tracks over a time window.
    
    Args:
        trackA: DataFrame with columns [BaseDateTime, LAT, LON, SOG, COG]
        trackB: DataFrame with columns [BaseDateTime, LAT, LON, SOG, COG]
        dt_step: Time step in seconds for sampling tracks
        time_window_min: Maximum time window in minutes to search for CPA
        
    Returns:
        Dictionary with:
        - min_cpa_nm: Minimum CPA in nautical miles
        - tcpa_min: Time to minimum CPA in minutes
        - cpa_time: DateTime when CPA occurs
        - vesselA_mmsi: MMSI of vessel A (if available)
        - vesselB_mmsi: MMSI of vessel B (if available)
    """
    # Ensure tracks are sorted by time
    trackA = trackA.sort_values('BaseDateTime').copy()
    trackB = trackB.sort_values('BaseDateTime').copy()
    
    # Find overlapping time range
    start_time = max(trackA['BaseDateTime'].min(), trackB['BaseDateTime'].min())
    end_time = min(trackA['BaseDateTime'].max(), trackB['BaseDateTime'].max())
    
    if start_time >= end_time:
        logger.warning("No overlapping time range between tracks")
        return {
            'min_cpa_nm': float('inf'),
            'tcpa_min': 0.0,
            'cpa_time': None,
            'vesselA_mmsi': trackA['MMSI'].iloc[0] if 'MMSI' in trackA else None,
            'vesselB_mmsi': trackB['MMSI'].iloc[0] if 'MMSI' in trackB else None
        }
    
    min_cpa = float('inf')
    best_tcpa = 0.0
    best_time = None
    
    # Sample time points
    current_time = start_time
    time_step = timedelta(seconds=dt_step)
    time_window = timedelta(minutes=time_window_min)
    
    while current_time <= end_time:
        # Get vessel positions at current time
        # Find closest data points
        idxA = (trackA['BaseDateTime'] - current_time).abs().idxmin()
        idxB = (trackB['BaseDateTime'] - current_time).abs().idxmin()
        
        # Check if data points are close enough in time
        if abs((trackA.loc[idxA, 'BaseDateTime'] - current_time).total_seconds()) > 60:
            current_time += time_step
            continue
        if abs((trackB.loc[idxB, 'BaseDateTime'] - current_time).total_seconds()) > 60:
            current_time += time_step
            continue
        
        # Get vessel states
        latA = trackA.loc[idxA, 'LAT']
        lonA = trackA.loc[idxA, 'LON']
        sogA = trackA.loc[idxA, 'SOG']
        cogA = trackA.loc[idxA, 'COG']
        
        latB = trackB.loc[idxB, 'LAT']
        lonB = trackB.loc[idxB, 'LON']
        sogB = trackB.loc[idxB, 'SOG']
        cogB = trackB.loc[idxB, 'COG']
        
        # Convert to ECEF
        pA = lat_lon_to_ecef(latA, lonA)
        vA = calc_velocity_ecef(latA, lonA, sogA, cogA)
        pB = lat_lon_to_ecef(latB, lonB)
        vB = calc_velocity_ecef(latB, lonB, sogB, cogB)
        
        # Calculate CPA
        cpa_nm, tcpa_min = calc_cpa(pA, vA, pB, vB)
        
        # Only consider future CPAs within time window
        if 0 <= tcpa_min <= time_window_min and cpa_nm < min_cpa:
            min_cpa = cpa_nm
            best_tcpa = tcpa_min
            best_time = current_time + timedelta(minutes=tcpa_min)
        
        current_time += time_step
    
    return {
        'min_cpa_nm': min_cpa,
        'tcpa_min': best_tcpa,
        'cpa_time': best_time,
        'vesselA_mmsi': trackA['MMSI'].iloc[0] if 'MMSI' in trackA else None,
        'vesselB_mmsi': trackB['MMSI'].iloc[0] if 'MMSI' in trackB else None
    }


def calc_simple_cpa(lat1: float, lon1: float, sog1: float, cog1: float,
                    lat2: float, lon2: float, sog2: float, cog2: float,
                    max_time_min: float = 30.0) -> Tuple[float, float]:
    """
    Simple CPA calculation for two vessels at a single time point.
    
    Args:
        lat1, lon1: Position of vessel 1 (degrees)
        sog1: Speed of vessel 1 (knots)
        cog1: Course of vessel 1 (degrees)
        lat2, lon2: Position of vessel 2 (degrees)
        sog2: Speed of vessel 2 (knots)
        cog2: Course of vessel 2 (degrees)
        max_time_min: Maximum time to consider for CPA (minutes)
        
    Returns:
        Tuple of (cpa_nm, tcpa_min)
    """
    # Convert to ECEF
    p1 = lat_lon_to_ecef(lat1, lon1)
    v1 = calc_velocity_ecef(lat1, lon1, sog1, cog1)
    p2 = lat_lon_to_ecef(lat2, lon2)
    v2 = calc_velocity_ecef(lat2, lon2, sog2, cog2)
    
    # Calculate CPA
    cpa_nm, tcpa_min = calc_cpa(p1, v1, p2, v2)
    
    # Clip TCPA to max time
    if tcpa_min > max_time_min:
        # Calculate distance at max time
        rel_p = p2 - p1
        rel_v = v2 - v1
        future_pos = rel_p + rel_v * (max_time_min * 60)
        cpa_nm = np.linalg.norm(future_pos) / 1852.0
        tcpa_min = max_time_min
    
    return cpa_nm, tcpa_min