"""Utility functions for AIS data processing."""

import math
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees)
        lat2, lon2: Latitude and longitude of point 2 (degrees)
    
    Returns:
        Distance in nautical miles
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in nautical miles
    R = 3440.065  # nautical miles
    
    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing between two points.
    
    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees)
        lat2, lon2: Latitude and longitude of point 2 (degrees)
    
    Returns:
        Bearing in degrees (0-360)
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    
    bearing = math.atan2(x, y)
    
    # Convert to degrees and normalize to 0-360
    return (math.degrees(bearing) + 360) % 360


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within valid ranges.
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
    
    Returns:
        True if valid, False otherwise
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def timestamp_to_iso(timestamp: float) -> str:
    """
    Convert Unix timestamp to ISO 8601 format.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch)
    
    Returns:
        ISO 8601 formatted string
    """
    return datetime.fromtimestamp(timestamp).isoformat() + 'Z'


def iso_to_timestamp(iso_string: str) -> float:
    """
    Convert ISO 8601 string to Unix timestamp.
    
    Args:
        iso_string: ISO 8601 formatted datetime string
    
    Returns:
        Unix timestamp
    """
    # Remove 'Z' suffix if present
    if iso_string.endswith('Z'):
        iso_string = iso_string[:-1]
    
    dt = datetime.fromisoformat(iso_string)
    return dt.timestamp()


def calculate_speed_between_points(
    lat1: float, lon1: float, time1: float,
    lat2: float, lon2: float, time2: float
) -> Optional[float]:
    """
    Calculate speed between two points.
    
    Args:
        lat1, lon1: Coordinates of point 1
        time1: Timestamp of point 1
        lat2, lon2: Coordinates of point 2
        time2: Timestamp of point 2
    
    Returns:
        Speed in knots, or None if time difference is 0
    """
    time_diff = time2 - time1
    if time_diff <= 0:
        return None
    
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    hours = time_diff / 3600
    
    return distance / hours


def normalize_column_names(columns: List[str]) -> Dict[str, str]:
    """
    Create a mapping from various column name formats to standard names.
    
    Args:
        columns: List of column names from CSV
    
    Returns:
        Dictionary mapping original names to standardized names
    """
    # Define mappings for common variations
    mappings = {
        # MMSI variations
        'MMSI': 'mmsi',
        'mmsi': 'mmsi',
        'vessel_id': 'mmsi',
        'ship_id': 'mmsi',
        
        # Latitude variations
        'LAT': 'latitude',
        'lat': 'latitude',
        'Lat': 'latitude',
        'latitude': 'latitude',
        'Latitude': 'latitude',
        
        # Longitude variations
        'LON': 'longitude',
        'lon': 'longitude',
        'Lon': 'longitude',
        'long': 'longitude',
        'Long': 'longitude',
        'longitude': 'longitude',
        'Longitude': 'longitude',
        
        # Time variations
        'timestamp': 'timestamp',
        'Timestamp': 'timestamp',
        'time': 'timestamp',
        'Time': 'timestamp',
        'datetime': 'timestamp',
        'DateTime': 'timestamp',
        'BaseDateTime': 'timestamp',
        'basedatetime': 'timestamp',
        
        # Speed variations
        'SOG': 'speed',
        'sog': 'speed',
        'speed': 'speed',
        'Speed': 'speed',
        'speed_over_ground': 'speed',
        
        # Heading variations
        'COG': 'cog',
        'cog': 'cog',
        'heading': 'heading',
        'Heading': 'heading',
        'course': 'heading',
        'Course': 'heading',
        'course_over_ground': 'heading'
    }
    
    column_mapping = {}
    for col in columns:
        if col in mappings:
            column_mapping[col] = mappings[col]
        else:
            # Keep original if no mapping found
            column_mapping[col] = col.lower()
    
    return column_mapping


def interpolate_missing_points(
    points: List[Tuple[float, float, float]],
    max_time_gap: float = 300  # 5 minutes
) -> List[Tuple[float, float, float]]:
    """
    Interpolate missing points in a trajectory.
    
    Args:
        points: List of (timestamp, lat, lon) tuples
        max_time_gap: Maximum time gap to interpolate (seconds)
    
    Returns:
        List with interpolated points added
    """
    if len(points) < 2:
        return points
    
    result = [points[0]]
    
    for i in range(1, len(points)):
        prev_time, prev_lat, prev_lon = points[i-1]
        curr_time, curr_lat, curr_lon = points[i]
        
        time_gap = curr_time - prev_time
        
        if time_gap > max_time_gap:
            # Interpolate points
            num_points = int(time_gap / max_time_gap)
            for j in range(1, num_points):
                ratio = j / num_points
                interp_time = prev_time + ratio * time_gap
                interp_lat = prev_lat + ratio * (curr_lat - prev_lat)
                interp_lon = prev_lon + ratio * (curr_lon - prev_lon)
                result.append((interp_time, interp_lat, interp_lon))
        
        result.append(points[i])
    
    return result


def calculate_trajectory_stats(
    points: List[Tuple[float, float, float, Optional[float]]]
) -> Dict[str, Any]:
    """
    Calculate statistics for a trajectory.
    
    Args:
        points: List of (timestamp, lat, lon, speed) tuples
    
    Returns:
        Dictionary with trajectory statistics
    """
    if not points:
        return {}
    
    timestamps = [p[0] for p in points]
    lats = [p[1] for p in points]
    lons = [p[2] for p in points]
    speeds = [p[3] for p in points if p[3] is not None]
    
    # Calculate total distance
    total_distance = 0.0
    for i in range(1, len(points)):
        dist = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
        total_distance += dist
    
    # Calculate speed statistics
    if speeds:
        min_speed = min(speeds)
        max_speed = max(speeds)
        avg_speed = sum(speeds) / len(speeds)
    else:
        min_speed = max_speed = avg_speed = 0.0
    
    return {
        'time_start': timestamp_to_iso(min(timestamps)),
        'time_end': timestamp_to_iso(max(timestamps)),
        'duration_hours': (max(timestamps) - min(timestamps)) / 3600,
        'total_distance': round(total_distance, 2),
        'min_speed': round(min_speed, 1),
        'max_speed': round(max_speed, 1),
        'avg_speed': round(avg_speed, 1),
        'point_count': len(points),
        'bbox': {
            'min_lon': min(lons),
            'min_lat': min(lats),
            'max_lon': max(lons),
            'max_lat': max(lats)
        }
    }


def detect_stationary_periods(
    points: List[Tuple[float, float, float]],
    speed_threshold: float = 0.5,  # knots
    min_duration: float = 600  # 10 minutes
) -> List[Tuple[int, int]]:
    """
    Detect periods where vessel is stationary.
    
    Args:
        points: List of (timestamp, lat, lon) tuples
        speed_threshold: Speed below which vessel is considered stationary
        min_duration: Minimum duration to be considered a stationary period
    
    Returns:
        List of (start_index, end_index) tuples for stationary periods
    """
    stationary_periods = []
    start_idx = None
    
    for i in range(1, len(points)):
        speed = calculate_speed_between_points(
            points[i-1][1], points[i-1][2], points[i-1][0],
            points[i][1], points[i][2], points[i][0]
        )
        
        if speed is not None and speed < speed_threshold:
            if start_idx is None:
                start_idx = i - 1
        else:
            if start_idx is not None:
                duration = points[i-1][0] - points[start_idx][0]
                if duration >= min_duration:
                    stationary_periods.append((start_idx, i-1))
                start_idx = None
    
    # Check if still stationary at end
    if start_idx is not None:
        duration = points[-1][0] - points[start_idx][0]
        if duration >= min_duration:
            stationary_periods.append((start_idx, len(points)-1))
    
    return stationary_periods