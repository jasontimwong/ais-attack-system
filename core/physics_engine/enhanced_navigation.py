#!/usr/bin/env python3
"""
Enhanced navigation module for precision improvements.
Implements waypoint interpolation and navigation enhancements.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnhancedWaypoint:
    """Enhanced waypoint with metadata"""
    lat: float
    lon: float
    distance_from_start: float  # Cumulative distance in nm
    segment_index: int  # Which original segment this belongs to
    interpolated: bool  # True if this is an interpolated point


class EnhancedNavigation:
    """Enhanced navigation functions for precision improvement"""
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance between two points in nautical miles.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in nautical miles
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in nautical miles
        r_nm = 3440.065  # Earth's radius in nautical miles
        return c * r_nm
    
    @staticmethod
    def interpolate_geodesic(lat1: float, lon1: float, lat2: float, lon2: float, 
                           fraction: float) -> Tuple[float, float]:
        """
        Interpolate point along great circle between two points.
        
        Args:
            lat1, lon1: Start point
            lat2, lon2: End point
            fraction: Fraction of distance (0.0 to 1.0)
            
        Returns:
            Interpolated (lat, lon)
        """
        # Convert to radians
        lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
        lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
        
        # Calculate angular distance
        d = 2 * np.arcsin(np.sqrt(
            np.sin((lat2_r - lat1_r) / 2)**2 + 
            np.cos(lat1_r) * np.cos(lat2_r) * np.sin((lon2_r - lon1_r) / 2)**2
        ))
        
        if d < 1e-10:  # Points are essentially the same
            return lat1, lon1
        
        # Calculate interpolated point using spherical interpolation
        a = np.sin((1 - fraction) * d) / np.sin(d)
        b = np.sin(fraction * d) / np.sin(d)
        
        x = a * np.cos(lat1_r) * np.cos(lon1_r) + b * np.cos(lat2_r) * np.cos(lon2_r)
        y = a * np.cos(lat1_r) * np.sin(lon1_r) + b * np.cos(lat2_r) * np.sin(lon2_r)
        z = a * np.sin(lat1_r) + b * np.sin(lat2_r)
        
        lat_i = np.arctan2(z, np.sqrt(x**2 + y**2))
        lon_i = np.arctan2(y, x)
        
        return np.degrees(lat_i), np.degrees(lon_i)
    
    @classmethod
    def enhance_searoute_waypoints(cls, route_points: List[Tuple[float, float]], 
                                 target_spacing_nm: float = 5.0) -> List[EnhancedWaypoint]:
        """
        Enhance sparse searoute waypoints with geodesic interpolation.
        
        Args:
            route_points: List of (lat, lon) tuples from searoute
            target_spacing_nm: Target spacing between waypoints in nautical miles
            
        Returns:
            List of enhanced waypoints with interpolated points
        """
        if not route_points or len(route_points) < 2:
            logger.warning("Insufficient route points for enhancement")
            return []
        
        enhanced_waypoints = []
        cumulative_distance = 0.0
        
        # Process each segment
        for seg_idx in range(len(route_points) - 1):
            start_lat, start_lon = route_points[seg_idx]
            end_lat, end_lon = route_points[seg_idx + 1]
            
            # Add start point (not interpolated)
            if seg_idx == 0:
                enhanced_waypoints.append(EnhancedWaypoint(
                    lat=start_lat,
                    lon=start_lon,
                    distance_from_start=0.0,
                    segment_index=seg_idx,
                    interpolated=False
                ))
            
            # Calculate segment distance
            segment_distance = cls.haversine_distance(start_lat, start_lon, end_lat, end_lon)
            
            # Determine number of intermediate points needed
            num_intermediate = max(0, int(segment_distance / target_spacing_nm) - 1)
            
            if num_intermediate > 0:
                # Add interpolated waypoints
                for i in range(1, num_intermediate + 1):
                    fraction = i / (num_intermediate + 1)
                    interp_lat, interp_lon = cls.interpolate_geodesic(
                        start_lat, start_lon, end_lat, end_lon, fraction
                    )
                    
                    # Calculate distance to this point
                    point_distance = segment_distance * fraction
                    
                    enhanced_waypoints.append(EnhancedWaypoint(
                        lat=interp_lat,
                        lon=interp_lon,
                        distance_from_start=cumulative_distance + point_distance,
                        segment_index=seg_idx,
                        interpolated=True
                    ))
                    
                    logger.debug(f"Added interpolated waypoint {i}/{num_intermediate} "
                               f"at ({interp_lat:.6f}, {interp_lon:.6f})")
            
            # Update cumulative distance
            cumulative_distance += segment_distance
            
            # Add end point (not interpolated) - except for last segment
            if seg_idx < len(route_points) - 2:
                enhanced_waypoints.append(EnhancedWaypoint(
                    lat=end_lat,
                    lon=end_lon,
                    distance_from_start=cumulative_distance,
                    segment_index=seg_idx + 1,
                    interpolated=False
                ))
        
        # Add final point
        final_lat, final_lon = route_points[-1]
        enhanced_waypoints.append(EnhancedWaypoint(
            lat=final_lat,
            lon=final_lon,
            distance_from_start=cumulative_distance,
            segment_index=len(route_points) - 1,
            interpolated=False
        ))
        
        # Log enhancement statistics
        original_count = len(route_points)
        enhanced_count = len(enhanced_waypoints)
        interpolated_count = sum(1 for wp in enhanced_waypoints if wp.interpolated)
        
        logger.info(f"Enhanced route: {original_count} → {enhanced_count} waypoints "
                   f"({interpolated_count} interpolated)")
        logger.info(f"Total route distance: {cumulative_distance:.1f} nm")
        logger.info(f"Average waypoint spacing: {cumulative_distance / (enhanced_count - 1):.2f} nm")
        
        return enhanced_waypoints
    
    @classmethod
    def calculate_cross_track_error(cls, vessel_lat: float, vessel_lon: float,
                                  wp1: Tuple[float, float], wp2: Tuple[float, float]) -> float:
        """
        Calculate cross-track error (XTE) from vessel position to track line.
        
        Args:
            vessel_lat, vessel_lon: Current vessel position
            wp1: Previous waypoint (lat, lon)
            wp2: Next waypoint (lat, lon)
            
        Returns:
            Cross-track error in nautical miles (positive = right of track)
        """
        # Convert to radians
        lat1, lon1 = np.radians(wp1)
        lat2, lon2 = np.radians(wp2)
        lat3, lon3 = np.radians(vessel_lat), np.radians(vessel_lon)
        
        # Calculate angular distance from wp1 to vessel
        d13 = 2 * np.arcsin(np.sqrt(
            np.sin((lat3 - lat1) / 2)**2 + 
            np.cos(lat1) * np.cos(lat3) * np.sin((lon3 - lon1) / 2)**2
        ))
        
        # Calculate initial bearing from wp1 to wp2
        y = np.sin(lon2 - lon1) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
        brng12 = np.arctan2(y, x)
        
        # Calculate initial bearing from wp1 to vessel
        y = np.sin(lon3 - lon1) * np.cos(lat3)
        x = np.cos(lat1) * np.sin(lat3) - np.sin(lat1) * np.cos(lat3) * np.cos(lon3 - lon1)
        brng13 = np.arctan2(y, x)
        
        # Calculate cross-track distance
        xte_rad = np.arcsin(np.sin(d13) * np.sin(brng13 - brng12))
        
        # Convert to nautical miles
        r_nm = 3440.065
        xte_nm = xte_rad * r_nm
        
        return xte_nm
    
    @classmethod
    def calculate_along_track_error(cls, vessel_lat: float, vessel_lon: float,
                                  wp1: Tuple[float, float], wp2: Tuple[float, float]) -> float:
        """
        Calculate along-track error (ATE) - distance along track from wp1.
        
        Args:
            vessel_lat, vessel_lon: Current vessel position
            wp1: Previous waypoint (lat, lon)
            wp2: Next waypoint (lat, lon)
            
        Returns:
            Along-track distance from wp1 in nautical miles
        """
        # Calculate cross-track error first
        xte = cls.calculate_cross_track_error(vessel_lat, vessel_lon, wp1, wp2)
        
        # Calculate distance from wp1 to vessel
        d13 = cls.haversine_distance(wp1[0], wp1[1], vessel_lat, vessel_lon)
        
        # Along-track distance using Pythagoras (approximation for small XTE)
        if abs(xte) < d13:
            ate = np.sqrt(d13**2 - xte**2)
        else:
            ate = 0.0
        
        return ate


def test_enhancement():
    """Test waypoint enhancement with sample data"""
    # Sample route (sparse waypoints)
    test_route = [
        (37.7749, -122.4194),  # San Francisco
        (37.3382, -121.8863),  # San Jose  
        (36.7783, -119.4179),  # Fresno
    ]
    
    enhancer = EnhancedNavigation()
    enhanced = enhancer.enhance_searoute_waypoints(test_route, target_spacing_nm=10.0)
    
    print(f"\nOriginal route: {len(test_route)} waypoints")
    print(f"Enhanced route: {len(enhanced)} waypoints")
    
    for i, wp in enumerate(enhanced):
        marker = "[I]" if wp.interpolated else "[O]"
        print(f"{i:3d} {marker} ({wp.lat:8.4f}, {wp.lon:9.4f}) "
              f"dist: {wp.distance_from_start:6.1f} nm")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_enhancement()