#!/usr/bin/env python3
"""
Target Selector - MCDA + Fuzzy Logic Implementation

Multi-Criteria Decision Analysis with fuzzy logic for intelligent target selection
in AIS attack scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VesselCandidate:
    """Vessel candidate for target selection"""
    mmsi: str
    lat: float
    lon: float
    speed: float
    course: float
    vessel_type: str
    length: Optional[float] = None
    beam: Optional[float] = None

class TargetSelector:
    """
    Multi-criteria target selection with vulnerability scoring
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize target selector with MCDA weights
        
        Args:
            config: Configuration dictionary with weights and fuzzy parameters
        """
        self.config = config or self._default_config()
        self.weights = self.config['weights']
        
    def _default_config(self) -> Dict:
        """Default configuration for target selection"""
        return {
            'weights': {
                'isolation_factor': 0.30,
                'predictability_score': 0.25,
                'target_value': 0.25,
                'cascade_potential': 0.20
            },
            'fuzzy_parameters': {
                'isolation_thresholds': [2.0, 5.0, 10.0],  # nautical miles
                'predictability_thresholds': [0.3, 0.6, 0.9],
                'value_thresholds': [0.2, 0.5, 0.8],
                'cascade_thresholds': [0.1, 0.4, 0.7]
            },
            'constraints': {
                'min_distance': 0.5,  # nautical miles
                'max_distance': 20.0,
                'min_speed': 2.0,  # knots
                'max_speed': 30.0
            }
        }
    
    def select_optimal_target(self, vessels: List[VesselCandidate], 
                            constraints: Dict = None) -> Optional[VesselCandidate]:
        """
        Select the most vulnerable target vessel
        
        Args:
            vessels: List of available vessel candidates
            constraints: Selection constraints
            
        Returns:
            Selected target with highest vulnerability score
        """
        if not vessels:
            return None
        
        # Filter vessels by constraints
        filtered_vessels = self._apply_constraints(vessels, constraints or {})
        
        if not filtered_vessels:
            return None
        
        # Calculate vulnerability scores for all candidates
        scored_vessels = []
        for vessel in filtered_vessels:
            score = self.calculate_vulnerability_score(vessel, vessels)
            scored_vessels.append((vessel, score))
        
        # Sort by vulnerability score (descending)
        scored_vessels.sort(key=lambda x: x[1], reverse=True)
        
        return scored_vessels[0][0] if scored_vessels else None
    
    def calculate_vulnerability_score(self, vessel: VesselCandidate, 
                                    all_vessels: List[VesselCandidate]) -> float:
        """
        Calculate vulnerability score for a vessel
        
        Args:
            vessel: Target vessel candidate
            all_vessels: All vessels in the area
            
        Returns:
            Vulnerability score (0.0 - 1.0)
        """
        # Calculate individual criteria scores
        isolation = self._calculate_isolation_factor(vessel, all_vessels)
        predictability = self._calculate_predictability_score(vessel)
        value = self._calculate_target_value(vessel)
        cascade = self._calculate_cascade_potential(vessel, all_vessels)
        
        # Apply fuzzy logic
        isolation_fuzzy = self._apply_fuzzy_logic(isolation, 'isolation')
        predictability_fuzzy = self._apply_fuzzy_logic(predictability, 'predictability')
        value_fuzzy = self._apply_fuzzy_logic(value, 'value')
        cascade_fuzzy = self._apply_fuzzy_logic(cascade, 'cascade')
        
        # Calculate weighted vulnerability score
        vulnerability_score = (
            self.weights['isolation_factor'] * isolation_fuzzy +
            self.weights['predictability_score'] * predictability_fuzzy +
            self.weights['target_value'] * value_fuzzy +
            self.weights['cascade_potential'] * cascade_fuzzy
        )
        
        return min(max(vulnerability_score, 0.0), 1.0)
    
    def _apply_constraints(self, vessels: List[VesselCandidate], 
                          constraints: Dict) -> List[VesselCandidate]:
        """Apply selection constraints to filter vessels"""
        filtered = []
        
        for vessel in vessels:
            # Speed constraints
            if (vessel.speed < constraints.get('min_speed', self.config['constraints']['min_speed']) or
                vessel.speed > constraints.get('max_speed', self.config['constraints']['max_speed'])):
                continue
            
            # Distance constraints (if reference position provided)
            if 'reference_lat' in constraints and 'reference_lon' in constraints:
                distance = self._haversine_distance(
                    vessel.lat, vessel.lon,
                    constraints['reference_lat'], constraints['reference_lon']
                )
                
                if (distance < constraints.get('min_distance', self.config['constraints']['min_distance']) or
                    distance > constraints.get('max_distance', self.config['constraints']['max_distance'])):
                    continue
            
            # Vessel type constraints
            if 'allowed_types' in constraints:
                if vessel.vessel_type not in constraints['allowed_types']:
                    continue
            
            filtered.append(vessel)
        
        return filtered
    
    def _calculate_isolation_factor(self, vessel: VesselCandidate, 
                                  all_vessels: List[VesselCandidate]) -> float:
        """Calculate spatial-temporal isolation factor"""
        distances = []
        
        for other_vessel in all_vessels:
            if other_vessel.mmsi != vessel.mmsi:
                distance = self._haversine_distance(
                    vessel.lat, vessel.lon,
                    other_vessel.lat, other_vessel.lon
                )
                distances.append(distance)
        
        if not distances:
            return 1.0  # Maximum isolation if no other vessels
        
        # Isolation is inversely related to proximity to other vessels
        min_distance = min(distances)
        avg_distance = np.mean(distances)
        
        # Normalize to 0-1 scale
        isolation_score = min(min_distance / 10.0, 1.0)  # 10nm = max isolation
        
        return isolation_score
    
    def _calculate_predictability_score(self, vessel: VesselCandidate) -> float:
        """Calculate vessel behavior predictability"""
        # In real implementation, this would analyze historical trajectory data
        # For now, use vessel type and speed as proxies
        
        predictability_factors = {
            'cargo': 0.8,
            'tanker': 0.9,
            'container': 0.7,
            'passenger': 0.6,
            'fishing': 0.3,
            'pleasure': 0.2
        }
        
        base_predictability = predictability_factors.get(vessel.vessel_type, 0.5)
        
        # Steady speed indicates predictable behavior
        speed_factor = 1.0 if 5 <= vessel.speed <= 20 else 0.7
        
        return base_predictability * speed_factor
    
    def _calculate_target_value(self, vessel: VesselCandidate) -> float:
        """Calculate strategic value of target"""
        # Value based on vessel type and size
        value_factors = {
            'cargo': 0.7,
            'tanker': 0.9,
            'container': 0.8,
            'passenger': 0.6,
            'fishing': 0.4,
            'pleasure': 0.2
        }
        
        base_value = value_factors.get(vessel.vessel_type, 0.5)
        
        # Size factor (if available)
        size_factor = 1.0
        if vessel.length:
            if vessel.length > 200:  # Large vessel
                size_factor = 1.2
            elif vessel.length < 50:  # Small vessel
                size_factor = 0.8
        
        return min(base_value * size_factor, 1.0)
    
    def _calculate_cascade_potential(self, vessel: VesselCandidate,
                                   all_vessels: List[VesselCandidate]) -> float:
        """Calculate potential for cascading effects"""
        # Vessels in high-traffic areas have higher cascade potential
        nearby_vessels = 0
        
        for other_vessel in all_vessels:
            if other_vessel.mmsi != vessel.mmsi:
                distance = self._haversine_distance(
                    vessel.lat, vessel.lon,
                    other_vessel.lat, other_vessel.lon
                )
                
                if distance <= 5.0:  # Within 5 nautical miles
                    nearby_vessels += 1
        
        # Normalize cascade potential
        cascade_score = min(nearby_vessels / 10.0, 1.0)  # Max 10 nearby vessels
        
        return cascade_score
    
    def _apply_fuzzy_logic(self, value: float, criterion: str) -> float:
        """Apply fuzzy logic membership functions"""
        thresholds = self.config['fuzzy_parameters'][f'{criterion}_thresholds']
        
        # Trapezoidal membership function
        if value <= thresholds[0]:
            return 0.0
        elif value <= thresholds[1]:
            return (value - thresholds[0]) / (thresholds[1] - thresholds[0])
        elif value <= thresholds[2]:
            return 1.0
        else:
            return max(0.0, 1.0 - (value - thresholds[2]) / (1.0 - thresholds[2]))
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in nautical miles"""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in nautical miles
        r = 3440.065
        
        return c * r
