"""
Flash Cross Attack Implementation

This module implements the S1 Flash Cross attack pattern, which is the
most sophisticated attack in our system. It demonstrates the full
capabilities of our multi-stage progressive attack orchestration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from ...core.attack_orchestrator import AttackOrchestrator, AttackStage
from .ghost_vessel import GhostVessel
from .stage_executor import StageExecutor

logger = logging.getLogger(__name__)


@dataclass
class FlashCrossConfig:
    """Configuration for Flash Cross attack"""
    parallel_distance: float = 2.0  # nautical miles
    approach_cpa: float = 0.3  # nautical miles
    cross_angle: float = 90.0  # degrees
    ghost_speed_factor: float = 1.5  # multiplier of target speed
    stage_durations: Dict[str, float] = None
    
    def __post_init__(self):
        if self.stage_durations is None:
            self.stage_durations = {
                'parallel_following': 120.0,  # 2 minutes
                'approach_initiation': 30.0,  # 30 seconds
                'flash_cross_maneuver': 45.0,  # 45 seconds
                'silent_disappearance': 30.0   # 30 seconds
            }


class FlashCrossAttack:
    """
    Flash Cross Attack (S1) - Multi-stage progressive attack
    
    This attack creates a ghost vessel that:
    1. Follows the target vessel at a safe parallel distance
    2. Gradually approaches while increasing speed
    3. Executes a rapid crossing maneuver to trigger collision alert
    4. Disappears after causing the target to take evasive action
    """
    
    def __init__(self, config: FlashCrossConfig = None):
        self.config = config or FlashCrossConfig()
        self.ghost_vessel = None
        self.stage_executor = StageExecutor()
        self.attack_id = None
        self.results = {}
        
    def initialize_attack(self, 
                         target_data: Dict, 
                         start_position: Tuple[float, float]) -> str:
        """
        Initialize Flash Cross attack
        
        Args:
            target_data: Target vessel information and trajectory
            start_position: Initial position for ghost vessel
            
        Returns:
            Unique attack identifier
        """
        self.attack_id = f"flash_cross_{target_data['mmsi']}_{int(datetime.now().timestamp())}"
        
        # Create ghost vessel
        ghost_mmsi = self._generate_ghost_mmsi(target_data['mmsi'])
        self.ghost_vessel = GhostVessel(
            mmsi=ghost_mmsi,
            initial_position=start_position,
            vessel_type=target_data.get('vessel_type', 'cargo'),
            target_mmsi=target_data['mmsi']
        )
        
        logger.info(f"Initialized Flash Cross attack: {self.attack_id}")
        logger.info(f"Ghost vessel MMSI: {ghost_mmsi}, Target MMSI: {target_data['mmsi']}")
        
        return self.attack_id
    
    def execute_stage_1_parallel_following(self, 
                                         target_trajectory: List[Dict], 
                                         duration: float) -> List[Dict]:
        """
        Stage 1: Parallel Following
        
        Ghost vessel maintains parallel course with target at safe distance.
        This stage builds trust and establishes the ghost as a "normal" vessel.
        
        Args:
            target_trajectory: Target vessel's planned trajectory
            duration: Stage duration in seconds
            
        Returns:
            Ghost vessel trajectory for this stage
        """
        logger.info("Executing Stage 1: Parallel Following")
        
        ghost_trajectory = []
        parallel_offset = self.config.parallel_distance
        
        for i, target_point in enumerate(target_trajectory):
            if i * 30 > duration:  # 30-second intervals
                break
                
            # Calculate parallel position
            target_lat, target_lon = target_point['lat'], target_point['lon']
            target_course = target_point.get('course', 0)
            
            # Offset position perpendicular to target course
            parallel_lat, parallel_lon = self._calculate_parallel_position(
                target_lat, target_lon, target_course, parallel_offset
            )
            
            ghost_point = {
                'timestamp': target_point['timestamp'],
                'lat': parallel_lat,
                'lon': parallel_lon,
                'speed': target_point.get('speed', 10) * 0.95,  # Slightly slower
                'course': target_course,
                'mmsi': self.ghost_vessel.mmsi,
                'stage': 'parallel_following',
                'distance_to_target': parallel_offset
            }
            
            ghost_trajectory.append(ghost_point)
        
        logger.info(f"Stage 1 completed: {len(ghost_trajectory)} trajectory points generated")
        return ghost_trajectory
    
    def execute_stage_2_approach_initiation(self, 
                                          target_trajectory: List[Dict],
                                          ghost_start_position: Dict,
                                          duration: float) -> List[Dict]:
        """
        Stage 2: Approach Initiation
        
        Ghost vessel begins closing distance with gradual speed increase.
        Maintains deception by appearing to be a normal vessel making a
        course adjustment.
        
        Args:
            target_trajectory: Target vessel's trajectory
            ghost_start_position: Ghost vessel's position from previous stage
            duration: Stage duration in seconds
            
        Returns:
            Ghost vessel trajectory for this stage
        """
        logger.info("Executing Stage 2: Approach Initiation")
        
        ghost_trajectory = []
        start_distance = ghost_start_position['distance_to_target']
        target_distance = self.config.approach_cpa
        
        for i, target_point in enumerate(target_trajectory):
            if i * 10 > duration:  # 10-second intervals for finer control
                break
            
            # Calculate approach progress (0.0 to 1.0)
            progress = (i * 10) / duration
            
            # Interpolate distance reduction
            current_distance = start_distance - (start_distance - target_distance) * progress
            
            # Calculate approach position
            target_lat, target_lon = target_point['lat'], target_point['lon']
            target_course = target_point.get('course', 0)
            
            ghost_lat, ghost_lon = self._calculate_approach_position(
                target_lat, target_lon, target_course, current_distance, progress
            )
            
            # Increase speed gradually
            speed_factor = 1.0 + 0.2 * progress  # Up to 20% speed increase
            ghost_speed = target_point.get('speed', 10) * speed_factor
            
            ghost_point = {
                'timestamp': target_point['timestamp'],
                'lat': ghost_lat,
                'lon': ghost_lon,
                'speed': ghost_speed,
                'course': self._calculate_approach_course(target_course, progress),
                'mmsi': self.ghost_vessel.mmsi,
                'stage': 'approach_initiation',
                'distance_to_target': current_distance,
                'approach_progress': progress
            }
            
            ghost_trajectory.append(ghost_point)
        
        logger.info(f"Stage 2 completed: Approach from {start_distance:.2f}nm to {target_distance:.2f}nm")
        return ghost_trajectory
    
    def execute_stage_3_flash_cross_maneuver(self, 
                                           target_trajectory: List[Dict],
                                           ghost_start_position: Dict,
                                           duration: float) -> List[Dict]:
        """
        Stage 3: Flash Cross Maneuver
        
        Ghost vessel executes rapid crossing maneuver to trigger collision alert.
        This is the critical stage that causes the target to take evasive action.
        
        Args:
            target_trajectory: Target vessel's trajectory
            ghost_start_position: Ghost vessel's position from previous stage
            duration: Stage duration in seconds
            
        Returns:
            Ghost vessel trajectory for this stage
        """
        logger.info("Executing Stage 3: Flash Cross Maneuver")
        
        ghost_trajectory = []
        cross_angle = self.config.cross_angle
        max_speed_factor = self.config.ghost_speed_factor
        
        for i, target_point in enumerate(target_trajectory):
            if i * 5 > duration:  # 5-second intervals for precise control
                break
            
            progress = (i * 5) / duration
            target_lat, target_lon = target_point['lat'], target_point['lon']
            target_course = target_point.get('course', 0)
            target_speed = target_point.get('speed', 10)
            
            # Calculate crossing trajectory
            cross_course = (target_course + cross_angle) % 360
            cross_speed = target_speed * max_speed_factor
            
            # Calculate position for crossing maneuver
            ghost_lat, ghost_lon = self._calculate_cross_position(
                ghost_start_position, target_point, progress, cross_angle
            )
            
            # Calculate CPA at this point
            cpa_distance = self._calculate_cpa(
                (ghost_lat, ghost_lon), cross_course, cross_speed,
                (target_lat, target_lon), target_course, target_speed
            )
            
            ghost_point = {
                'timestamp': target_point['timestamp'],
                'lat': ghost_lat,
                'lon': ghost_lon,
                'speed': cross_speed,
                'course': cross_course,
                'mmsi': self.ghost_vessel.mmsi,
                'stage': 'flash_cross_maneuver',
                'cpa_distance': cpa_distance,
                'cross_progress': progress,
                'collision_risk': 'HIGH' if cpa_distance < 0.5 else 'MEDIUM'
            }
            
            ghost_trajectory.append(ghost_point)
            
            # Check if we've achieved minimum CPA
            if cpa_distance <= self.config.approach_cpa:
                logger.warning(f"Minimum CPA achieved: {cpa_distance:.3f}nm - COLLISION ALERT!")
        
        logger.info(f"Stage 3 completed: Flash cross maneuver executed")
        return ghost_trajectory
    
    def execute_stage_4_silent_disappearance(self, 
                                           ghost_last_position: Dict,
                                           duration: float) -> List[Dict]:
        """
        Stage 4: Silent Disappearance
        
        Ghost vessel disappears from AIS after causing target response.
        This stage simulates the attacker's exit strategy.
        
        Args:
            ghost_last_position: Ghost vessel's final position from previous stage
            duration: Stage duration in seconds
            
        Returns:
            Ghost vessel trajectory for this stage (empty - vessel disappears)
        """
        logger.info("Executing Stage 4: Silent Disappearance")
        
        # Create final position marker
        disappearance_point = {
            'timestamp': ghost_last_position['timestamp'],
            'lat': ghost_last_position['lat'],
            'lon': ghost_last_position['lon'],
            'speed': 0.0,
            'course': ghost_last_position['course'],
            'mmsi': self.ghost_vessel.mmsi,
            'stage': 'silent_disappearance',
            'status': 'DISAPPEARED',
            'message': 'Ghost vessel vanished from AIS'
        }
        
        logger.info("Stage 4 completed: Ghost vessel disappeared")
        return [disappearance_point]
    
    def _generate_ghost_mmsi(self, target_mmsi: str) -> str:
        """Generate unique ghost vessel MMSI"""
        import hashlib
        hash_input = f"flash_cross_{target_mmsi}_{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(hash_input.encode())
        ghost_id = int(hash_obj.hexdigest()[:8], 16) % 900000000 + 100000000
        return str(ghost_id)
    
    def _calculate_parallel_position(self, 
                                   target_lat: float, 
                                   target_lon: float,
                                   target_course: float, 
                                   offset_distance: float) -> Tuple[float, float]:
        """Calculate position parallel to target at specified distance"""
        # Convert to radians
        lat_rad = np.radians(target_lat)
        lon_rad = np.radians(target_lon)
        course_rad = np.radians(target_course + 90)  # Perpendicular offset
        
        # Earth radius in nautical miles
        R = 3440.065
        
        # Calculate offset position
        offset_lat = np.arcsin(np.sin(lat_rad) * np.cos(offset_distance/R) +
                              np.cos(lat_rad) * np.sin(offset_distance/R) * np.cos(course_rad))
        
        offset_lon = lon_rad + np.arctan2(np.sin(course_rad) * np.sin(offset_distance/R) * np.cos(lat_rad),
                                         np.cos(offset_distance/R) - np.sin(lat_rad) * np.sin(offset_lat))
        
        return np.degrees(offset_lat), np.degrees(offset_lon)
    
    def _calculate_approach_position(self, 
                                   target_lat: float, 
                                   target_lon: float,
                                   target_course: float, 
                                   distance: float,
                                   progress: float) -> Tuple[float, float]:
        """Calculate position during approach phase"""
        # Adjust approach angle based on progress
        approach_angle = 90 - 30 * progress  # Gradually turn toward target
        return self._calculate_parallel_position(target_lat, target_lon, 
                                               target_course + approach_angle, distance)
    
    def _calculate_approach_course(self, target_course: float, progress: float) -> float:
        """Calculate ghost vessel course during approach"""
        # Gradually adjust course toward crossing angle
        base_course = target_course
        cross_adjustment = self.config.cross_angle * progress * 0.3
        return (base_course + cross_adjustment) % 360
    
    def _calculate_cross_position(self, 
                                start_pos: Dict, 
                                target_point: Dict, 
                                progress: float,
                                cross_angle: float) -> Tuple[float, float]:
        """Calculate position during crossing maneuver"""
        # Implement crossing trajectory calculation
        # This is a simplified version - full implementation would use more sophisticated navigation
        start_lat, start_lon = start_pos['lat'], start_pos['lon']
        target_lat, target_lon = target_point['lat'], target_point['lon']
        
        # Linear interpolation for crossing path
        cross_lat = start_lat + (target_lat - start_lat) * progress * 0.8
        cross_lon = start_lon + (target_lon - start_lon) * progress * 0.8
        
        return cross_lat, cross_lon
    
    def _calculate_cpa(self, 
                      ghost_pos: Tuple[float, float], 
                      ghost_course: float, 
                      ghost_speed: float,
                      target_pos: Tuple[float, float], 
                      target_course: float, 
                      target_speed: float) -> float:
        """Calculate Closest Point of Approach between vessels"""
        # Simplified CPA calculation
        # Full implementation would use proper maritime navigation formulas
        
        # Calculate relative positions and velocities
        dx = target_pos[1] - ghost_pos[1]  # longitude difference
        dy = target_pos[0] - ghost_pos[0]  # latitude difference
        
        # Convert to distance (approximate)
        distance = np.sqrt(dx**2 + dy**2) * 60  # Convert to nautical miles
        
        # Simple CPA approximation
        # In practice, this would use vector calculations with proper geodetic formulas
        relative_speed = abs(ghost_speed - target_speed)
        if relative_speed > 0:
            time_to_cpa = distance / relative_speed
            cpa_distance = distance * 0.5  # Simplified calculation
        else:
            cpa_distance = distance
        
        return max(cpa_distance, 0.01)  # Minimum 0.01nm
