#!/usr/bin/env python3
"""
Physics Engine - MMG Ship Dynamics Model

6-DOF ship dynamics model with physics constraints for realistic vessel movement
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VesselState:
    """Vessel state representation"""
    lat: float
    lon: float
    speed: float  # knots
    course: float  # degrees
    acceleration: float = 0.0  # knots/minute
    turn_rate: float = 0.0  # degrees/second
    timestamp: float = 0.0

@dataclass
class PhysicsConstraints:
    """Physics constraints for vessel movement"""
    max_speed: float = 25.0  # knots
    max_acceleration: float = 2.0  # knots/minute
    max_turn_rate: float = 10.0  # degrees/second
    min_cpa: float = 0.1  # nautical miles
    vessel_length: float = 100.0  # meters
    vessel_beam: float = 15.0  # meters

class PhysicsEngine:
    """
    6-DOF ship dynamics model with physics constraints
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize physics engine with MMG parameters
        
        Args:
            config: Physics configuration with vessel constraints
        """
        self.config = config or self._default_config()
        self.constraints = PhysicsConstraints(**self.config.get('constraints', {}))
        
    def _default_config(self) -> Dict:
        """Default physics configuration"""
        return {
            'constraints': {
                'max_speed': 25.0,
                'max_acceleration': 2.0,
                'max_turn_rate': 10.0,
                'min_cpa': 0.1,
                'vessel_length': 100.0,
                'vessel_beam': 15.0
            },
            'mmg_parameters': {
                'mass_coefficient': 1.05,
                'added_mass_x': 0.05,
                'added_mass_y': 0.95,
                'drag_coefficient_x': 0.022,
                'drag_coefficient_y': 0.67,
                'moment_inertia': 0.4
            },
            'environmental': {
                'water_density': 1025.0,  # kg/m³
                'current_speed': 0.0,  # knots
                'current_direction': 0.0,  # degrees
                'wind_speed': 0.0,  # knots
                'wind_direction': 0.0  # degrees
            }
        }
    
    def validate_trajectory(self, trajectory: List[VesselState]) -> Tuple[bool, List[str]]:
        """
        Validate trajectory against physics constraints
        
        Args:
            trajectory: List of vessel states forming a trajectory
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        if len(trajectory) < 2:
            return True, []
        
        violations = []
        
        for i in range(1, len(trajectory)):
            prev_state = trajectory[i-1]
            curr_state = trajectory[i]
            
            # Calculate time difference
            dt = curr_state.timestamp - prev_state.timestamp
            if dt <= 0:
                violations.append(f"Invalid time sequence at index {i}")
                continue
            
            # Speed constraints
            if curr_state.speed > self.constraints.max_speed:
                violations.append(f"Speed {curr_state.speed:.1f} kts exceeds maximum {self.constraints.max_speed:.1f} kts at index {i}")
            
            # Acceleration constraints
            speed_change = curr_state.speed - prev_state.speed
            acceleration = speed_change / (dt / 60.0)  # knots per minute
            
            if abs(acceleration) > self.constraints.max_acceleration:
                violations.append(f"Acceleration {acceleration:.2f} kts/min exceeds limit {self.constraints.max_acceleration:.2f} at index {i}")
            
            # Turn rate constraints
            course_change = self._normalize_angle(curr_state.course - prev_state.course)
            turn_rate = abs(course_change) / dt  # degrees per second
            
            if turn_rate > self.constraints.max_turn_rate:
                violations.append(f"Turn rate {turn_rate:.2f} deg/s exceeds limit {self.constraints.max_turn_rate:.2f} at index {i}")
            
            # Position jump validation
            distance = self._haversine_distance(
                prev_state.lat, prev_state.lon,
                curr_state.lat, curr_state.lon
            )
            
            expected_distance = prev_state.speed * (dt / 3600.0)  # nautical miles
            
            if distance > expected_distance * 1.5:  # Allow 50% tolerance
                violations.append(f"Position jump {distance:.3f} nm exceeds expected {expected_distance:.3f} nm at index {i}")
        
        return len(violations) == 0, violations
    
    def calculate_mmg_forces(self, state: VesselState, control_input: Dict) -> Tuple[float, float, float]:
        """
        Calculate MMG model forces and moments
        
        Args:
            state: Current vessel state
            control_input: Control inputs (rudder_angle, propeller_rpm)
            
        Returns:
            Tuple of (surge_force, sway_force, yaw_moment)
        """
        # Convert speed to m/s
        u = state.speed * 0.514444
        v = 0.0  # Assume no initial sway velocity
        r = np.radians(state.turn_rate)
        
        # Control inputs
        rudder_angle = np.radians(control_input.get('rudder_angle', 0.0))
        propeller_rpm = control_input.get('propeller_rpm', 100.0)
        
        # MMG parameters
        mmg = self.config['mmg_parameters']
        L = self.constraints.vessel_length
        B = self.constraints.vessel_beam
        
        # Hull forces
        X_H = self._calculate_hull_force_surge(u, v, r, L, B)
        Y_H = self._calculate_hull_force_sway(u, v, r, L, B)
        N_H = self._calculate_hull_moment_yaw(u, v, r, L, B)
        
        # Propeller forces
        X_P = self._calculate_propeller_thrust(u, propeller_rpm, L)
        
        # Rudder forces
        X_R, Y_R, N_R = self._calculate_rudder_forces(u, v, r, rudder_angle, L, B)
        
        # Total forces
        X_total = X_H + X_P + X_R
        Y_total = Y_H + Y_R
        N_total = N_H + N_R
        
        return X_total, Y_total, N_total
    
    def predict_trajectory(self, initial_state: VesselState, 
                          control_sequence: List[Dict],
                          time_step: float = 1.0) -> List[VesselState]:
        """
        Predict vessel trajectory using MMG model
        
        Args:
            initial_state: Starting vessel state
            control_sequence: Sequence of control inputs
            time_step: Time step in seconds
            
        Returns:
            Predicted trajectory as list of vessel states
        """
        trajectory = [initial_state]
        current_state = initial_state
        
        for i, control_input in enumerate(control_sequence):
            # Calculate MMG forces
            X, Y, N = self.calculate_mmg_forces(current_state, control_input)
            
            # Integrate equations of motion
            next_state = self._integrate_motion(current_state, X, Y, N, time_step)
            next_state.timestamp = current_state.timestamp + time_step
            
            trajectory.append(next_state)
            current_state = next_state
        
        return trajectory
    
    def get_max_speed(self, vessel_type: str) -> float:
        """
        Get maximum speed for vessel type
        
        Args:
            vessel_type: Type of vessel (cargo, tanker, container, passenger)
            
        Returns:
            Maximum speed in knots
        """
        max_speeds = {
            'cargo': 18.0,
            'tanker': 16.0,
            'container': 24.0,
            'passenger': 28.0,
            'fishing': 15.0,
            'pleasure': 35.0
        }
        
        return max_speeds.get(vessel_type, self.constraints.max_speed)
    
    def get_max_turn_rate(self, vessel_type: str = None) -> float:
        """
        Get maximum turn rate based on IMO standards
        
        Args:
            vessel_type: Optional vessel type for specific limits
            
        Returns:
            Maximum turn rate in degrees per second
        """
        # IMO standard: vessel should be able to turn 90° in less than 240 seconds
        # This gives approximately 0.375°/s, but we use more conservative values
        
        turn_rates = {
            'cargo': 3.0,
            'tanker': 2.0,
            'container': 4.0,
            'passenger': 5.0,
            'fishing': 8.0,
            'pleasure': 10.0
        }
        
        return turn_rates.get(vessel_type, self.constraints.max_turn_rate)
    
    def _calculate_hull_force_surge(self, u: float, v: float, r: float, L: float, B: float) -> float:
        """Calculate hull resistance in surge direction"""
        # Non-dimensional coefficients (typical values)
        X_uu = -0.0004
        X_vv = -0.0016
        X_rr = -0.0012
        
        rho = self.config['environmental']['water_density']
        
        return 0.5 * rho * L * B * (X_uu * u * abs(u) + X_vv * v * abs(v) + X_rr * r * abs(r))
    
    def _calculate_hull_force_sway(self, u: float, v: float, r: float, L: float, B: float) -> float:
        """Calculate hull force in sway direction"""
        Y_v = -0.315
        Y_r = 0.083
        Y_vvv = -1.607
        Y_rrr = 0.008
        
        rho = self.config['environmental']['water_density']
        
        return 0.5 * rho * L * B * (Y_v * v + Y_r * r * L + Y_vvv * v * abs(v) + Y_rrr * r * abs(r) * L**2)
    
    def _calculate_hull_moment_yaw(self, u: float, v: float, r: float, L: float, B: float) -> float:
        """Calculate hull moment about yaw axis"""
        N_v = -0.137
        N_r = -0.049
        N_vvv = -0.030
        N_rrr = -0.007
        
        rho = self.config['environmental']['water_density']
        
        return 0.5 * rho * L**2 * B * (N_v * v + N_r * r * L + N_vvv * v * abs(v) + N_rrr * r * abs(r) * L**2)
    
    def _calculate_propeller_thrust(self, u: float, rpm: float, L: float) -> float:
        """Calculate propeller thrust"""
        # Simplified propeller model
        D_prop = L * 0.06  # Propeller diameter as fraction of length
        J = u / (rpm / 60.0 * D_prop)  # Advance ratio
        
        # Thrust coefficient (simplified)
        K_T = max(0.0, 0.4 - 0.3 * J)
        
        rho = self.config['environmental']['water_density']
        
        return rho * (rpm / 60.0)**2 * D_prop**4 * K_T
    
    def _calculate_rudder_forces(self, u: float, v: float, r: float, 
                               delta: float, L: float, B: float) -> Tuple[float, float, float]:
        """Calculate rudder forces and moment"""
        # Effective inflow velocity at rudder
        u_R = u * (1.0 - 0.42)  # Wake fraction
        v_R = v + r * L * 0.5
        
        alpha_R = np.arctan2(v_R, u_R) - delta
        U_R = np.sqrt(u_R**2 + v_R**2)
        
        # Rudder force
        A_R = L * B * 0.05  # Rudder area as fraction
        C_L = 2 * np.pi * np.sin(alpha_R)  # Lift coefficient
        
        F_N = 0.5 * self.config['environmental']['water_density'] * A_R * U_R**2 * C_L
        
        # Force components
        X_R = -F_N * np.sin(delta)
        Y_R = F_N * np.cos(delta)
        N_R = Y_R * L * 0.5  # Moment arm
        
        return X_R, Y_R, N_R
    
    def _integrate_motion(self, state: VesselState, X: float, Y: float, N: float, dt: float) -> VesselState:
        """Integrate equations of motion"""
        # Convert current state
        u = state.speed * 0.514444  # m/s
        course_rad = np.radians(state.course)
        
        # Mass and inertia (simplified)
        L = self.constraints.vessel_length
        B = self.constraints.vessel_beam
        m = self.config['environmental']['water_density'] * L * B * 5.0  # Rough estimate
        I_z = m * L**2 / 12.0  # Moment of inertia
        
        # Accelerations
        du_dt = X / m
        r = np.radians(state.turn_rate)
        dr_dt = N / I_z
        
        # Update velocities
        u_new = u + du_dt * dt
        r_new = r + dr_dt * dt
        
        # Convert back to navigation units
        speed_new = u_new / 0.514444  # knots
        turn_rate_new = np.degrees(r_new)  # degrees/second
        course_new = state.course + turn_rate_new * dt
        
        # Update position (simplified great circle approximation)
        distance = speed_new * dt / 3600.0  # nautical miles
        lat_new, lon_new = self._project_position(state.lat, state.lon, course_new, distance)
        
        return VesselState(
            lat=lat_new,
            lon=lon_new,
            speed=max(0.0, speed_new),
            course=self._normalize_angle(course_new),
            turn_rate=turn_rate_new
        )
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in nautical miles"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return c * 3440.065  # Earth radius in nautical miles
    
    def _project_position(self, lat: float, lon: float, course: float, distance: float) -> Tuple[float, float]:
        """Project position given course and distance"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        course_rad = np.radians(course)
        
        # Earth radius in nautical miles
        R = 3440.065
        
        lat_new = np.arcsin(np.sin(lat_rad) * np.cos(distance/R) + 
                           np.cos(lat_rad) * np.sin(distance/R) * np.cos(course_rad))
        
        lon_new = lon_rad + np.arctan2(np.sin(course_rad) * np.sin(distance/R) * np.cos(lat_rad),
                                      np.cos(distance/R) - np.sin(lat_rad) * np.sin(lat_new))
        
        return np.degrees(lat_new), np.degrees(lon_new)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to 0-360 range"""
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle
