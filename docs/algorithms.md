# 🧮 Algorithm Details

This document provides detailed technical descriptions of the core algorithms implemented in the AIS Attack Generation System.

## Table of Contents

- [Multi-Stage Progressive Attack Orchestration](#multi-stage-progressive-attack-orchestration)
- [MCDA + Fuzzy Logic Target Selection](#mcda--fuzzy-logic-target-selection)
- [MMG Physics Constraint Engine](#mmg-physics-constraint-engine)
- [COLREGs Compliance Validation](#colregs-compliance-validation)
- [Automated Attack Labeling](#automated-attack-labeling)
- [Performance Optimization](#performance-optimization)

## Multi-Stage Progressive Attack Orchestration

### Flash-Cross Strategy

The Flash-Cross strategy is our signature attack pattern that mimics real-world attack behaviors through a 4-stage progression:

#### Mathematical Model

The attack progression can be modeled as a state machine with temporal constraints:

```
S = {S₀, S₁, S₂, S₃}  // Attack stages
T = {t₀, t₁, t₂, t₃}  // Stage durations
C = {c₀, c₁, c₂, c₃}  // Stage constraints

Stage Transition Function:
f(Sᵢ, tᵢ, cᵢ) → Sᵢ₊₁

Where:
- S₀: Parallel Following (t₀ = 120s)
- S₁: Approach Initiation (t₁ = 30s)  
- S₂: Flash Cross Maneuver (t₂ = 45s)
- S₃: Silent Disappearance (t₃ = 30s)
```

#### Stage 0: Parallel Following

**Objective**: Establish tracking pattern and build trust

**Algorithm**:
```python
def calculate_parallel_position(target_pos, target_course, offset_distance):
    """
    Calculate ghost vessel position parallel to target
    """
    # Convert to radians
    lat_rad = radians(target_pos.lat)
    lon_rad = radians(target_pos.lon)
    course_rad = radians(target_course + 90)  # Perpendicular offset
    
    # Earth radius in nautical miles
    R = 3440.065
    
    # Calculate offset position using great circle navigation
    offset_lat = arcsin(sin(lat_rad) * cos(offset_distance/R) +
                       cos(lat_rad) * sin(offset_distance/R) * cos(course_rad))
    
    offset_lon = lon_rad + arctan2(sin(course_rad) * sin(offset_distance/R) * cos(lat_rad),
                                  cos(offset_distance/R) - sin(lat_rad) * sin(offset_lat))
    
    return degrees(offset_lat), degrees(offset_lon)
```

**Key Parameters**:
- Parallel distance: 2.0 nautical miles
- Speed factor: 0.95 (slightly slower than target)
- Course alignment: ±2° variation

#### Stage 1: Approach Initiation

**Objective**: Begin closing distance while maintaining deception

**Algorithm**:
```python
def calculate_approach_trajectory(start_distance, target_distance, duration, progress):
    """
    Calculate gradual approach trajectory
    """
    # Exponential approach curve for natural behavior
    distance_factor = exp(-2 * progress)  # Exponential decay
    current_distance = target_distance + (start_distance - target_distance) * distance_factor
    
    # Speed increase follows sigmoid curve
    speed_factor = 1.0 + 0.2 / (1 + exp(-10 * (progress - 0.5)))
    
    # Course adjustment for intercept
    intercept_angle = calculate_intercept_angle(current_distance, target_velocity)
    
    return current_distance, speed_factor, intercept_angle
```

**Key Parameters**:
- Target approach distance: 1.0 nautical miles
- Maximum speed increase: 20%
- Course deviation: ≤15°

#### Stage 2: Flash Cross Maneuver

**Objective**: Execute rapid crossing to trigger collision alert

**Algorithm**:
```python
def calculate_flash_cross_trajectory(ghost_pos, target_pos, cross_angle, progress):
    """
    Calculate crossing maneuver trajectory
    """
    # Calculate intercept point using vector mathematics
    target_velocity_vector = [target_speed * cos(radians(target_course)),
                             target_speed * sin(radians(target_course))]
    
    # Time to intercept calculation
    relative_position = [target_pos[0] - ghost_pos[0], target_pos[1] - ghost_pos[1]]
    time_to_intercept = calculate_intercept_time(relative_position, target_velocity_vector, ghost_speed)
    
    # Cross trajectory with collision risk calculation
    cross_course = (target_course + cross_angle) % 360
    ghost_velocity = ghost_speed * speed_factor
    
    # CPA calculation for collision alert triggering
    cpa_distance = calculate_cpa(ghost_pos, cross_course, ghost_velocity,
                                target_pos, target_course, target_speed)
    
    return cross_course, ghost_velocity, cpa_distance, time_to_intercept
```

**Key Parameters**:
- Cross angle: 90° (perpendicular crossing)
- Speed factor: 1.5× target speed
- Minimum CPA: 0.3 nautical miles

#### Stage 3: Silent Disappearance

**Objective**: Vanish after causing target response

**Algorithm**:
```python
def execute_disappearance(ghost_last_position, fade_duration):
    """
    Implement strategic disappearance
    """
    # Gradual signal degradation simulation
    for t in range(0, fade_duration, 5):  # 5-second intervals
        signal_strength = 1.0 - (t / fade_duration)  # Linear fade
        
        if signal_strength < 0.1:  # Complete disappearance threshold
            return None  # Ghost vessel vanished
        
        # Reduce position accuracy during fade
        position_noise = (1.0 - signal_strength) * 0.01  # Up to 0.01° noise
        
    return disappearance_marker
```

## MCDA + Fuzzy Logic Target Selection

### Multi-Criteria Decision Analysis

The target selection algorithm uses MCDA with four key criteria:

#### Mathematical Framework

```
Vulnerability Score = Σ(wᵢ × μᵢ(xᵢ))

Where:
- wᵢ = weight for criterion i
- μᵢ(xᵢ) = fuzzy membership function for criterion i
- xᵢ = normalized criterion value

Criteria:
- w₁ = 0.30 (Isolation Factor)
- w₂ = 0.25 (Predictability Score)  
- w₃ = 0.25 (Target Value)
- w₄ = 0.20 (Cascade Potential)
```

#### Isolation Factor Calculation

```python
def calculate_isolation_factor(vessel, nearby_vessels, time_window=300):
    """
    Calculate spatial-temporal isolation of target vessel
    """
    isolation_scores = []
    
    for t in range(0, time_window, 30):  # 30-second intervals
        # Spatial isolation
        distances = [haversine_distance(vessel.position(t), v.position(t)) 
                    for v in nearby_vessels]
        
        min_distance = min(distances) if distances else float('inf')
        spatial_isolation = min(min_distance / 5.0, 1.0)  # Normalize to 5nm
        
        # Temporal isolation (predictable time windows)
        temporal_isolation = calculate_temporal_isolation(vessel, t)
        
        # Combined isolation score
        combined_score = 0.7 * spatial_isolation + 0.3 * temporal_isolation
        isolation_scores.append(combined_score)
    
    return np.mean(isolation_scores)
```

#### Fuzzy Logic Implementation

```python
def fuzzy_membership_functions():
    """
    Define fuzzy membership functions for each criterion
    """
    # Trapezoidal membership functions
    isolation_low = [0, 0, 0.3, 0.5]
    isolation_medium = [0.2, 0.4, 0.6, 0.8]
    isolation_high = [0.6, 0.8, 1.0, 1.0]
    
    predictability_low = [0, 0, 0.4, 0.6]
    predictability_medium = [0.3, 0.5, 0.7, 0.9]
    predictability_high = [0.7, 0.9, 1.0, 1.0]
    
    return {
        'isolation': {'low': isolation_low, 'medium': isolation_medium, 'high': isolation_high},
        'predictability': {'low': predictability_low, 'medium': predictability_medium, 'high': predictability_high}
    }

def fuzzy_inference(isolation_score, predictability_score):
    """
    Fuzzy inference system for target selection
    """
    # Fuzzification
    isolation_membership = calculate_membership(isolation_score, fuzzy_sets['isolation'])
    predict_membership = calculate_membership(predictability_score, fuzzy_sets['predictability'])
    
    # Rule base
    rules = [
        ('high', 'high', 'excellent'),
        ('high', 'medium', 'good'),
        ('medium', 'high', 'good'),
        ('medium', 'medium', 'fair'),
        ('low', 'high', 'fair'),
        ('high', 'low', 'fair'),
        ('medium', 'low', 'poor'),
        ('low', 'medium', 'poor'),
        ('low', 'low', 'poor')
    ]
    
    # Defuzzification using centroid method
    return defuzzify_centroid(rules, isolation_membership, predict_membership)
```

## MMG Physics Constraint Engine

### 6-DOF Ship Dynamics Model

The Maneuvering Modeling Group (MMG) model provides realistic ship motion constraints:

#### Mathematical Model

```
Ship Motion Equations:
m(u̇ - vr) = X
m(v̇ + ur) = Y  
Iₓ ṙ = N

Where:
- m = ship mass
- u, v = surge and sway velocities
- r = yaw rate
- X, Y, N = forces and moment
- Iₓ = moment of inertia about z-axis
```

#### Force Components

```python
def calculate_mmg_forces(vessel_state, control_input):
    """
    Calculate MMG model forces and moments
    """
    u, v, r = vessel_state.velocities
    delta, n = control_input.rudder_angle, control_input.propeller_rpm
    
    # Hull forces
    X_H = calculate_hull_force_surge(u, v, r)
    Y_H = calculate_hull_force_sway(u, v, r)
    N_H = calculate_hull_moment_yaw(u, v, r)
    
    # Propeller forces
    X_P = calculate_propeller_thrust(u, n)
    
    # Rudder forces
    X_R, Y_R, N_R = calculate_rudder_forces(u, v, r, delta)
    
    # Total forces
    X_total = X_H + X_P + X_R
    Y_total = Y_H + Y_R
    N_total = N_H + N_R
    
    return X_total, Y_total, N_total

def calculate_hull_force_surge(u, v, r):
    """Hull resistance in surge direction"""
    # Non-dimensional coefficients
    X_uu = -0.0004  # Surge resistance coefficient
    X_vv = -0.0016  # Sway-induced surge force
    X_rr = -0.0012  # Yaw-induced surge force
    
    return X_uu * u * abs(u) + X_vv * v * abs(v) + X_rr * r * abs(r)
```

#### Constraint Validation

```python
def validate_physics_constraints(trajectory, vessel_params):
    """
    Validate trajectory against physics constraints
    """
    violations = []
    
    for i in range(1, len(trajectory)):
        dt = trajectory[i].timestamp - trajectory[i-1].timestamp
        
        # Speed change constraint
        speed_change = abs(trajectory[i].speed - trajectory[i-1].speed)
        max_speed_change = vessel_params.max_acceleration * dt / 60  # knots/min to knots/s
        
        if speed_change > max_speed_change:
            violations.append({
                'type': 'speed_change',
                'time': trajectory[i].timestamp,
                'violation': speed_change - max_speed_change
            })
        
        # Course change constraint
        course_change = abs(normalize_angle(trajectory[i].course - trajectory[i-1].course))
        max_course_change = vessel_params.max_turn_rate * dt  # degrees/s
        
        if course_change > max_course_change:
            violations.append({
                'type': 'course_change', 
                'time': trajectory[i].timestamp,
                'violation': course_change - max_course_change
            })
    
    return violations
```

## COLREGs Compliance Validation

### International Maritime Collision Avoidance Rules

Implementation of Rules 8, 13-17 for realistic vessel responses:

#### Rule Classification Algorithm

```python
def classify_encounter_situation(vessel_a, vessel_b):
    """
    Classify encounter type according to COLREGs
    """
    # Calculate relative bearing
    bearing_a_to_b = calculate_bearing(vessel_a.position, vessel_b.position)
    bearing_b_to_a = calculate_bearing(vessel_b.position, vessel_a.position)
    
    relative_bearing_a = normalize_angle(bearing_a_to_b - vessel_a.course)
    relative_bearing_b = normalize_angle(bearing_b_to_a - vessel_b.course)
    
    # Rule 13: Overtaking situation
    if is_overtaking(relative_bearing_a, relative_bearing_b):
        return 'overtaking'
    
    # Rule 14: Head-on situation
    elif is_head_on(vessel_a.course, vessel_b.course, relative_bearing_a):
        return 'head_on'
    
    # Rule 15: Crossing situation
    elif is_crossing(relative_bearing_a, relative_bearing_b):
        return 'crossing'
    
    return 'no_risk'

def is_head_on(course_a, course_b, relative_bearing):
    """Rule 14: Head-on situation detection"""
    course_difference = abs(normalize_angle(course_a - course_b))
    
    # Vessels on reciprocal courses (within ±6°)
    reciprocal_courses = 174 <= course_difference <= 186
    
    # Each vessel sees the other ahead (within ±6° of bow)
    ahead_bearing = abs(relative_bearing) <= 6
    
    return reciprocal_courses and ahead_bearing
```

#### CPA/TCPA Calculation

```python
def calculate_cpa_tcpa(vessel_a, vessel_b):
    """
    Calculate Closest Point of Approach and Time to CPA
    """
    # Position vectors
    pos_a = np.array([vessel_a.lat, vessel_a.lon])
    pos_b = np.array([vessel_b.lat, vessel_b.lon])
    
    # Velocity vectors (convert to m/s)
    vel_a = np.array([
        vessel_a.speed * np.cos(np.radians(vessel_a.course)) * 0.514444,
        vessel_a.speed * np.sin(np.radians(vessel_a.course)) * 0.514444
    ])
    vel_b = np.array([
        vessel_b.speed * np.cos(np.radians(vessel_b.course)) * 0.514444,
        vessel_b.speed * np.sin(np.radians(vessel_b.course)) * 0.514444
    ])
    
    # Relative position and velocity
    rel_pos = pos_b - pos_a
    rel_vel = vel_b - vel_a
    
    # Time to CPA
    if np.dot(rel_vel, rel_vel) < 1e-6:  # Parallel courses
        tcpa = float('inf')
        cpa = np.linalg.norm(rel_pos)
    else:
        tcpa = -np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel)
        
        # CPA distance
        if tcpa < 0:  # CPA in the past
            cpa = np.linalg.norm(rel_pos)
        else:
            cpa_pos = rel_pos + tcpa * rel_vel
            cpa = np.linalg.norm(cpa_pos)
    
    # Convert CPA to nautical miles
    cpa_nm = cpa / 1852.0
    
    return cpa_nm, tcpa
```

## Automated Attack Labeling

### Multi-Level Labeling System

The automated labeling system generates comprehensive attack metadata:

#### Label Generation Algorithm

```python
def generate_attack_labels(attack_results):
    """
    Generate comprehensive attack labels and metadata
    """
    labels = {
        'attack_metadata': {
            'attack_id': attack_results['attack_id'],
            'attack_type': attack_results['attack_type'],
            'execution_time': attack_results['execution_time'],
            'success_rate': attack_results['success_rate']
        },
        'trajectory_labels': [],
        'interaction_labels': [],
        'impact_labels': {}
    }
    
    # Trajectory-level labels
    for stage in attack_results['stages']:
        stage_labels = {
            'stage_name': stage['name'],
            'start_time': stage['start_time'],
            'duration': stage['duration'],
            'trajectory_points': len(stage['trajectory']),
            'physics_valid': stage['physics_valid'],
            'colregs_compliant': stage['colregs_compliant']
        }
        labels['trajectory_labels'].append(stage_labels)
    
    # Interaction-level labels
    for interaction in attack_results['interactions']:
        interaction_label = {
            'timestamp': interaction['timestamp'],
            'encounter_type': interaction['encounter_type'],
            'cpa_distance': interaction['cpa_distance'],
            'tcpa': interaction['tcpa'],
            'risk_level': classify_risk_level(interaction['cpa_distance']),
            'colregs_rule_applied': interaction['colregs_rule'],
            'target_response': interaction['target_response']
        }
        labels['interaction_labels'].append(interaction_label)
    
    # Impact assessment
    labels['impact_labels'] = assess_attack_impact(attack_results)
    
    return labels
```

#### Impact Assessment

```python
def assess_attack_impact(attack_results):
    """
    Assess the impact and effectiveness of the attack
    """
    impact_metrics = {
        'target_response_triggered': False,
        'evasive_maneuver_executed': False,
        'colregs_violation_induced': False,
        'collision_risk_created': False,
        'traffic_disruption_level': 0,
        'attack_detectability': 0.0
    }
    
    # Analyze target responses
    for interaction in attack_results['interactions']:
        if interaction['target_response']:
            impact_metrics['target_response_triggered'] = True
            
            if interaction['target_response']['type'] == 'evasive_maneuver':
                impact_metrics['evasive_maneuver_executed'] = True
            
            if interaction['cpa_distance'] < 0.5:  # High collision risk
                impact_metrics['collision_risk_created'] = True
    
    # Calculate detectability score
    impact_metrics['attack_detectability'] = calculate_detectability_score(attack_results)
    
    return impact_metrics
```

## Performance Optimization

### Vectorized Trajectory Processing

```python
import numpy as np
from numba import jit

@jit(nopython=True)
def vectorized_cpa_calculation(positions_a, velocities_a, positions_b, velocities_b):
    """
    Vectorized CPA calculation for batch processing
    """
    n = len(positions_a)
    cpa_distances = np.zeros(n)
    tcpa_times = np.zeros(n)
    
    for i in range(n):
        rel_pos = positions_b[i] - positions_a[i]
        rel_vel = velocities_b[i] - velocities_a[i]
        
        rel_vel_squared = np.dot(rel_vel, rel_vel)
        
        if rel_vel_squared < 1e-6:
            tcpa_times[i] = np.inf
            cpa_distances[i] = np.linalg.norm(rel_pos)
        else:
            tcpa_times[i] = -np.dot(rel_pos, rel_vel) / rel_vel_squared
            
            if tcpa_times[i] < 0:
                cpa_distances[i] = np.linalg.norm(rel_pos)
            else:
                cpa_pos = rel_pos + tcpa_times[i] * rel_vel
                cpa_distances[i] = np.linalg.norm(cpa_pos)
    
    return cpa_distances, tcpa_times
```

### Parallel Processing Architecture

```python
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import asyncio

class ParallelAttackProcessor:
    """
    Parallel processing system for batch attack generation
    """
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or mp.cpu_count()
    
    async def process_scenarios_async(self, scenarios):
        """
        Asynchronous scenario processing
        """
        loop = asyncio.get_event_loop()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, self.process_single_scenario, scenario)
                for scenario in scenarios
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def process_single_scenario(self, scenario):
        """
        Process individual scenario with optimizations
        """
        # Implement scenario processing with:
        # - Memory-mapped file I/O
        # - Vectorized calculations
        # - Caching of intermediate results
        # - Early termination conditions
        
        return processed_scenario
```

### Memory Optimization

```python
class MemoryEfficientTrajectoryProcessor:
    """
    Memory-efficient processing for large-scale trajectory data
    """
    
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
        self.cache = {}
    
    def process_trajectory_stream(self, trajectory_stream):
        """
        Process trajectory data in chunks to minimize memory usage
        """
        for chunk in self.chunk_iterator(trajectory_stream):
            # Process chunk with vectorized operations
            processed_chunk = self.vectorized_processing(chunk)
            
            # Yield results immediately to avoid memory accumulation
            yield processed_chunk
    
    def chunk_iterator(self, data_stream):
        """
        Iterator that yields data in chunks
        """
        chunk = []
        for item in data_stream:
            chunk.append(item)
            
            if len(chunk) >= self.chunk_size:
                yield np.array(chunk)
                chunk = []
        
        if chunk:  # Yield remaining items
            yield np.array(chunk)
```

---

These algorithms form the foundation of the AIS Attack Generation System, providing both theoretical rigor and practical performance for maritime cybersecurity research.
