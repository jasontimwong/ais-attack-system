# 📚 API Reference

This document provides a comprehensive reference for the AIS Attack Generation System API.

## Table of Contents

- [Core Components](#core-components)
- [Attack Types](#attack-types)
- [Visualization](#visualization)
- [Data Models](#data-models)
- [Configuration](#configuration)
- [Examples](#examples)

## Core Components

### AttackOrchestrator

The main orchestrator for coordinating multi-stage AIS attacks.

```python
from core.attack_orchestrator import AttackOrchestrator

class AttackOrchestrator:
    """
    Coordinates multi-stage AIS attacks with physics constraints and COLREGs compliance
    """
    
    def __init__(self, target_selector, physics_engine, colregs_validator):
        """
        Initialize the attack orchestrator.
        
        Args:
            target_selector (TargetSelector): MCDA target selection component
            physics_engine (PhysicsEngine): Physics constraint validation
            colregs_validator (COLREGSValidator): COLREGs compliance checker
        """
    
    def select_target(self, vessels: List[Dict], constraints: Dict) -> Optional[str]:
        """
        Select optimal target vessel using MCDA + fuzzy logic.
        
        Args:
            vessels: List of vessel data dictionaries
            constraints: Selection constraints
            
        Returns:
            Target vessel MMSI or None if no suitable target found
        """
    
    def plan_attack(self, target_mmsi: str, target_data: Dict, attack_type: str = "flash_cross") -> AttackParameters:
        """
        Plan attack execution with physics constraints.
        
        Args:
            target_mmsi: Target vessel MMSI
            target_data: Target vessel trajectory data
            attack_type: Type of attack to execute
            
        Returns:
            AttackParameters object with execution plan
        """
    
    def execute_attack(self, attack_params: AttackParameters) -> Dict:
        """
        Execute multi-stage attack with real-time validation.
        
        Args:
            attack_params: Attack execution parameters
            
        Returns:
            Attack execution results and metrics
        """
```

### TargetSelector

Implements MCDA + fuzzy logic for intelligent target selection.

```python
from core.target_selector import TargetSelector

class TargetSelector:
    """
    Multi-criteria target selection with vulnerability scoring
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize target selector with MCDA weights.
        
        Args:
            config: Configuration dictionary with weights and fuzzy parameters
        """
    
    def select_optimal_target(self, vessels: List[Dict], constraints: Dict) -> Optional[Dict]:
        """
        Select the most vulnerable target vessel.
        
        Args:
            vessels: List of available vessels
            constraints: Selection constraints
            
        Returns:
            Selected target with vulnerability score
        """
    
    def calculate_vulnerability_score(self, vessel: Dict) -> float:
        """
        Calculate vulnerability score for a vessel.
        
        Args:
            vessel: Vessel data dictionary
            
        Returns:
            Vulnerability score (0.0 - 1.0)
        """
```

### PhysicsEngine

Enforces maritime physics constraints using MMG ship dynamics model.

```python
from core.physics_engine import PhysicsEngine

class PhysicsEngine:
    """
    6-DOF ship dynamics model with physics constraints
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize physics engine with MMG parameters.
        
        Args:
            config: Physics configuration with vessel constraints
        """
    
    def validate_trajectory(self, trajectory: List[Dict], constraints: Dict) -> bool:
        """
        Validate trajectory against physics constraints.
        
        Args:
            trajectory: List of trajectory points
            constraints: Physics constraints
            
        Returns:
            True if trajectory is physically valid
        """
    
    def get_max_speed(self, vessel_type: str) -> float:
        """
        Get maximum speed for vessel type.
        
        Args:
            vessel_type: Type of vessel (cargo, tanker, container, passenger)
            
        Returns:
            Maximum speed in knots
        """
    
    def get_max_turn_rate(self) -> float:
        """
        Get maximum turn rate based on IMO standards.
        
        Returns:
            Maximum turn rate in degrees per second
        """
```

### COLREGSValidator

Validates compliance with International Maritime Collision Avoidance Rules.

```python
from core.colregs_validator import COLREGSValidator

class COLREGSValidator:
    """
    COLREGs compliance validation for maritime collision avoidance
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize COLREGs validator.
        
        Args:
            config: COLREGs configuration with rule parameters
        """
    
    def check_compliance(self, interactions: List[Dict]) -> List[Dict]:
        """
        Check vessel interactions for COLREGs violations.
        
        Args:
            interactions: List of vessel interaction scenarios
            
        Returns:
            List of detected violations
        """
    
    def classify_encounter(self, vessel_a: Dict, vessel_b: Dict) -> str:
        """
        Classify encounter type between two vessels.
        
        Args:
            vessel_a: First vessel data
            vessel_b: Second vessel data
            
        Returns:
            Encounter type (head_on, crossing, overtaking)
        """
```

## Attack Types

### Flash Cross Attack (S1)

The signature 4-stage progressive attack pattern.

```python
from attacks.flash_cross import FlashCrossAttack

class FlashCrossAttack:
    """
    S1 Flash Cross Attack - Multi-stage progressive attack
    """
    
    def __init__(self, config: FlashCrossConfig = None):
        """
        Initialize Flash Cross attack.
        
        Args:
            config: Attack configuration parameters
        """
    
    def initialize_attack(self, target_data: Dict, start_position: Tuple[float, float]) -> str:
        """
        Initialize Flash Cross attack.
        
        Args:
            target_data: Target vessel information and trajectory
            start_position: Initial position for ghost vessel
            
        Returns:
            Unique attack identifier
        """
    
    def execute_stage_1_parallel_following(self, target_trajectory: List[Dict], duration: float) -> List[Dict]:
        """
        Stage 1: Parallel Following
        
        Args:
            target_trajectory: Target vessel's planned trajectory
            duration: Stage duration in seconds
            
        Returns:
            Ghost vessel trajectory for this stage
        """
    
    def execute_stage_2_approach_initiation(self, target_trajectory: List[Dict], ghost_start_position: Dict, duration: float) -> List[Dict]:
        """
        Stage 2: Approach Initiation
        
        Args:
            target_trajectory: Target vessel's trajectory
            ghost_start_position: Ghost vessel's position from previous stage
            duration: Stage duration in seconds
            
        Returns:
            Ghost vessel trajectory for this stage
        """
    
    def execute_stage_3_flash_cross_maneuver(self, target_trajectory: List[Dict], ghost_start_position: Dict, duration: float) -> List[Dict]:
        """
        Stage 3: Flash Cross Maneuver
        
        Args:
            target_trajectory: Target vessel's trajectory
            ghost_start_position: Ghost vessel's position from previous stage
            duration: Stage duration in seconds
            
        Returns:
            Ghost vessel trajectory for this stage
        """
    
    def execute_stage_4_silent_disappearance(self, ghost_last_position: Dict, duration: float) -> List[Dict]:
        """
        Stage 4: Silent Disappearance
        
        Args:
            ghost_last_position: Ghost vessel's final position from previous stage
            duration: Stage duration in seconds
            
        Returns:
            Ghost vessel trajectory for this stage (empty - vessel disappears)
        """
```

### Other Attack Types

All attack types follow similar patterns:

- **S2: Zone Violation** - `attacks.zone_violation.ZoneViolationAttack`
- **S3: Ghost Swarm** - `attacks.ghost_swarm.GhostSwarmAttack`
- **S4: Position Offset** - `attacks.position_offset.PositionOffsetAttack`
- **S5: Port Spoofing** - `attacks.port_spoofing.PortSpoofingAttack`
- **S6: Course Disruption** - `attacks.course_disruption.CourseDisruptionAttack`
- **S7: Identity Swap** - `attacks.identity_swap.IdentitySwapAttack`
- **S8: Identity Clone** - `attacks.identity_clone.IdentityCloneAttack`
- **S9: Identity Whitewashing** - `attacks.identity_whitewashing.IdentityWhitewashingAttack`

## Visualization

### ECDIS Renderer

Professional maritime chart visualization.

```python
from visualization.ecdis_renderer import ECDISRenderer

class ECDISRenderer:
    """
    Professional ECDIS chart rendering with IMO standards
    """
    
    def __init__(self, style: str = "imo_standard"):
        """
        Initialize ECDIS renderer.
        
        Args:
            style: Chart style (imo_standard, dark, light)
        """
    
    def render_scenario(self, scenario_data: Dict, output_path: str) -> str:
        """
        Render complete attack scenario on ECDIS chart.
        
        Args:
            scenario_data: Attack scenario data
            output_path: Output file path
            
        Returns:
            Path to generated chart image
        """
    
    def add_trajectory(self, trajectory: List[Dict], style: Dict = None) -> None:
        """
        Add vessel trajectory to chart.
        
        Args:
            trajectory: Vessel trajectory points
            style: Rendering style options
        """
```

### Web Interface

Interactive web-based visualization.

```python
from visualization.web_interface import WebInterface

class WebInterface:
    """
    Web-based interactive visualization interface
    """
    
    def __init__(self, port: int = 5173):
        """
        Initialize web interface.
        
        Args:
            port: Server port number
        """
    
    def start_server(self) -> None:
        """Start the web visualization server."""
    
    def load_scenario(self, scenario_path: str) -> None:
        """
        Load attack scenario for visualization.
        
        Args:
            scenario_path: Path to scenario data
        """
```

## Data Models

### Vessel Data

```python
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class VesselData:
    """Vessel information and state"""
    mmsi: str
    lat: float
    lon: float
    speed: float  # knots
    course: float  # degrees
    timestamp: str
    vessel_type: str  # cargo, tanker, container, passenger
    length: Optional[float] = None  # meters
    beam: Optional[float] = None  # meters
```

### Attack Parameters

```python
@dataclass
class AttackStage:
    """Attack stage configuration"""
    name: str
    duration: float  # seconds
    speed_factor: float
    course_deviation: float  # degrees
    distance_threshold: float  # nautical miles
    behavior: str

@dataclass
class AttackParameters:
    """Attack execution parameters"""
    target_mmsi: str
    ghost_mmsi: str
    start_time: datetime
    stages: List[AttackStage]
    physics_constraints: Dict
    colregs_compliance: bool = True
```

### Configuration Models

```python
@dataclass
class FlashCrossConfig:
    """Configuration for Flash Cross attack"""
    parallel_distance: float = 2.0  # nautical miles
    approach_cpa: float = 0.3  # nautical miles
    cross_angle: float = 90.0  # degrees
    ghost_speed_factor: float = 1.5  # multiplier of target speed
    stage_durations: Dict[str, float] = None
```

## Configuration

### System Configuration

```yaml
# System configuration example
system:
  version: "1.0.0"
  log_level: "INFO"
  output_format: "geojson"

physics:
  max_turn_rate: 3.0  # degrees per second
  max_acceleration: 0.5  # knots per minute
  min_cpa_threshold: 0.1  # nautical miles

target_selection:
  algorithm: "mcda_fuzzy"
  weights:
    isolation_factor: 0.3
    predictability_score: 0.25
    target_value: 0.25
    cascade_potential: 0.2
```

### Attack Configuration

```yaml
# Attack configuration example
attacks:
  s1_flash_cross:
    enabled: true
    stages:
      parallel_following:
        duration: 120.0  # seconds
        parallel_distance: 2.0  # nautical miles
      approach_initiation:
        duration: 30.0
        speed_increase: 0.2
      flash_cross_maneuver:
        duration: 45.0
        cross_angle: 90.0
        speed_factor: 1.5
      silent_disappearance:
        duration: 30.0
```

## Examples

### Basic Usage

```python
#!/usr/bin/env python3
from core.attack_orchestrator import AttackOrchestrator
from core.target_selector import TargetSelector
from core.physics_engine import PhysicsEngine
from core.colregs_validator import COLREGSValidator
from attacks.flash_cross import FlashCrossAttack

# Initialize components
target_selector = TargetSelector()
physics_engine = PhysicsEngine()
colregs_validator = COLREGSValidator()

orchestrator = AttackOrchestrator(
    target_selector, physics_engine, colregs_validator
)

# Create attack
attack = FlashCrossAttack()

# Define target
target_data = {
    'mmsi': '123456789',
    'lat': 40.7128,
    'lon': -74.0060,
    'speed': 12.0,
    'course': 90.0,
    'vessel_type': 'cargo'
}

# Execute attack
attack_id = attack.initialize_attack(target_data, (40.7100, -74.0100))
print(f"Attack initialized: {attack_id}")
```

### Batch Processing

```python
from tools.batch_runner import BatchRunner

# Initialize batch runner
runner = BatchRunner("configs/default_attack_config.yaml")

# Run all scenarios
results = runner.run_all_scenarios(parallel=True, max_workers=4)

# Print summary
print(f"Success rate: {results['summary']['success_rate']:.1%}")
print(f"Average quality: {results['summary']['avg_quality_score']:.3f}")
```

### Custom Configuration

```python
import yaml
from attacks.flash_cross import FlashCrossAttack, FlashCrossConfig

# Load custom configuration
with open("configs/custom_attack.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Create custom attack configuration
flash_config = FlashCrossConfig(
    parallel_distance=1.5,
    cross_angle=120.0,
    ghost_speed_factor=2.0
)

# Initialize attack with custom config
attack = FlashCrossAttack(flash_config)
```

## Error Handling

### Common Exceptions

```python
from core.exceptions import (
    AttackExecutionError,
    PhysicsValidationError,
    TargetSelectionError,
    COLREGSViolationError
)

try:
    results = orchestrator.execute_attack(attack_params)
except AttackExecutionError as e:
    print(f"Attack execution failed: {e}")
except PhysicsValidationError as e:
    print(f"Physics validation failed: {e}")
except TargetSelectionError as e:
    print(f"Target selection failed: {e}")
except COLREGSViolationError as e:
    print(f"COLREGs violation detected: {e}")
```

## Performance Considerations

### Memory Usage

- Use streaming processing for large datasets
- Implement data chunking for batch operations
- Monitor memory usage with built-in profilers

### Processing Speed

- Enable parallel processing for batch scenarios
- Use vectorized operations for trajectory calculations
- Implement caching for repeated calculations

### Scalability

- Configure appropriate worker counts for your system
- Use database backends for large-scale operations
- Implement distributed processing for enterprise use

---

For more detailed examples and advanced usage patterns, see the [tutorials](tutorials/) directory.
