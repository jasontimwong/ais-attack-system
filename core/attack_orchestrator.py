"""
Attack Orchestrator - Multi-Stage Progressive Attack Coordination

Implements the Flash-Cross Strategy with 4-stage attack progression:
1. Parallel Following (2 minutes) - Establish tracking, build trust
2. Approach Initiation (30 seconds) - Gradual speed increase, maintain deception  
3. Flash Cross Maneuver (45 seconds) - Rapid approach, trigger collision alert
4. Silent Disappearance (30+ seconds) - Vanish after causing reaction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from .target_selector import TargetSelector
from .physics_engine import PhysicsEngine
from .colregs_validator import COLREGSValidator

logger = logging.getLogger(__name__)


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


class AttackOrchestrator:
    """
    Coordinates multi-stage AIS attacks with physics constraints and COLREGs compliance
    """
    
    def __init__(self, 
                 target_selector: TargetSelector,
                 physics_engine: PhysicsEngine,
                 colregs_validator: COLREGSValidator):
        self.target_selector = target_selector
        self.physics_engine = physics_engine
        self.colregs_validator = colregs_validator
        
        # Default Flash-Cross attack stages
        self.default_stages = [
            AttackStage(
                name="parallel_following",
                duration=120.0,  # 2 minutes
                speed_factor=1.0,
                course_deviation=0.0,
                distance_threshold=2.0,  # 2 nm parallel distance
                behavior="follow"
            ),
            AttackStage(
                name="approach_initiation", 
                duration=30.0,  # 30 seconds
                speed_factor=1.2,
                course_deviation=15.0,
                distance_threshold=1.0,  # 1 nm approach
                behavior="approach"
            ),
            AttackStage(
                name="flash_cross_maneuver",
                duration=45.0,  # 45 seconds
                speed_factor=1.5,
                course_deviation=90.0,
                distance_threshold=0.3,  # 0.3 nm CPA threshold
                behavior="cross"
            ),
            AttackStage(
                name="silent_disappearance",
                duration=30.0,  # 30+ seconds
                speed_factor=0.0,
                course_deviation=0.0,
                distance_threshold=5.0,  # 5 nm disappear distance
                behavior="disappear"
            )
        ]
    
    def select_target(self, vessels: List[Dict], constraints: Dict) -> Optional[str]:
        """
        Select optimal target vessel using MCDA + fuzzy logic
        
        Args:
            vessels: List of vessel data dictionaries
            constraints: Selection constraints
            
        Returns:
            Target vessel MMSI or None if no suitable target found
        """
        try:
            target = self.target_selector.select_optimal_target(vessels, constraints)
            if target:
                logger.info(f"Selected target vessel: {target['mmsi']} "
                          f"(vulnerability score: {target['vulnerability']:.3f})")
                return target['mmsi']
            else:
                logger.warning("No suitable target found")
                return None
        except Exception as e:
            logger.error(f"Target selection failed: {e}")
            return None
    
    def plan_attack(self, 
                   target_mmsi: str,
                   target_data: Dict,
                   attack_type: str = "flash_cross") -> AttackParameters:
        """
        Plan attack execution with physics constraints
        
        Args:
            target_mmsi: Target vessel MMSI
            target_data: Target vessel trajectory data
            attack_type: Type of attack to execute
            
        Returns:
            AttackParameters object with execution plan
        """
        # Generate ghost vessel MMSI
        ghost_mmsi = self._generate_ghost_mmsi(target_mmsi)
        
        # Calculate optimal start time
        start_time = self._calculate_start_time(target_data)
        
        # Apply physics constraints to stages
        constrained_stages = self._apply_physics_constraints(
            self.default_stages, target_data
        )
        
        # Create attack parameters
        attack_params = AttackParameters(
            target_mmsi=target_mmsi,
            ghost_mmsi=ghost_mmsi,
            start_time=start_time,
            stages=constrained_stages,
            physics_constraints=self.physics_engine.get_default_constraints(),
            colregs_compliance=True
        )
        
        logger.info(f"Planned {attack_type} attack: {target_mmsi} -> {ghost_mmsi}")
        return attack_params
    
    def execute_attack(self, attack_params: AttackParameters) -> Dict:
        """
        Execute multi-stage attack with real-time validation
        
        Args:
            attack_params: Attack execution parameters
            
        Returns:
            Attack execution results and metrics
        """
        results = {
            'attack_id': f"{attack_params.target_mmsi}_{attack_params.ghost_mmsi}",
            'start_time': attack_params.start_time,
            'stages': [],
            'metrics': {},
            'success': False,
            'colregs_violations': []
        }
        
        current_time = attack_params.start_time
        ghost_position = None
        target_response = None
        
        try:
            for stage in attack_params.stages:
                logger.info(f"Executing stage: {stage.name}")
                
                # Execute stage
                stage_result = self._execute_stage(
                    stage, current_time, ghost_position, attack_params
                )
                
                # Validate physics constraints
                if not self.physics_engine.validate_trajectory(
                    stage_result['trajectory'], attack_params.physics_constraints
                ):
                    logger.error(f"Physics validation failed in stage: {stage.name}")
                    break
                
                # Check COLREGs compliance if enabled
                if attack_params.colregs_compliance:
                    violations = self.colregs_validator.check_compliance(
                        stage_result['trajectory'], stage_result.get('target_trajectory')
                    )
                    results['colregs_violations'].extend(violations)
                
                results['stages'].append(stage_result)
                current_time += timedelta(seconds=stage.duration)
                ghost_position = stage_result.get('final_position')
                
                # Check for target response
                if stage_result.get('target_response'):
                    target_response = stage_result['target_response']
                    logger.info(f"Target response detected: {target_response['type']}")
            
            # Calculate attack effectiveness metrics
            results['metrics'] = self._calculate_metrics(results, target_response)
            results['success'] = target_response is not None
            
            logger.info(f"Attack execution completed. Success: {results['success']}")
            
        except Exception as e:
            logger.error(f"Attack execution failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_ghost_mmsi(self, target_mmsi: str) -> str:
        """Generate unique ghost vessel MMSI"""
        # Use hash of target MMSI + timestamp for uniqueness
        import hashlib
        hash_input = f"{target_mmsi}_{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(hash_input.encode())
        ghost_id = int(hash_obj.hexdigest()[:8], 16) % 900000000 + 100000000
        return str(ghost_id)
    
    def _calculate_start_time(self, target_data: Dict) -> datetime:
        """Calculate optimal attack start time based on target behavior"""
        # Analyze target trajectory for optimal timing
        # For now, return current time + buffer
        return datetime.now() + timedelta(seconds=60)
    
    def _apply_physics_constraints(self, 
                                 stages: List[AttackStage], 
                                 target_data: Dict) -> List[AttackStage]:
        """Apply physics constraints to attack stages"""
        constrained_stages = []
        
        for stage in stages:
            # Apply speed constraints based on vessel type and conditions
            max_speed = self.physics_engine.get_max_speed(target_data.get('vessel_type'))
            constrained_speed = min(stage.speed_factor * target_data.get('speed', 10), 
                                  max_speed)
            
            # Apply turn rate constraints
            max_turn_rate = self.physics_engine.get_max_turn_rate()
            constrained_deviation = min(abs(stage.course_deviation), 
                                      max_turn_rate * stage.duration)
            
            # Create constrained stage
            constrained_stage = AttackStage(
                name=stage.name,
                duration=stage.duration,
                speed_factor=constrained_speed / target_data.get('speed', 10),
                course_deviation=constrained_deviation * np.sign(stage.course_deviation),
                distance_threshold=stage.distance_threshold,
                behavior=stage.behavior
            )
            
            constrained_stages.append(constrained_stage)
        
        return constrained_stages
    
    def _execute_stage(self, 
                      stage: AttackStage, 
                      start_time: datetime,
                      initial_position: Optional[Tuple[float, float]],
                      attack_params: AttackParameters) -> Dict:
        """Execute individual attack stage"""
        
        # Stage execution logic would be implemented here
        # This is a simplified placeholder
        
        stage_result = {
            'stage_name': stage.name,
            'start_time': start_time,
            'duration': stage.duration,
            'trajectory': [],  # Would contain actual trajectory points
            'final_position': (0.0, 0.0),  # Would contain actual final position
            'target_response': None,  # Would contain detected target response
            'metrics': {
                'distance_traveled': 0.0,
                'max_speed': 0.0,
                'cpa_achieved': 0.0
            }
        }
        
        logger.debug(f"Stage {stage.name} executed successfully")
        return stage_result
    
    def _calculate_metrics(self, results: Dict, target_response: Optional[Dict]) -> Dict:
        """Calculate attack effectiveness metrics"""
        metrics = {
            'total_duration': sum(stage.get('duration', 0) for stage in results['stages']),
            'stages_completed': len(results['stages']),
            'target_response_triggered': target_response is not None,
            'colregs_violations_count': len(results['colregs_violations']),
            'attack_effectiveness': 0.0
        }
        
        # Calculate effectiveness score (0.0 - 1.0)
        if target_response:
            effectiveness = 0.5  # Base score for triggering response
            if target_response.get('type') == 'collision_avoidance':
                effectiveness += 0.3
            if target_response.get('severity') == 'high':
                effectiveness += 0.2
            metrics['attack_effectiveness'] = min(effectiveness, 1.0)
        
        return metrics
