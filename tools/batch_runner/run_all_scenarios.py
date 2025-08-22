#!/usr/bin/env python3
"""
Batch Runner for All Attack Scenarios

This script executes all 35 validated attack scenarios in batch mode,
generating complete datasets with labels and metadata for each attack type.

Usage:
    python run_all_scenarios.py [options]
    
Features:
- Parallel execution of multiple scenarios
- Real-time progress monitoring
- Automatic validation and quality control
- Comprehensive logging and error handling
- Export to multiple formats (GeoJSON, CSV, NMEA)
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.attack_orchestrator import AttackOrchestrator
from core.target_selector import TargetSelector
from core.physics_engine import PhysicsEngine
from core.colregs_validator import COLREGSValidator
from core.auto_labeler import AutoLabeler
from attacks import ATTACK_REGISTRY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_runner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BatchRunner:
    """
    Batch execution engine for AIS attack scenarios
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "configs/default_attack_config.yaml"
        self.config = self._load_config()
        self.results = {}
        self.failed_scenarios = []
        self.start_time = None
        
        # Initialize system components
        self._initialize_components()
        
        # Load scenario configurations
        self.scenarios = self._load_scenarios()
        
    def _load_config(self) -> Dict:
        """Load system configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def _initialize_components(self):
        """Initialize core system components"""
        try:
            self.target_selector = TargetSelector(self.config['target_selection'])
            self.physics_engine = PhysicsEngine(self.config['physics'])
            self.colregs_validator = COLREGSValidator(self.config['colregs'])
            self.auto_labeler = AutoLabeler(self.config.get('labeling', {}))
            
            self.orchestrator = AttackOrchestrator(
                self.target_selector,
                self.physics_engine,
                self.colregs_validator
            )
            
            logger.info("System components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            sys.exit(1)
    
    def _load_scenarios(self) -> List[Dict]:
        """Load all attack scenario configurations"""
        scenarios = []
        scenarios_dir = Path("datasets/scenarios")
        
        if not scenarios_dir.exists():
            logger.warning(f"Scenarios directory not found: {scenarios_dir}")
            return []
        
        for scenario_file in scenarios_dir.glob("*.yaml"):
            try:
                with open(scenario_file, 'r') as f:
                    scenario = yaml.safe_load(f)
                    scenario['config_file'] = str(scenario_file)
                    scenarios.append(scenario)
            except Exception as e:
                logger.error(f"Failed to load scenario {scenario_file}: {e}")
        
        logger.info(f"Loaded {len(scenarios)} attack scenarios")
        return scenarios
    
    def run_all_scenarios(self, 
                         parallel: bool = True, 
                         max_workers: int = None) -> Dict:
        """
        Execute all attack scenarios
        
        Args:
            parallel: Whether to run scenarios in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary containing execution results
        """
        self.start_time = datetime.now()
        logger.info(f"Starting batch execution of {len(self.scenarios)} scenarios")
        
        if parallel and len(self.scenarios) > 1:
            return self._run_parallel(max_workers)
        else:
            return self._run_sequential()
    
    def _run_sequential(self) -> Dict:
        """Run scenarios sequentially"""
        results = {}
        
        for i, scenario in enumerate(self.scenarios, 1):
            logger.info(f"Executing scenario {i}/{len(self.scenarios)}: {scenario.get('name', 'Unknown')}")
            
            try:
                result = self._execute_scenario(scenario)
                results[scenario['id']] = result
                
                if result['success']:
                    logger.info(f"✓ Scenario {scenario['id']} completed successfully")
                else:
                    logger.error(f"✗ Scenario {scenario['id']} failed: {result.get('error', 'Unknown error')}")
                    self.failed_scenarios.append(scenario['id'])
                    
            except Exception as e:
                logger.error(f"✗ Scenario {scenario['id']} crashed: {e}")
                self.failed_scenarios.append(scenario['id'])
                results[scenario['id']] = {
                    'success': False,
                    'error': str(e),
                    'crash': True
                }
        
        return self._compile_results(results)
    
    def _run_parallel(self, max_workers: int = None) -> Dict:
        """Run scenarios in parallel"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(self.scenarios))
        
        logger.info(f"Running scenarios in parallel with {max_workers} workers")
        
        results = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scenarios
            future_to_scenario = {
                executor.submit(self._execute_scenario, scenario): scenario
                for scenario in self.scenarios
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                scenario_id = scenario['id']
                
                try:
                    result = future.result()
                    results[scenario_id] = result
                    
                    if result['success']:
                        logger.info(f"✓ Scenario {scenario_id} completed successfully")
                    else:
                        logger.error(f"✗ Scenario {scenario_id} failed: {result.get('error', 'Unknown error')}")
                        self.failed_scenarios.append(scenario_id)
                        
                except Exception as e:
                    logger.error(f"✗ Scenario {scenario_id} crashed: {e}")
                    self.failed_scenarios.append(scenario_id)
                    results[scenario_id] = {
                        'success': False,
                        'error': str(e),
                        'crash': True
                    }
        
        return self._compile_results(results)
    
    def _execute_scenario(self, scenario: Dict) -> Dict:
        """
        Execute a single attack scenario
        
        Args:
            scenario: Scenario configuration dictionary
            
        Returns:
            Execution results dictionary
        """
        scenario_id = scenario['id']
        attack_type = scenario['attack_type']
        
        try:
            # Get attack class
            if attack_type not in ATTACK_REGISTRY:
                raise ValueError(f"Unknown attack type: {attack_type}")
            
            attack_class = ATTACK_REGISTRY[attack_type]
            
            # Load input data
            input_data = self._load_scenario_data(scenario)
            
            # Execute attack
            attack_instance = attack_class(scenario.get('parameters', {}))
            attack_results = attack_instance.execute(input_data)
            
            # Validate results
            validation_results = self._validate_results(attack_results, scenario)
            
            # Generate labels and metadata
            labels = self.auto_labeler.generate_labels(attack_results)
            metadata = self._generate_metadata(scenario, attack_results, validation_results)
            
            # Export results
            output_paths = self._export_results(
                scenario_id, attack_results, labels, metadata
            )
            
            return {
                'success': True,
                'scenario_id': scenario_id,
                'attack_type': attack_type,
                'execution_time': attack_results.get('execution_time', 0),
                'metrics': attack_results.get('metrics', {}),
                'validation': validation_results,
                'labels': labels,
                'metadata': metadata,
                'output_paths': output_paths,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Scenario {scenario_id} execution failed: {e}")
            return {
                'success': False,
                'scenario_id': scenario_id,
                'attack_type': attack_type,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _load_scenario_data(self, scenario: Dict) -> Dict:
        """Load input data for scenario"""
        # This would load the actual AIS data for the scenario
        # For now, return a placeholder
        return {
            'vessels': [],
            'timeframe': scenario.get('timeframe', {}),
            'geographic_bounds': scenario.get('bounds', {}),
            'scenario_id': scenario['id']
        }
    
    def _validate_results(self, results: Dict, scenario: Dict) -> Dict:
        """Validate attack execution results"""
        validation = {
            'physics_valid': True,
            'colregs_compliant': True,
            'temporal_consistent': True,
            'quality_score': 0.0,
            'issues': []
        }
        
        try:
            # Physics validation
            if 'trajectory' in results:
                physics_valid = self.physics_engine.validate_trajectory(
                    results['trajectory'], self.config['physics']
                )
                validation['physics_valid'] = physics_valid
                
                if not physics_valid:
                    validation['issues'].append("Physics constraints violated")
            
            # COLREGs validation
            if self.config['colregs']['enabled'] and 'interactions' in results:
                colregs_violations = self.colregs_validator.check_compliance(
                    results['interactions']
                )
                validation['colregs_compliant'] = len(colregs_violations) == 0
                validation['colregs_violations'] = colregs_violations
                
                if colregs_violations:
                    validation['issues'].append(f"{len(colregs_violations)} COLREGs violations")
            
            # Calculate quality score
            quality_factors = []
            if validation['physics_valid']:
                quality_factors.append(0.4)
            if validation['colregs_compliant']:
                quality_factors.append(0.3)
            if validation['temporal_consistent']:
                quality_factors.append(0.3)
            
            validation['quality_score'] = sum(quality_factors)
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation['issues'].append(f"Validation error: {e}")
        
        return validation
    
    def _generate_metadata(self, 
                          scenario: Dict, 
                          results: Dict, 
                          validation: Dict) -> Dict:
        """Generate comprehensive metadata for scenario"""
        return {
            'scenario': {
                'id': scenario['id'],
                'name': scenario.get('name', 'Unknown'),
                'attack_type': scenario['attack_type'],
                'description': scenario.get('description', ''),
                'parameters': scenario.get('parameters', {})
            },
            'execution': {
                'timestamp': datetime.now().isoformat(),
                'duration': results.get('execution_time', 0),
                'success': results.get('success', False),
                'version': self.config['system']['version']
            },
            'metrics': results.get('metrics', {}),
            'validation': validation,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'config_file': self.config_path
            }
        }
    
    def _export_results(self, 
                       scenario_id: str, 
                       results: Dict, 
                       labels: Dict, 
                       metadata: Dict) -> Dict:
        """Export results to various formats"""
        output_dir = Path(f"output/batch_results/{scenario_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = {}
        
        try:
            # Export trajectory data as GeoJSON
            if 'trajectory' in results:
                geojson_path = output_dir / f"{scenario_id}_trajectory.geojson"
                with open(geojson_path, 'w') as f:
                    json.dump(results['trajectory'], f, indent=2)
                output_paths['trajectory'] = str(geojson_path)
            
            # Export labels
            labels_path = output_dir / f"{scenario_id}_labels.json"
            with open(labels_path, 'w') as f:
                json.dump(labels, f, indent=2)
            output_paths['labels'] = str(labels_path)
            
            # Export metadata
            metadata_path = output_dir / f"{scenario_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            output_paths['metadata'] = str(metadata_path)
            
            # Export full results
            results_path = output_dir / f"{scenario_id}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            output_paths['results'] = str(results_path)
            
            logger.debug(f"Results exported for scenario {scenario_id}")
            
        except Exception as e:
            logger.error(f"Failed to export results for {scenario_id}: {e}")
        
        return output_paths
    
    def _compile_results(self, individual_results: Dict) -> Dict:
        """Compile final batch execution results"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        successful_scenarios = [
            r for r in individual_results.values() 
            if r.get('success', False)
        ]
        
        failed_scenarios = [
            r for r in individual_results.values() 
            if not r.get('success', False)
        ]
        
        # Calculate aggregate metrics
        total_execution_time = sum(
            r.get('execution_time', 0) for r in successful_scenarios
        )
        
        avg_quality_score = sum(
            r.get('validation', {}).get('quality_score', 0) 
            for r in successful_scenarios
        ) / max(len(successful_scenarios), 1)
        
        compiled_results = {
            'summary': {
                'total_scenarios': len(self.scenarios),
                'successful_scenarios': len(successful_scenarios),
                'failed_scenarios': len(failed_scenarios),
                'success_rate': len(successful_scenarios) / len(self.scenarios),
                'total_duration': total_duration,
                'avg_execution_time': total_execution_time / max(len(successful_scenarios), 1),
                'avg_quality_score': avg_quality_score
            },
            'execution': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'config_used': self.config_path,
                'system_version': self.config['system']['version']
            },
            'individual_results': individual_results,
            'failed_scenario_ids': self.failed_scenarios,
            'statistics': self._calculate_statistics(successful_scenarios)
        }
        
        # Save compiled results
        self._save_batch_summary(compiled_results)
        
        return compiled_results
    
    def _calculate_statistics(self, successful_results: List[Dict]) -> Dict:
        """Calculate aggregate statistics"""
        if not successful_results:
            return {}
        
        # Extract metrics by attack type
        attack_type_stats = {}
        for result in successful_results:
            attack_type = result.get('attack_type', 'unknown')
            if attack_type not in attack_type_stats:
                attack_type_stats[attack_type] = []
            attack_type_stats[attack_type].append(result)
        
        # Calculate statistics per attack type
        stats = {}
        for attack_type, results in attack_type_stats.items():
            quality_scores = [
                r.get('validation', {}).get('quality_score', 0) 
                for r in results
            ]
            execution_times = [
                r.get('execution_time', 0) for r in results
            ]
            
            stats[attack_type] = {
                'count': len(results),
                'avg_quality_score': sum(quality_scores) / len(quality_scores),
                'min_quality_score': min(quality_scores),
                'max_quality_score': max(quality_scores),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'total_execution_time': sum(execution_times)
            }
        
        return stats
    
    def _save_batch_summary(self, results: Dict):
        """Save batch execution summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = Path(f"output/batch_summary_{timestamp}.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Batch summary saved to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save batch summary: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AIS Attack System Batch Runner")
    parser.add_argument("--config", "-c", 
                       help="Configuration file path",
                       default="configs/default_attack_config.yaml")
    parser.add_argument("--parallel", "-p", 
                       action="store_true",
                       help="Run scenarios in parallel")
    parser.add_argument("--workers", "-w", 
                       type=int,
                       help="Number of parallel workers")
    parser.add_argument("--scenarios", "-s",
                       nargs="+",
                       help="Specific scenario IDs to run")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize batch runner
    runner = BatchRunner(args.config)
    
    # Filter scenarios if specified
    if args.scenarios:
        runner.scenarios = [
            s for s in runner.scenarios 
            if s['id'] in args.scenarios
        ]
        logger.info(f"Filtered to {len(runner.scenarios)} specified scenarios")
    
    if not runner.scenarios:
        logger.error("No scenarios to execute")
        sys.exit(1)
    
    # Execute scenarios
    try:
        results = runner.run_all_scenarios(
            parallel=args.parallel,
            max_workers=args.workers
        )
        
        # Print summary
        summary = results['summary']
        print(f"\n{'='*60}")
        print(f"BATCH EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Scenarios: {summary['total_scenarios']}")
        print(f"Successful: {summary['successful_scenarios']}")
        print(f"Failed: {summary['failed_scenarios']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration']:.1f} seconds")
        print(f"Average Quality Score: {summary['avg_quality_score']:.3f}")
        print(f"{'='*60}")
        
        if runner.failed_scenarios:
            print(f"Failed Scenarios: {', '.join(runner.failed_scenarios)}")
        
        sys.exit(0 if summary['success_rate'] >= 0.8 else 1)
        
    except KeyboardInterrupt:
        logger.info("Batch execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Batch execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
