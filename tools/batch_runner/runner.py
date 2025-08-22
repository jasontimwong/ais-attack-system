#!/usr/bin/env python3
"""
Batch processing runner for AIS attack scenarios.
Supports parallel execution of multiple scenarios with performance monitoring.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ScenarioResult:
    """Result from running a single scenario."""
    
    def __init__(self, scenario_name: str, success: bool, 
                 duration: float, output_dir: Optional[str] = None,
                 error: Optional[str] = None, metrics: Optional[Dict] = None):
        self.scenario_name = scenario_name
        self.success = success
        self.duration = duration
        self.output_dir = output_dir
        self.error = error
        self.metrics = metrics or {}
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'scenario_name': self.scenario_name,
            'success': self.success,
            'duration': self.duration,
            'output_dir': self.output_dir,
            'error': self.error,
            'metrics': self.metrics
        }


class BatchRunner:
    """Batch runner for AIS attack scenarios."""
    
    def __init__(self, max_workers: int = 5, base_output_dir: str = "output/batch"):
        self.max_workers = max_workers
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_scenario(self, config_file: str, input_file: str) -> ScenarioResult:
        """Run a single scenario and return results."""
        scenario_name = Path(config_file).stem
        start_time = time.time()
        
        # Create unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_output_dir / f"{scenario_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[{scenario_name}] Starting scenario execution...")
        
        try:
            # Build command
            cmd = [
                sys.executable,
                "generate_attack_data.py",
                "-c", config_file,
                "-i", input_file,
                "-o", str(output_dir),
                "--plot"
            ]
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per scenario
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"[{scenario_name}] ✅ Completed successfully in {duration:.1f}s")
                
                # Extract metrics if available
                metrics = self._extract_metrics(output_dir)
                
                return ScenarioResult(
                    scenario_name=scenario_name,
                    success=True,
                    duration=duration,
                    output_dir=str(output_dir),
                    metrics=metrics
                )
            else:
                error_msg = result.stderr or "Unknown error"
                logger.error(f"[{scenario_name}] ❌ Failed: {error_msg}")
                
                return ScenarioResult(
                    scenario_name=scenario_name,
                    success=False,
                    duration=duration,
                    output_dir=str(output_dir),
                    error=error_msg
                )
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Timeout after {duration:.1f}s"
            logger.error(f"[{scenario_name}] ⏱️ {error_msg}")
            
            return ScenarioResult(
                scenario_name=scenario_name,
                success=False,
                duration=duration,
                output_dir=str(output_dir),
                error=error_msg
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.error(f"[{scenario_name}] 💥 Exception: {error_msg}")
            
            return ScenarioResult(
                scenario_name=scenario_name,
                success=False,
                duration=duration,
                output_dir=str(output_dir),
                error=error_msg
            )
    
    def _extract_metrics(self, output_dir: Path) -> Dict:
        """Extract key metrics from scenario output."""
        metrics = {}
        
        # Try to load diff file
        diff_files = list(output_dir.glob("diff*.json"))
        if diff_files:
            try:
                with open(diff_files[0], 'r') as f:
                    diff_data = json.load(f)
                    metrics['messages_added'] = diff_data.get('metrics', {}).get('added_messages', 0)
                    metrics['baseline_messages'] = diff_data.get('metrics', {}).get('baseline_messages', 0)
            except Exception as e:
                logger.warning(f"Failed to load diff file: {e}")
        
        # Try to load effectiveness file
        effectiveness_file = output_dir / "attack_effectiveness.json"
        if effectiveness_file.exists():
            try:
                with open(effectiveness_file, 'r') as f:
                    effectiveness = json.load(f)
                    metrics['overall_effective'] = effectiveness.get('overall_effective', False)
                    metrics['severity'] = effectiveness.get('severity', 'unknown')
            except Exception as e:
                logger.warning(f"Failed to load effectiveness file: {e}")
        
        # Check for visualization
        viz_files = list((output_dir / "viz").glob("*.png")) if (output_dir / "viz").exists() else []
        metrics['visualizations_generated'] = len(viz_files)
        
        return metrics
    
    def run_batch(self, scenario_configs: List[Tuple[str, str]]) -> Dict:
        """Run multiple scenarios in parallel."""
        logger.info(f"Starting batch execution of {len(scenario_configs)} scenarios")
        logger.info(f"Max parallel workers: {self.max_workers}")
        
        start_time = time.time()
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_scenario = {
                executor.submit(self.run_scenario, config, input_file): (config, input_file)
                for config, input_file in scenario_configs
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_scenario):
                config, input_file = future_to_scenario[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress update
                    progress = completed / len(scenario_configs) * 100
                    logger.info(f"Progress: {completed}/{len(scenario_configs)} ({progress:.0f}%)")
                    
                except Exception as e:
                    logger.error(f"Failed to get result for {config}: {e}")
                    results.append(ScenarioResult(
                        scenario_name=Path(config).stem,
                        success=False,
                        duration=0,
                        error=str(e)
                    ))
        
        # Calculate summary statistics
        total_duration = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        summary = {
            'total_scenarios': len(results),
            'successful': successful,
            'failed': failed,
            'total_duration': total_duration,
            'average_duration': total_duration / len(results) if results else 0,
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        # Save summary report
        summary_file = self.base_output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Execution Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total scenarios: {len(results)}")
        logger.info(f"Successful: {successful} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Total duration: {total_duration:.1f}s")
        logger.info(f"Average duration: {summary['average_duration']:.1f}s per scenario")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"{'='*60}\n")
        
        return summary


def discover_scenarios(config_dir: str = "configs") -> List[Tuple[str, str]]:
    """Discover all scenario configuration files."""
    config_path = Path(config_dir)
    scenarios = []
    
    # Default input file
    default_input = "data/AIS_2020_01_01.csv"
    
    # Find all scenario YAML files
    for config_file in sorted(config_path.glob("scenario_*.yaml")):
        # Skip test and validation scenarios
        if any(skip in config_file.name for skip in ['test', 'validation', 'defaults']):
            continue
            
        scenarios.append((str(config_file), default_input))
        logger.info(f"Discovered scenario: {config_file.name}")
    
    return scenarios


def main():
    """Main entry point for batch runner."""
    parser = argparse.ArgumentParser(description='Batch runner for AIS attack scenarios')
    parser.add_argument('-w', '--workers', type=int, default=5,
                        help='Maximum number of parallel workers (default: 5)')
    parser.add_argument('-o', '--output', default='output/batch',
                        help='Base output directory (default: output/batch)')
    parser.add_argument('-c', '--config-dir', default='configs',
                        help='Directory containing scenario configs (default: configs)')
    parser.add_argument('--scenarios', nargs='*',
                        help='Specific scenario names to run (runs all if not specified)')
    
    args = parser.parse_args()
    
    # Discover scenarios
    all_scenarios = discover_scenarios(args.config_dir)
    
    # Filter scenarios if specified
    if args.scenarios:
        scenarios = []
        for config, input_file in all_scenarios:
            scenario_name = Path(config).stem
            if any(s in scenario_name for s in args.scenarios):
                scenarios.append((config, input_file))
    else:
        scenarios = all_scenarios
    
    if not scenarios:
        logger.error("No scenarios found to execute")
        return 1
    
    logger.info(f"\nFound {len(scenarios)} scenarios to execute")
    
    # Create batch runner
    runner = BatchRunner(max_workers=args.workers, base_output_dir=args.output)
    
    # Run batch
    summary = runner.run_batch(scenarios)
    
    # Return exit code based on results
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())