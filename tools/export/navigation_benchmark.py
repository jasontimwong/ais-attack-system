#!/usr/bin/env python3
"""
Navigation System Benchmark Suite
Measures current navigation precision and establishes baseline for improvements.
"""

import sys
import os
import logging
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.navigation_fallback import NavigationFallback, SimpleAutopilot
from core.repair_logic import RepairLogic
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NavigationBenchmark:
    """
    Benchmark suite for navigation system precision testing.
    """
    
    def __init__(self, output_dir: str = "output/benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize navigation components with telemetry enabled
        self.navigation = NavigationFallback(enable_telemetry=True)
        self.autopilot = SimpleAutopilot(enable_telemetry=True)
        
        # Test scenarios
        self.test_routes = [
            # Short routes (< 10nm)
            ((22.3000, 114.1700), (22.3500, 114.2200), "Hong Kong Harbor"),
            ((37.7749, -122.4194), (37.8044, -122.4108), "San Francisco Bay"),
            
            # Medium routes (10-100nm)
            ((22.3000, 114.1700), (22.5000, 114.5000), "Hong Kong to Macau"),
            ((40.7128, -74.0060), (40.7831, -73.9712), "NYC to Newark"),
            ((51.5074, -0.1278), (51.4994, -0.1245), "London Thames"),
            
            # Long routes (>100nm)
            ((22.3000, 114.1700), (25.0330, 121.5654), "Hong Kong to Taipei"),
            ((37.7749, -122.4194), (34.0522, -118.2437), "SF to LA"),
            ((40.7128, -74.0060), (25.7617, -80.1918), "NYC to Miami"),
            
            # Very long routes (>500nm)
            ((22.3000, 114.1700), (35.6762, 139.6503), "Hong Kong to Tokyo"),
            ((51.5074, -0.1278), (40.7128, -74.0060), "London to NYC"),
        ]
        
        self.waypoint_intervals = [100.0, 50.0, 25.0, 10.0, 5.0]  # Test different densities
        
    def run_route_accuracy_benchmark(self) -> Dict[str, Any]:
        """
        Test route generation accuracy with different waypoint densities.
        
        Returns:
            Dictionary with route accuracy results
        """
        logger.info("Starting route accuracy benchmark...")
        
        results = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'test_routes': [],
            'waypoint_intervals': [],
            'summary': {}
        }
        
        for interval in self.waypoint_intervals:
            logger.info(f"Testing waypoint interval: {interval}nm")
            
            # Update navigation with new interval
            self.navigation.waypoint_interval_nm = interval
            
            interval_results = []
            
            for start, end, name in self.test_routes:
                try:
                    # Generate route
                    route = self.navigation.generate_route(start, end)
                    
                    # Calculate route metrics
                    planned_distance = route['properties']['distance']
                    actual_distance = self.navigation.great_circle_distance(
                        start[0], start[1], end[0], end[1]
                    )
                    
                    route_result = {
                        'route_name': name,
                        'start_pos': start,
                        'end_pos': end,
                        'planned_distance': planned_distance,
                        'actual_distance': actual_distance,
                        'distance_error': abs(planned_distance - actual_distance),
                        'waypoint_count': route['properties']['waypoint_count'],
                        'waypoint_density': route['properties']['waypoint_count'] / planned_distance
                    }
                    
                    interval_results.append(route_result)
                    
                except Exception as e:
                    logger.error(f"Error testing route {name}: {e}")
                    continue
            
            # Calculate interval statistics
            if interval_results:
                distance_errors = [r['distance_error'] for r in interval_results]
                waypoint_densities = [r['waypoint_density'] for r in interval_results]
                
                interval_stats = {
                    'waypoint_interval_nm': interval,
                    'routes_tested': len(interval_results),
                    'avg_distance_error': np.mean(distance_errors),
                    'max_distance_error': np.max(distance_errors),
                    'std_distance_error': np.std(distance_errors),
                    'avg_waypoint_density': np.mean(waypoint_densities),
                    'routes': interval_results
                }
                
                results['waypoint_intervals'].append(interval_stats)
                
                logger.info(f"Interval {interval}nm: avg_error={np.mean(distance_errors):.3f}nm, "
                           f"max_error={np.max(distance_errors):.3f}nm")
        
        # Generate summary statistics
        if results['waypoint_intervals']:
            all_errors = []
            all_densities = []
            
            for interval_data in results['waypoint_intervals']:
                for route in interval_data['routes']:
                    all_errors.append(route['distance_error'])
                    all_densities.append(route['waypoint_density'])
            
            results['summary'] = {
                'total_routes_tested': len(all_errors),
                'overall_avg_error': np.mean(all_errors),
                'overall_max_error': np.max(all_errors),
                'overall_std_error': np.std(all_errors),
                'overall_avg_density': np.mean(all_densities),
                'best_interval': min(results['waypoint_intervals'], 
                                   key=lambda x: x['avg_distance_error'])['waypoint_interval_nm']
            }
        
        logger.info("Route accuracy benchmark completed")
        return results
    
    def run_autopilot_precision_benchmark(self) -> Dict[str, Any]:
        """
        Test autopilot heading precision with simulated course changes.
        
        Returns:
            Dictionary with autopilot precision results
        """
        logger.info("Starting autopilot precision benchmark...")
        
        # Reset autopilot telemetry
        self.autopilot.clear_telemetry_data()
        
        # Simulate course changes
        test_scenarios = [
            # Small course changes (< 10°)
            [(0, 5, 60), (5, 10, 60), (10, 7, 60), (7, 3, 60)],
            
            # Medium course changes (10-45°)
            [(0, 30, 120), (30, 15, 120), (15, 45, 120), (45, 0, 120)],
            
            # Large course changes (45-90°)
            [(0, 90, 180), (90, 45, 180), (45, 135, 180), (135, 0, 180)],
            
            # Very large course changes (>90°)
            [(0, 180, 240), (180, 90, 240), (90, 270, 240), (270, 0, 240)],
        ]
        
        results = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'scenarios': [],
            'summary': {}
        }
        
        for scenario_idx, course_changes in enumerate(test_scenarios):
            logger.info(f"Testing scenario {scenario_idx + 1}: {len(course_changes)} course changes")
            
            scenario_results = []
            
            for current_hdg, desired_hdg, duration in course_changes:
                # Simulate course change over time
                steps = duration // 10  # 10-second steps
                
                for step in range(steps):
                    correction = self.autopilot.compute_heading_correction(
                        current_hdg, desired_hdg, 10.0
                    )
                    
                    # Apply correction (simplified)
                    current_hdg = (current_hdg + correction) % 360
                    
                    # Calculate error
                    error = abs(desired_hdg - current_hdg)
                    if error > 180:
                        error = 360 - error
                    
                    scenario_results.append({
                        'step': step,
                        'current_heading': current_hdg,
                        'desired_heading': desired_hdg,
                        'heading_error': error,
                        'correction': correction
                    })
            
            # Calculate scenario statistics
            if scenario_results:
                errors = [r['heading_error'] for r in scenario_results]
                corrections = [abs(r['correction']) for r in scenario_results]
                
                scenario_stats = {
                    'scenario_id': scenario_idx + 1,
                    'steps_simulated': len(scenario_results),
                    'avg_heading_error': np.mean(errors),
                    'max_heading_error': np.max(errors),
                    'std_heading_error': np.std(errors),
                    'avg_correction': np.mean(corrections),
                    'max_correction': np.max(corrections),
                    'steps': scenario_results
                }
                
                results['scenarios'].append(scenario_stats)
                
                logger.info(f"Scenario {scenario_idx + 1}: avg_error={np.mean(errors):.3f}°, "
                           f"max_error={np.max(errors):.3f}°")
        
        # Generate summary
        if results['scenarios']:
            all_errors = []
            all_corrections = []
            
            for scenario in results['scenarios']:
                for step in scenario['steps']:
                    all_errors.append(step['heading_error'])
                    all_corrections.append(abs(step['correction']))
            
            results['summary'] = {
                'total_steps': len(all_errors),
                'overall_avg_error': np.mean(all_errors),
                'overall_max_error': np.max(all_errors),
                'overall_std_error': np.std(all_errors),
                'overall_avg_correction': np.mean(all_corrections),
                'precision_grade': self._grade_precision(np.mean(all_errors)),
                'current_baseline': {
                    'avg_error_degrees': np.mean(all_errors),
                    'target_error_degrees': 0.031,  # Target precision
                    'improvement_factor_needed': np.mean(all_errors) / 0.031
                }
            }
        
        logger.info("Autopilot precision benchmark completed")
        return results
    
    def _grade_precision(self, avg_error: float) -> str:
        """Grade precision based on average error."""
        if avg_error <= 0.1:
            return 'Excellent'
        elif avg_error <= 0.5:
            return 'Good'
        elif avg_error <= 1.0:
            return 'Fair'
        elif avg_error <= 2.0:
            return 'Poor'
        else:
            return 'Critical'
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Returns:
            Complete benchmark results
        """
        logger.info("Starting complete navigation benchmark suite...")
        
        start_time = time.time()
        
        # Run route accuracy benchmark
        route_results = self.run_route_accuracy_benchmark()
        
        # Run autopilot precision benchmark
        autopilot_results = self.run_autopilot_precision_benchmark()
        
        # Combine results
        full_results = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': time.time() - start_time,
                'version': '1.0.0',
                'components_tested': ['NavigationFallback', 'SimpleAutopilot']
            },
            'route_accuracy': route_results,
            'autopilot_precision': autopilot_results,
            'overall_summary': {
                'route_accuracy_grade': self._grade_precision(route_results['summary']['overall_avg_error']),
                'autopilot_precision_grade': autopilot_results['summary']['precision_grade'],
                'current_baseline_error': autopilot_results['summary']['overall_avg_error'],
                'target_precision': 0.031,
                'improvement_factor_needed': autopilot_results['summary']['current_baseline']['improvement_factor_needed']
            }
        }
        
        # Save results
        output_file = self.output_dir / f"navigation_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"Complete benchmark results saved to {output_file}")
        
        # Log summary
        logger.info("=== BENCHMARK SUMMARY ===")
        logger.info(f"Route accuracy: {full_results['overall_summary']['route_accuracy_grade']}")
        logger.info(f"Autopilot precision: {full_results['overall_summary']['autopilot_precision_grade']}")
        logger.info(f"Current baseline error: {full_results['overall_summary']['current_baseline_error']:.3f}°")
        logger.info(f"Target precision: {full_results['overall_summary']['target_precision']:.3f}°")
        logger.info(f"Improvement factor needed: {full_results['overall_summary']['improvement_factor_needed']:.1f}x")
        
        return full_results


def main():
    """CLI entry point for navigation benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Navigation System Benchmark Suite")
    parser.add_argument("--output-dir", default="output/benchmark",
                       help="Output directory for benchmark results")
    parser.add_argument("--routes-only", action="store_true",
                       help="Only test route accuracy")
    parser.add_argument("--autopilot-only", action="store_true",
                       help="Only test autopilot precision")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = NavigationBenchmark(output_dir=args.output_dir)
    
    try:
        if args.routes_only:
            results = benchmark.run_route_accuracy_benchmark()
            print("Route accuracy benchmark completed")
        elif args.autopilot_only:
            results = benchmark.run_autopilot_precision_benchmark()
            print("Autopilot precision benchmark completed")
        else:
            results = benchmark.run_full_benchmark()
            print("Complete benchmark suite completed")
            
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())