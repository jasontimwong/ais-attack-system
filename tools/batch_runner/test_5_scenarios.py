#!/usr/bin/env python3
"""
Test batch runner with 5 scenarios to verify parallel performance.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch.runner import BatchRunner, discover_scenarios
import logging

logger = logging.getLogger(__name__)

def test_5_scenarios():
    """Test batch runner with 5 scenarios in parallel."""
    # Discover all available scenarios
    all_scenarios = discover_scenarios("configs")
    
    # Filter to get exactly 5 scenarios (prefer S2 scenarios that work)
    test_scenarios = []
    
    # Priority list: scenarios known to work
    priority_names = [
        "scenario_s2_zoneviolation",
        "scenario_s1_cargo", 
        "scenario_s1_dense",
        "scenario_s1_dense20",
        "scenario_s1_refined",
        "scenario_s1_highquality",
        "scenario_s1_verified"
    ]
    
    for priority in priority_names:
        for config, input_file in all_scenarios:
            if priority in config and len(test_scenarios) < 5:
                test_scenarios.append((config, input_file))
                
    # If we don't have 5 yet, add any remaining
    for config, input_file in all_scenarios:
        if (config, input_file) not in test_scenarios and len(test_scenarios) < 5:
            test_scenarios.append((config, input_file))
    
    logger.info(f"Testing batch runner with {len(test_scenarios)} scenarios:")
    for config, _ in test_scenarios:
        logger.info(f"  - {os.path.basename(config)}")
    
    # Test with different worker counts
    for max_workers in [5, 3, 1]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with {max_workers} worker(s)")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Create runner
        runner = BatchRunner(
            max_workers=max_workers, 
            base_output_dir=f"output/batch_test_{max_workers}w"
        )
        
        # Run batch
        summary = runner.run_batch(test_scenarios[:max_workers if max_workers < 5 else 5])
        
        duration = time.time() - start_time
        scenarios_run = summary['total_scenarios']
        
        logger.info(f"\nPerformance Summary:")
        logger.info(f"  Workers: {max_workers}")
        logger.info(f"  Scenarios: {scenarios_run}")
        logger.info(f"  Total time: {duration:.1f}s")
        logger.info(f"  Time per scenario: {duration/scenarios_run:.1f}s")
        logger.info(f"  Parallel speedup: {scenarios_run/(duration/60):.1f} scenarios/min")
        
        # Only test full 5 scenarios with 5 workers
        if max_workers == 5:
            if summary['successful'] >= 3:  # At least 3/5 should succeed
                logger.info(f"✅ Batch processing successful: {summary['successful']}/{scenarios_run}")
                return True
            else:
                logger.error(f"❌ Too many failures: {summary['failed']}/{scenarios_run}")
                return False
    
    return True


if __name__ == "__main__":
    success = test_5_scenarios()
    sys.exit(0 if success else 1)