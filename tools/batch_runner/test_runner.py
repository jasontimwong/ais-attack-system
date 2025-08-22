#!/usr/bin/env python3
"""
Test script for batch runner with limited scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch.runner import BatchRunner
import logging

logger = logging.getLogger(__name__)

def test_batch_runner():
    """Test batch runner with a few scenarios."""
    # Select a few test scenarios
    test_scenarios = [
        ("configs/scenario_s1_drift.yaml", "data/AIS_2020_01_01.csv"),
        ("configs/scenario_s2_zoneviolation.yaml", "data/AIS_2020_01_01.csv"),
    ]
    
    logger.info("Testing batch runner with 2 scenarios...")
    
    # Create runner with 2 workers
    runner = BatchRunner(max_workers=2, base_output_dir="output/batch_test")
    
    # Run batch
    summary = runner.run_batch(test_scenarios)
    
    # Check results
    if summary['successful'] == len(test_scenarios):
        logger.info("✅ All test scenarios completed successfully!")
        return True
    else:
        logger.error(f"❌ {summary['failed']} scenarios failed")
        return False


if __name__ == "__main__":
    success = test_batch_runner()
    sys.exit(0 if success else 1)