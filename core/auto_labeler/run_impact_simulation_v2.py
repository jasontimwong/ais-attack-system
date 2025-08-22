#!/usr/bin/env python3
"""
Run impact simulation with ghost vessel detection and attack effectiveness evaluation.

This enhanced version integrates:
1. GhostWatcher for detecting ghost vessels and threatened real vessels
2. Updated ManeuverSimulator that prioritizes ghost-triggered maneuvers
3. Attack effectiveness evaluation that establishes causal links
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.impact_simulator.traffic_snapshot_builder import TrafficSnapshotBuilder
from core.impact_simulator.situation_evaluator import SituationEvaluator
from core.impact_simulator.maneuver_simulator import ManeuverSimulator
from core.impact_simulator.impact_exporter import ImpactExporter
from core.impact_simulator.ghost_watcher import GhostWatcher
from core.metrics.effectiveness import AttackEffectivenessEvaluator
from core.data_loader_v2 import StreamingDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_impact_simulation_with_effectiveness(attack_csv_path: str, output_dir: str, 
                                           baseline_csv_path: str = None):
    """
    Run enhanced impact simulation with ghost detection and effectiveness evaluation.
    
    Args:
        attack_csv_path: Path to attack data CSV
        output_dir: Directory for output files
        baseline_csv_path: Path to baseline data CSV (optional, will extract from output_dir if not provided)
    """
    logger.info(f"Starting enhanced impact simulation")
    logger.info(f"Attack data: {attack_csv_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load diff.json for attack metadata
    diff_path = output_path / "diff.json"
    if not diff_path.exists():
        logger.error(f"diff.json not found in {output_dir}")
        return
    
    with open(diff_path) as f:
        diff_data = json.load(f)
    
    # Determine baseline path
    if baseline_csv_path is None:
        baseline_csv_path = str(output_path / "baseline.csv")
    
    if not Path(baseline_csv_path).exists():
        logger.error(f"Baseline data not found at {baseline_csv_path}")
        return
    
    # Load attack data
    logger.info("Loading attack data...")
    loader = StreamingDataLoader()
    attack_chunks = list(loader.load_csv_stream(attack_csv_path))
    attack_data = pd.concat(attack_chunks) if attack_chunks else pd.DataFrame()
    logger.info(f"Loaded {len(attack_data)} attack records")
    
    # Load baseline data
    logger.info("Loading baseline data...")
    baseline_chunks = list(loader.load_csv_stream(baseline_csv_path))
    baseline_data = pd.concat(baseline_chunks) if baseline_chunks else pd.DataFrame()
    logger.info(f"Loaded {len(baseline_data)} baseline records")
    
    # Phase 1: Build traffic snapshots with time stepping
    logger.info("\n=== Phase 1: Building Traffic Snapshots ===")
    builder = TrafficSnapshotBuilder(time_step_seconds=10)
    snapshots = builder.build_snapshots_from_dataframe(attack_data)
    logger.info(f"Built {len(snapshots)} snapshots")
    
    # Phase 2: Initialize ghost watcher and detect ghost vessels
    logger.info("\n=== Phase 2: Detecting Ghost Vessels ===")
    ghost_watcher = GhostWatcher(cpa_threshold_nm=1.0, tcpa_threshold_min=15.0)
    
    # Process all snapshots to find ghost threats
    all_ghost_threats = {}
    for timestamp, snapshot in snapshots.items():
        threats = ghost_watcher.process_snapshot(snapshot, diff_data)
        for vessel_mmsi, vessel_threats in threats.items():
            if vessel_mmsi not in all_ghost_threats:
                all_ghost_threats[vessel_mmsi] = []
            all_ghost_threats[vessel_mmsi].extend(vessel_threats)
    
    # Log ghost threat summary
    threat_summary = ghost_watcher.get_threat_summary()
    logger.info(f"Ghost threat summary: {json.dumps(threat_summary, indent=2)}")
    
    # Phase 3: Evaluate situations (vessel-to-vessel conflicts)
    logger.info("\n=== Phase 3: Evaluating Collision Risks ===")
    evaluator = SituationEvaluator(cpa_threshold_nm=1.0, tcpa_threshold_seconds=900)
    
    all_conflicts = []
    for timestamp, snapshot in snapshots.items():
        conflicts = evaluator.evaluate_snapshot(snapshot)
        all_conflicts.extend(conflicts)
    
    logger.info(f"Found {len(all_conflicts)} vessel-to-vessel conflicts")
    
    # Phase 4: Simulate maneuvers with ghost threat priority
    logger.info("\n=== Phase 4: Simulating Avoidance Maneuvers ===")
    simulator = ManeuverSimulator()
    modified_snapshots = simulator.simulate_maneuvers(
        snapshots, all_conflicts, ghost_threats=all_ghost_threats
    )
    
    maneuver_events = simulator.get_all_maneuver_events()
    logger.info(f"Generated {len(maneuver_events)} maneuver events")
    
    # Log ghost-triggered maneuvers
    ghost_maneuvers = [m for m in maneuver_events if m.trigger_type == 'ghost_threat']
    logger.info(f"Ghost-triggered maneuvers: {len(ghost_maneuvers)}")
    
    # Phase 5: Export impact data
    logger.info("\n=== Phase 5: Exporting Impact Data ===")
    exporter = ImpactExporter()
    
    # Export modified vessel trajectories
    impact_csv_path = output_path / "impact.csv"
    exporter.export_impact_data(modified_snapshots, str(impact_csv_path))
    
    # Export maneuver events
    events_json_path = output_path / "impact_events.json"
    exporter.export_maneuver_events(maneuver_events, str(events_json_path))
    
    # Phase 6: Evaluate attack effectiveness
    logger.info("\n=== Phase 6: Evaluating Attack Effectiveness ===")
    
    # Load the impact events we just created
    with open(events_json_path) as f:
        impact_events = json.load(f)
    
    # Run effectiveness evaluation
    effectiveness_evaluator = AttackEffectivenessEvaluator()
    effectiveness_results = effectiveness_evaluator.evaluate_attack(
        diff_data, impact_events, attack_data, baseline_data
    )
    
    # Save effectiveness results
    effectiveness_path = output_path / "attack_effectiveness.json"
    effectiveness_evaluator.save_results(effectiveness_results, str(effectiveness_path))
    
    # Create summary report
    summary = {
        "simulation_completed": datetime.now().isoformat(),
        "total_snapshots": len(snapshots),
        "ghost_vessels_detected": len(ghost_watcher.known_ghosts),
        "vessels_threatened_by_ghosts": len(all_ghost_threats),
        "vessel_to_vessel_conflicts": len(all_conflicts),
        "total_maneuvers": len(maneuver_events),
        "ghost_triggered_maneuvers": len(ghost_maneuvers),
        "attack_effectiveness": {
            "total_attacks": len(effectiveness_results.get('attacks', [])),
            "effective_attacks": sum(1 for a in effectiveness_results.get('attacks', []) 
                                   if a.get('overall_effective', False))
        }
    }
    
    summary_path = output_path / "simulation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSimulation complete! Results saved to {output_dir}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_impact_simulation_v2.py <attack_csv> <output_dir> [baseline_csv]")
        sys.exit(1)
    
    attack_csv = sys.argv[1]
    output_dir = sys.argv[2]
    baseline_csv = sys.argv[3] if len(sys.argv) > 3 else None
    
    run_impact_simulation_with_effectiveness(attack_csv, output_dir, baseline_csv)