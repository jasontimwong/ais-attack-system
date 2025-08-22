# Impact Simulator

The Impact Simulator is a second-stage processing pipeline that demonstrates how vessels react to false target attacks. It takes attack.csv and diff.json as input and produces impact.csv and impact_events.json showing realistic vessel avoidance maneuvers.

## Overview

The Impact Simulator processes AIS attack data to show secondary effects on maritime traffic. When a false target attack is detected, nearby vessels may take evasive action, creating a cascading impact on normal shipping operations.

## Pipeline Components

### 1. TrafficSnapshotBuilder
- Converts attack.csv into time-stepped snapshots (configurable intervals)
- Interpolates vessel positions between AIS messages
- Creates systematic timeline for conflict detection

### 2. SituationEvaluator
- Detects collision risks using CPA (Closest Point of Approach) calculations
- Calculates TCPA (Time to Closest Point of Approach)
- Identifies critical situations requiring maneuvers

### 3. ManeuverSimulator
- Simulates vessel turning and speed changes
- Implements standard collision avoidance maneuvers
- Tracks maneuver execution over time

### 4. ImpactExporter
- Exports complete vessel trajectories with maneuvers applied
- Generates event logs of all triggered maneuvers
- Provides simulation summary with key metrics

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cpa_threshold_nm` | 1.0 | CPA distance threshold in nautical miles |
| `tcpa_threshold_seconds` | 900 | TCPA time threshold (15 minutes) |
| `time_step_seconds` | 10 | Time interval between snapshots |
| `max_turn_rate_deg_per_min` | 15.0 | Maximum vessel turn rate |
| `standard_turn_degrees` | 30.0 | Standard avoidance turn angle |

## Output Files

### impact.csv
Complete vessel trajectories with maneuvers applied, showing:
- Original vessel movements
- Maneuver execution (turning, speed changes)
- Post-maneuver trajectories
- Timeline of all vessel positions

### impact_events.json
Event log of all maneuvers triggered, including:
- Vessel MMSI performing maneuver
- Maneuver type (turn_starboard, turn_port, speed_change)
- Start and end times
- Course/speed changes
- Trigger conflict details

### simulation_summary.json
Summary statistics including:
- Total vessels analyzed
- Number of vessels with maneuvers
- Impact rate (percentage of vessels affected)
- Maneuver type distribution
- Time range and duration

## Usage

### Basic Usage
```python
from core.impact_simulator import (
    TrafficSnapshotBuilder, SituationEvaluator, 
    ManeuverSimulator, ImpactExporter
)

# Initialize components
builder = TrafficSnapshotBuilder(time_step_seconds=10)
evaluator = SituationEvaluator(cpa_threshold_nm=1.0)
simulator = ManeuverSimulator(standard_turn_degrees=30.0)
exporter = ImpactExporter(output_dir="results")

# Build snapshots
snapshots = builder.build_snapshots("attack.csv")

# Evaluate conflicts
conflicts = []
for timestamp, snapshot in snapshots.items():
    conflicts.extend(evaluator.evaluate_snapshot(snapshot))

# Simulate maneuvers
modified_snapshots = simulator.simulate_maneuvers(snapshots, conflicts)
events = simulator.get_all_maneuver_events()

# Export results
exporter.export_impact_data(modified_snapshots, events)
```

### Command Line
```bash
# Run complete pipeline
python generate_impact_simulation.py --input-dir smoke_out_v5 --output-dir smoke_out_v5

# Run demonstration
python impact_simulation_demo.py
```

## MVP Scope

The current implementation focuses on:
- Vessel with minimum CPA from target (identified in diff.json)
- Simple starboard turn avoidance maneuver (30° turn)
- Complete timeline showing before/during/after maneuver
- Realistic turn rates and execution times

## Technical Details

### CPA/TCPA Calculation
Uses vector-based approach to calculate closest point of approach:
- Converts lat/lon to Cartesian coordinates
- Calculates relative velocity vectors
- Determines minimum distance and time to reach it

### Maneuver Simulation
Implements realistic vessel dynamics:
- Maximum turn rate: 15°/minute
- Linear interpolation during maneuver execution
- Position updates based on new course/speed
- Maneuver duration based on turn angle

### Conflict Detection
Identifies situations requiring immediate action:
- CPA distance < threshold (default 1.0 nm)
- TCPA within time window (default 15 minutes)
- Vessels on converging courses

## Future Enhancements

- Multiple maneuver types (port turns, speed changes)
- Rules-of-the-road implementation
- Multiple vessel conflicts
- Weather and environmental factors
- Integration with maritime traffic simulation