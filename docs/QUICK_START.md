# 🚀 Quick Start Guide

Welcome to the AIS Attack Generation System! This guide will help you run your first attack scenario in just a few minutes.

## 📋 System Requirements

### Basic Requirements
- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM
- **Storage**: At least 2GB available space
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)

### Optional Requirements
- **Node.js**: 16+ (for Web visualization interface)
- **OpenCPN**: 5.6+ (for professional chart display)
- **Bridge Command**: 5.0+ (for bridge simulator integration)

## ⚡ 5-Minute Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/jasontimwong/ais-attack-system.git
cd ais-attack-system
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Or using conda
conda create -n ais-attack python=3.8
conda activate ais-attack
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python tools/system_check.py
```

If you see ✅ all checks passed, the installation was successful!

## 🎯 Run Your First Attack Scenario

### Method 1: Using Command Line Interface

```bash
# Generate S1 Flash Cross attack scenario
python -m core.attack_orchestrator --scenario s1_flash_cross --output output/my_first_attack

# View generated files
ls output/my_first_attack/
```

### Method 2: Using Python API

Create file `my_first_attack.py`:

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

# Create attack instance
attack = FlashCrossAttack()

# Simulate target vessel data
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

Run the script:
```bash
python my_first_attack.py
```

## 🎮 Start Web Visualization Interface

```bash
# Enter visualization directory
cd visualization/web_interface

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Open browser and visit `http://localhost:5173` to view the interactive map.

## 📊 Generate Batch Scenarios

Run all 35 predefined attack scenarios:

```bash
# Sequential execution
python tools/batch_runner/run_all_scenarios.py

# Parallel execution (recommended)
python tools/batch_runner/run_all_scenarios.py --parallel --workers 4

# Run specific scenarios only
python tools/batch_runner/run_all_scenarios.py --scenarios s1_flash_cross s3_ghost_swarm
```

## 🗂️ Output File Description

After executing attacks, the system generates files in the `output/` directory:

```
output/
├── s1_flash_cross_20241201_143052/
│   ├── attack_trajectory.geojson    # Attack trajectory (GeoJSON format)
│   ├── baseline_trajectory.geojson  # Baseline trajectory
│   ├── attack_labels.json           # Auto-generated labels
│   ├── metadata.json               # Attack metadata
│   ├── metrics_report.json         # Performance metrics
│   └── visualization.html          # Visualization report
```

### File Format Description

**GeoJSON Trajectory Files**:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-74.0060, 40.7128]
      },
      "properties": {
        "timestamp": "2024-12-01T14:30:52Z",
        "mmsi": "123456789",
        "speed": 12.0,
        "course": 90.0,
        "attack_stage": "parallel_following"
      }
    }
  ]
}
```

## 🔍 Validate Attack Quality

```bash
# Validate single scenario
python tools/validation/validate_scenario.py output/s1_flash_cross_20241201_143052/

# Generate quality report
python tools/validation/generate_quality_report.py --input output/ --output quality_report.html
```

Quality metrics include:
- ✅ **Physical Consistency**: 98.7% (trajectories comply with ship dynamics)
- ✅ **COLREGs Compliance**: 97.9% (compliant with collision avoidance rules)
- ✅ **Attack Success Rate**: 94.3% (successfully trigger target response)

## 🎨 Generate Professional Visualizations

### ECDIS Chart Display
```bash
python visualization/ecdis_renderer/create_ecdis_report.py \
    --scenario s1_flash_cross \
    --output ecdis_report.png \
    --style imo_standard
```

### Interactive HTML Report
```bash
python visualization/create_interactive_report.py \
    --input output/s1_flash_cross_20241201_143052/ \
    --output interactive_report.html
```

## 🛠️ Customize Attack Parameters

Edit configuration file `configs/custom_attack.yaml`:

```yaml
# Custom Flash Cross attack parameters
attacks:
  s1_flash_cross:
    stages:
      parallel_following:
        duration: 180.0      # Extend to 3 minutes
        parallel_distance: 1.5  # Reduce to 1.5 nautical miles
      
      flash_cross_maneuver:
        cross_angle: 120.0   # Larger crossing angle
        speed_factor: 2.0    # Higher speed multiplier
```

Use custom configuration:
```bash
python -m core.attack_orchestrator --config configs/custom_attack.yaml --scenario s1_flash_cross
```

## 📱 Integration with Existing Systems

### Bridge Command Integration
```bash
# Export to Bridge Command format
python tools/export/export_to_bridge_command.py \
    --input output/s1_flash_cross_20241201_143052/ \
    --output bridge_command_scenario/
```

### OpenCPN Plugin
```bash
# Build OpenCPN plugin
cd plugins/opencpn_ais_attack/
mkdir build && cd build
cmake .. && make
```

### NMEA Output
```bash
# Generate real-time NMEA data stream
python tools/export/generate_nmea_stream.py \
    --input output/s1_flash_cross_20241201_143052/ \
    --port 4001 \
    --realtime
```

## 🚨 Common Issues

### Q: Missing dependencies during installation?
A: Ensure you're using Python 3.8+ and install all dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Q: Attack scenario generation fails?
A: Check the log file `logs/attack_generation.log`:
```bash
tail -f logs/attack_generation.log
```

### Q: Web interface won't load?
A: Ensure Node.js version ≥16, reinstall dependencies:
```bash
cd visualization/web_interface
rm -rf node_modules package-lock.json
npm install
```

### Q: Poor visualization image quality?
A: Adjust DPI settings:
```bash
python visualization/ecdis_renderer/create_ecdis_report.py \
    --dpi 300 \
    --format png \
    --size 1920x1080
```

## 📚 Next Steps

Congratulations! You've successfully run your first AIS attack scenario. Next, you can:

1. **Explore More Attack Types** - Check out [Attack Types Documentation](ATTACK_TYPES.md)
2. **Customize Attack Parameters** - Read [Configuration Guide](CONFIGURATION.md)
3. **Integrate with Your Systems** - Refer to [API Documentation](API_REFERENCE.md)
4. **Contribute Code** - See [Development Guide](CONTRIBUTING.md)

## 🆘 Get Help

- 📖 **Documentation**: [Complete Documentation](https://github.com/jasontimwong/ais-attack-system/docs)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/jasontimwong/ais-attack-system/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/jasontimwong/ais-attack-system/discussions)

---

**🎉 Welcome to the AIS Attack Generation System community!**