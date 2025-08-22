# 🤝 Contributing to AIS Attack System

Thank you for your interest in contributing to the AIS Attack Generation System! This project aims to advance maritime cybersecurity research, and we welcome contributions from both academia and industry.

## 📋 Table of Contents

- [Types of Contributions](#types-of-contributions)
- [Development Environment Setup](#development-environment-setup)
- [Code Contribution Process](#code-contribution-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Contributions](#documentation-contributions)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## 🎯 Types of Contributions

We welcome the following types of contributions:

### 🔧 Code Contributions
- **New Attack Type Implementations** - Add new AIS attack patterns
- **Algorithm Improvements** - Optimize existing target selection, physics engine, etc.
- **Performance Optimizations** - Improve system processing speed and memory efficiency
- **Visualization Enhancements** - Improve ECDIS rendering and Web interface
- **Integration Features** - Add integration with other maritime systems

### 📊 Data Contributions
- **New Test Datasets** - Provide real AIS data for validation
- **Attack Scenario Configurations** - Create new attack scenario configurations
- **Benchmark Test Data** - Provide performance benchmark test data

### 📖 Documentation Contributions
- **API Documentation** - Improve code documentation and API descriptions
- **Tutorials and Guides** - Write usage tutorials and best practices
- **Academic Papers** - Research results based on the system
- **Translations** - Translate documentation to other languages

### 🐛 Issue Reporting
- **Bug Reports** - Discover and report system defects
- **Performance Issues** - Report performance bottlenecks
- **Compatibility Issues** - Report platform compatibility problems

## 🛠️ Development Environment Setup

### 1. Fork Repository

Click the "Fork" button in the top right corner of the GitHub page to create your branch.

### 2. Clone Code

```bash
git clone https://github.com/YOUR_USERNAME/ais-attack-system.git
cd ais-attack-system
git remote add upstream https://github.com/jasontimwong/ais-attack-system.git
```

### 3. Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### 4. Verify Environment

```bash
# Run test suite
pytest

# Run code checks
flake8 core/ attacks/ visualization/
black --check core/ attacks/ visualization/
mypy core/

# Run system check
python tools/system_check.py
```

## 🔄 Code Contribution Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

Branch naming conventions:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `perf/` - Performance optimizations

### 2. Development and Testing

```bash
# Perform development...

# Run tests
pytest tests/

# Run specific tests
pytest tests/test_attack_orchestrator.py -v

# Check code coverage
pytest --cov=core --cov=attacks --cov-report=html
```

### 3. Commit Code

```bash
# Add files
git add .

# Commit (follow commit message conventions)
git commit -m "feat: add new ghost swarm attack pattern

- Implement coordinated 8-vessel attack formation
- Add V-formation and diamond-formation patterns
- Include collision avoidance between ghost vessels
- Add comprehensive test coverage

Closes #123"
```

#### Commit Message Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types include:
- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation updates
- `style` - Code formatting (no functional impact)
- `refactor` - Code refactoring
- `perf` - Performance optimizations
- `test` - Test related
- `chore` - Build tools, dependencies, etc.

### 4. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Create a Pull Request on GitHub, including:
- **Clear title and description**
- **Links to related issues**
- **Detailed explanation of changes**
- **Test result screenshots** (if applicable)
- **Breaking changes description** (if any)

## 📝 Coding Standards

### Python Code Standards

We use the following tools to ensure code quality:

```bash
# Code formatting
black core/ attacks/ visualization/

# Import sorting
isort core/ attacks/ visualization/

# Code checking
flake8 core/ attacks/ visualization/

# Type checking
mypy core/
```

#### Code Style Requirements

```python
"""
Module-level docstring

Detailed description of module functionality and purpose.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.base import BaseAttack


class FlashCrossAttack(BaseAttack):
    """
    Flash Cross Attack implementation.
    
    This class implements the S1 Flash Cross attack pattern with
    4-stage progressive attack orchestration.
    
    Args:
        config: Attack configuration parameters
        
    Attributes:
        attack_id: Unique attack identifier
        stages: List of attack stages
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.attack_id: Optional[str] = None
        self.stages: List[AttackStage] = []
    
    def execute_attack(self, target_data: Dict) -> Dict:
        """
        Execute the Flash Cross attack.
        
        Args:
            target_data: Target vessel information
            
        Returns:
            Attack execution results
            
        Raises:
            ValueError: If target data is invalid
            RuntimeError: If attack execution fails
        """
        if not self._validate_target_data(target_data):
            raise ValueError("Invalid target data provided")
        
        try:
            # Implementation...
            pass
        except Exception as e:
            raise RuntimeError(f"Attack execution failed: {e}") from e
```

### Docstring Standards

Use Google-style docstrings:

```python
def calculate_cpa(vessel_a: Dict, vessel_b: Dict) -> float:
    """
    Calculate Closest Point of Approach between two vessels.
    
    This function computes the CPA using standard maritime navigation
    formulas, accounting for vessel speeds, courses, and positions.
    
    Args:
        vessel_a: First vessel data containing lat, lon, speed, course
        vessel_b: Second vessel data containing lat, lon, speed, course
        
    Returns:
        CPA distance in nautical miles
        
    Raises:
        ValueError: If vessel data is incomplete or invalid
        
    Example:
        >>> vessel_1 = {'lat': 40.7128, 'lon': -74.0060, 'speed': 12, 'course': 90}
        >>> vessel_2 = {'lat': 40.7200, 'lon': -74.0000, 'speed': 10, 'course': 180}
        >>> cpa = calculate_cpa(vessel_1, vessel_2)
        >>> print(f"CPA: {cpa:.2f} nm")
        CPA: 0.45 nm
    """
```

## 🧪 Testing Requirements

### Test Coverage Requirements

- **Core modules**: Minimum 90% coverage
- **Attack modules**: Minimum 85% coverage
- **Visualization modules**: Minimum 75% coverage
- **Tool modules**: Minimum 70% coverage

### Test Types

#### 1. Unit Tests

```python
# tests/test_attack_orchestrator.py
import pytest
from unittest.mock import Mock, patch

from core.attack_orchestrator import AttackOrchestrator
from core.target_selector import TargetSelector


class TestAttackOrchestrator:
    """Test cases for AttackOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        target_selector = Mock(spec=TargetSelector)
        physics_engine = Mock()
        colregs_validator = Mock()
        
        return AttackOrchestrator(
            target_selector, physics_engine, colregs_validator
        )
    
    def test_select_target_success(self, orchestrator):
        """Test successful target selection."""
        vessels = [
            {'mmsi': '123456789', 'lat': 40.7128, 'lon': -74.0060}
        ]
        constraints = {'min_distance': 1.0}
        
        orchestrator.target_selector.select_optimal_target.return_value = {
            'mmsi': '123456789',
            'vulnerability': 0.85
        }
        
        result = orchestrator.select_target(vessels, constraints)
        
        assert result == '123456789'
        orchestrator.target_selector.select_optimal_target.assert_called_once_with(
            vessels, constraints
        )
```

#### 2. Integration Tests

```python
# tests/integration/test_flash_cross_integration.py
import pytest
from pathlib import Path

from attacks.flash_cross import FlashCrossAttack


class TestFlashCrossIntegration:
    """Integration tests for Flash Cross attack."""
    
    def test_complete_attack_execution(self):
        """Test complete attack execution pipeline."""
        # Load test data
        test_data = self._load_test_data()
        
        # Initialize attack
        attack = FlashCrossAttack()
        attack_id = attack.initialize_attack(
            test_data['target'], test_data['start_position']
        )
        
        # Execute all stages
        results = attack.execute_complete_attack(test_data['trajectory'])
        
        # Validate results
        assert results['success'] is True
        assert len(results['stages']) == 4
        assert results['metrics']['attack_effectiveness'] > 0.8
```

#### 3. Performance Tests

```python
# tests/performance/test_batch_performance.py
import time
import pytest

from tools.batch_runner import BatchRunner


class TestBatchPerformance:
    """Performance tests for batch processing."""
    
    @pytest.mark.performance
    def test_batch_execution_performance(self):
        """Test batch execution meets performance requirements."""
        runner = BatchRunner()
        
        # Load test scenarios
        test_scenarios = self._create_test_scenarios(count=10)
        runner.scenarios = test_scenarios
        
        # Measure execution time
        start_time = time.time()
        results = runner.run_all_scenarios(parallel=True)
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert execution_time < 60.0  # Should complete in under 1 minute
        assert results['summary']['success_rate'] >= 0.9
        
        # Memory usage should be reasonable
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 1000  # Should use less than 1GB
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/
pytest tests/integration/
pytest -m performance

# Generate coverage report
pytest --cov=core --cov=attacks --cov-report=html
open htmlcov/index.html

# Run performance tests
pytest -m performance --benchmark-only
```

## 📚 Documentation Contributions

### Documentation Structure

```
docs/
├── api/                 # API reference documentation
├── tutorials/           # Tutorials and guides
├── algorithms/          # Algorithm detailed descriptions
├── examples/           # Example code
└── images/             # Documentation images
```

### Documentation Writing Standards

1. **Use Markdown format**
2. **Include code examples**
3. **Add appropriate images and diagrams**
4. **Keep content updated**

### Building Documentation

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build HTML documentation
cd docs/
make html

# Start local documentation server
python -m http.server 8000 -d _build/html/
```

## 🐛 Issue Reporting

### Bug Report Template

Use GitHub Issues template to report problems:

```markdown
**Bug Description**
A clear and concise description of the bug.

**Steps to Reproduce**
1. Run command '...'
2. Click button '....'
3. See error '....'

**Expected Behavior**
Describe what you expected to happen.

**Actual Behavior**
Describe what actually happened.

**Environment Information**
- OS: [e.g. macOS 12.6]
- Python Version: [e.g. 3.9.7]
- System Version: [e.g. 1.0.0]

**Additional Information**
- Error logs
- Configuration files
- Test data
```

### Feature Request Template

```markdown
**Feature Description**
A clear and concise description of the feature you want.

**Problem Background**
Describe the problem this feature would solve.

**Suggested Solution**
Describe your proposed solution.

**Alternative Solutions**
Describe other solutions you've considered.

**Additional Information**
Add any other relevant information or screenshots.
```

## 🏆 Contributor Recognition

We value every contribution and recognize contributors through:

### Contributor Types

- **Core Maintainers** - Long-term project maintainers
- **Code Contributors** - Developers who submit code
- **Documentation Contributors** - Contributors who improve documentation
- **Test Contributors** - Contributors who provide testing and QA
- **Community Supporters** - Contributors who help other users

### Recognition Methods

1. **README contributor list**
2. **Release notes acknowledgments**
3. **GitHub contributor badges**
4. **Academic paper acknowledgments**

## 📜 Community Guidelines

### Code of Conduct

We adopt the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct:

- **Friendly and Inclusive** - Welcome contributors from all backgrounds
- **Respect Others** - Respect different viewpoints and experiences
- **Constructive Feedback** - Provide helpful, constructive feedback
- **Professional Attitude** - Maintain professional and courteous communication

### Academic Integrity

Since this project involves cybersecurity research:

- **Responsible Disclosure** - Responsibly disclose discovered security issues
- **Research Use** - Ensure research results are used for defensive purposes
- **Citation Standards** - Properly cite relevant research and data sources
- **Ethical Considerations** - Follow research ethics and legal regulations

## 📧 Contact

If you have any questions or need help:

- **GitHub Issues**: [Report Issues](https://github.com/jasontimwong/ais-attack-system/issues)
- **GitHub Discussions**: [Join Discussions](https://github.com/jasontimwong/ais-attack-system/discussions)
- **Email**: jason@example.com

---

Thank you for contributing to the AIS Attack Generation System! Your participation will advance maritime cybersecurity research. 🚢⚓