# Contributing Guide

This document provides detailed guidelines for contributing to the AIS Attack Generation System project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Contribution Process](#code-contribution-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Guidelines](#documentation-guidelines)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8 or higher
- Git version control
- Basic understanding of maritime navigation and AIS systems
- Familiarity with maritime cybersecurity concepts

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ais-attack-system.git
   cd ais-attack-system
   ```
3. **Set up the development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .[dev]
   ```
4. **Run system checks**:
   ```bash
   python tools/system_check.py
   ```

## Development Environment

### Required Tools

- **Python 3.8+**: Primary development language
- **Git**: Version control
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

### Optional Tools

- **Node.js 16+**: For web interface development
- **Docker**: For containerized development
- **OpenCPN**: For ECDIS testing
- **Bridge Command**: For simulator integration testing

### Environment Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Verify installation
python tools/system_check.py --verbose
```

## Code Contribution Process

### 1. Choose an Issue

- Browse [open issues](https://github.com/jasontimwong/ais-attack-system/issues)
- Look for issues labeled `good-first-issue` for beginners
- Comment on the issue to indicate you're working on it

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Your Changes

- Write clean, well-documented code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python tools/batch_runner/test_all_scenarios.py

# Run data quality checks
python tools/validation/check_data_quality.py

# Check code formatting
black --check .
flake8 .
mypy core/ attacks/ tools/
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new attack type implementation

- Implement S10 advanced spoofing attack
- Add comprehensive test coverage
- Update documentation and examples
- Ensure COLREGs compliance validation"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots/videos if applicable
- Test results and performance metrics

## Coding Standards

### Python Code Style

We follow PEP 8 with some specific guidelines:

```python
# Use type hints for all functions
def calculate_cpa(vessel_a: VesselData, vessel_b: VesselData) -> float:
    """
    Calculate Closest Point of Approach between two vessels.
    
    Args:
        vessel_a: First vessel data
        vessel_b: Second vessel data
        
    Returns:
        CPA distance in nautical miles
    """
    # Implementation here
    pass

# Use dataclasses for structured data
@dataclass
class AttackResult:
    """Result of an attack execution"""
    attack_id: str
    success: bool
    metrics: Dict[str, float]
    timestamp: datetime
```

### Naming Conventions

- **Classes**: PascalCase (`AttackOrchestrator`)
- **Functions/Variables**: snake_case (`calculate_cpa`)
- **Constants**: UPPER_CASE (`MAX_TURN_RATE`)
- **Files**: snake_case (`attack_orchestrator.py`)
- **Modules**: lowercase (`core`, `attacks`)

### Documentation Standards

- All public functions must have docstrings
- Use Google-style docstrings
- Include type hints for all parameters and return values
- Add inline comments for complex algorithms

### Error Handling

```python
# Use specific exception types
class PhysicsValidationError(Exception):
    """Raised when physics constraints are violated"""
    pass

# Provide informative error messages
try:
    validate_trajectory(trajectory)
except PhysicsValidationError as e:
    logger.error(f"Physics validation failed: {e}")
    raise
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **System Tests**: Test complete attack scenarios
4. **Performance Tests**: Validate performance requirements

### Writing Tests

```python
import pytest
from core.attack_orchestrator import AttackOrchestrator

class TestAttackOrchestrator:
    """Test suite for AttackOrchestrator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.orchestrator = AttackOrchestrator()
    
    def test_target_selection(self):
        """Test target selection functionality"""
        vessels = [self.create_test_vessel()]
        target = self.orchestrator.select_target(vessels)
        assert target is not None
        assert target.mmsi in [v.mmsi for v in vessels]
    
    @pytest.mark.slow
    def test_full_attack_execution(self):
        """Test complete attack execution (slow test)"""
        # Implementation here
        pass
```

### Test Data

- Use realistic but anonymized AIS data
- Create reproducible test scenarios
- Include edge cases and error conditions
- Test with various vessel types and geographic locations

### Performance Benchmarks

All contributions must maintain performance standards:

- Attack generation: <5 seconds per scenario
- Batch processing: >100 scenarios/hour
- Memory usage: <1GB for typical workloads
- Physics validation: 98%+ accuracy

## Documentation Guidelines

### Code Documentation

- Document all public APIs
- Include usage examples
- Explain complex algorithms
- Document configuration options

### User Documentation

- Update README.md for new features
- Add entries to API reference
- Create tutorials for complex features
- Update deployment guides

### Technical Documentation

- Document architectural decisions
- Explain algorithm implementations
- Include performance characteristics
- Add troubleshooting guides

## Issue Reporting

### Bug Reports

Use the bug report template and include:

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- System version: [e.g., v1.2.0]

**Additional Context**
Any other relevant information
```

### Feature Requests

Use the feature request template and include:

- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Potential impact on existing functionality

### Security Issues

For security-related issues:

1. **DO NOT** create a public issue
2. Email security concerns privately
3. Allow time for security review
4. Follow responsible disclosure practices

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- Be respectful and professional
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Request Reviews**: Code-specific discussions

### Review Process

All contributions go through peer review:

1. **Automated Checks**: CI/CD pipeline validation
2. **Code Review**: Manual review by maintainers
3. **Testing**: Comprehensive test validation
4. **Documentation**: Documentation completeness check

### Recognition

Contributors are recognized through:

- Contribution acknowledgments in release notes
- Contributor listings in project documentation
- GitHub contributor statistics
- Special recognition for significant contributions

## Development Workflow

### Branch Strategy

- `main`: Stable release branch
- `develop`: Development integration branch
- `feature/*`: Feature development branches
- `fix/*`: Bug fix branches
- `release/*`: Release preparation branches

### Release Process

1. **Feature Freeze**: Stop adding new features
2. **Testing Phase**: Comprehensive testing
3. **Documentation Update**: Ensure docs are current
4. **Release Candidate**: Create RC for testing
5. **Final Release**: Tag and publish release

### Continuous Integration

Our CI/CD pipeline includes:

- Automated testing on multiple Python versions
- Code quality checks (linting, formatting)
- Security vulnerability scanning
- Performance regression testing
- Documentation building and validation

## Getting Help

### Resources

- **Documentation**: Check existing documentation first
- **Examples**: Look at example implementations
- **Tests**: Review test cases for usage patterns
- **Issues**: Search existing issues for similar problems

### Asking Questions

When asking for help:

1. Search existing documentation and issues
2. Provide complete context and error messages
3. Include minimal reproducible examples
4. Be specific about your environment and setup

### Mentorship

New contributors can request mentorship:

- Pair programming sessions
- Code review guidance
- Architecture discussions
- Career development advice

---

Thank you for contributing to the AIS Attack Generation System! Your contributions help advance maritime cybersecurity research and defense capabilities.
