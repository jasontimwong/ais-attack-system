# 🤝 Contributing to AIS Attack System

感谢您对AIS攻击生成系统的贡献兴趣！本项目旨在推进海事网络安全研究，我们欢迎来自学术界和工业界的贡献。

## 📋 目录

- [贡献类型](#贡献类型)
- [开发环境设置](#开发环境设置)
- [代码贡献流程](#代码贡献流程)
- [编码规范](#编码规范)
- [测试要求](#测试要求)
- [文档贡献](#文档贡献)
- [问题报告](#问题报告)
- [社区准则](#社区准则)

## 🎯 贡献类型

我们欢迎以下类型的贡献：

### 🔧 代码贡献
- **新攻击类型实现** - 添加新的AIS攻击模式
- **算法改进** - 优化现有的目标选择、物理引擎等算法
- **性能优化** - 提升系统处理速度和内存效率
- **可视化增强** - 改进ECDIS渲染和Web界面
- **集成功能** - 添加与其他海事系统的集成

### 📊 数据贡献
- **新测试数据集** - 提供真实AIS数据用于验证
- **攻击场景配置** - 创建新的攻击场景配置
- **基准测试数据** - 提供性能基准测试数据

### 📖 文档贡献
- **API文档** - 改进代码文档和API说明
- **教程和指南** - 编写使用教程和最佳实践
- **学术论文** - 基于系统的研究成果
- **翻译** - 将文档翻译为其他语言

### 🐛 问题报告
- **Bug报告** - 发现和报告系统缺陷
- **性能问题** - 报告性能瓶颈
- **兼容性问题** - 报告平台兼容性问题

## 🛠️ 开发环境设置

### 1. Fork 仓库

点击GitHub页面右上角的"Fork"按钮创建您的分支。

### 2. 克隆代码

```bash
git clone https://github.com/YOUR_USERNAME/ais-attack-system.git
cd ais-attack-system
git remote add upstream https://github.com/jasontimwong/ais-attack-system.git
```

### 3. 设置开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -e .[dev]

# 安装pre-commit hooks
pre-commit install
```

### 4. 验证环境

```bash
# 运行测试套件
pytest

# 运行代码检查
flake8 core/ attacks/ visualization/
black --check core/ attacks/ visualization/
mypy core/

# 运行系统检查
python tools/system_check.py
```

## 🔄 代码贡献流程

### 1. 创建功能分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b bugfix/issue-number-description
```

分支命名规范：
- `feature/` - 新功能
- `bugfix/` - Bug修复
- `docs/` - 文档更新
- `refactor/` - 代码重构
- `perf/` - 性能优化

### 2. 开发和测试

```bash
# 进行开发...

# 运行测试
pytest tests/

# 运行特定测试
pytest tests/test_attack_orchestrator.py -v

# 检查代码覆盖率
pytest --cov=core --cov=attacks --cov-report=html
```

### 3. 提交代码

```bash
# 添加文件
git add .

# 提交（遵循commit message规范）
git commit -m "feat: add new ghost swarm attack pattern

- Implement coordinated 8-vessel attack formation
- Add V-formation and diamond-formation patterns
- Include collision avoidance between ghost vessels
- Add comprehensive test coverage

Closes #123"
```

#### Commit Message 规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型包括：
- `feat` - 新功能
- `fix` - Bug修复
- `docs` - 文档更新
- `style` - 代码格式（不影响功能）
- `refactor` - 代码重构
- `perf` - 性能优化
- `test` - 测试相关
- `chore` - 构建工具、依赖等

### 4. 推送和创建Pull Request

```bash
# 推送到您的fork
git push origin feature/your-feature-name
```

在GitHub上创建Pull Request，包含：
- **清晰的标题和描述**
- **相关issue的链接**
- **变更内容的详细说明**
- **测试结果截图**（如适用）
- **Breaking changes说明**（如有）

## 📝 编码规范

### Python代码规范

我们使用以下工具确保代码质量：

```bash
# 代码格式化
black core/ attacks/ visualization/

# 导入排序
isort core/ attacks/ visualization/

# 代码检查
flake8 core/ attacks/ visualization/

# 类型检查
mypy core/
```

#### 代码风格要求

```python
"""
模块级文档字符串

详细描述模块的功能和用途。
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

### 文档字符串规范

使用Google风格的文档字符串：

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

## 🧪 测试要求

### 测试覆盖率要求

- **核心模块**: 最低90%覆盖率
- **攻击模块**: 最低85%覆盖率
- **可视化模块**: 最低75%覆盖率
- **工具模块**: 最低70%覆盖率

### 测试类型

#### 1. 单元测试

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

#### 2. 集成测试

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

#### 3. 性能测试

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

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定类型的测试
pytest tests/unit/
pytest tests/integration/
pytest -m performance

# 生成覆盖率报告
pytest --cov=core --cov=attacks --cov-report=html
open htmlcov/index.html

# 运行性能测试
pytest -m performance --benchmark-only
```

## 📚 文档贡献

### 文档结构

```
docs/
├── api/                 # API参考文档
├── tutorials/           # 教程和指南
├── algorithms/          # 算法详细说明
├── examples/           # 示例代码
└── images/             # 文档图片
```

### 文档编写规范

1. **使用Markdown格式**
2. **包含代码示例**
3. **添加适当的图片和图表**
4. **保持内容更新**

### 构建文档

```bash
# 安装文档依赖
pip install -e .[docs]

# 构建HTML文档
cd docs/
make html

# 启动本地文档服务器
python -m http.server 8000 -d _build/html/
```

## 🐛 问题报告

### Bug报告模板

使用GitHub Issues模板报告问题：

```markdown
**Bug描述**
简洁清晰地描述bug。

**复现步骤**
1. 运行命令 '...'
2. 点击按钮 '....'
3. 查看错误 '....'

**预期行为**
描述您期望发生的行为。

**实际行为**
描述实际发生的行为。

**环境信息**
- OS: [e.g. macOS 12.6]
- Python版本: [e.g. 3.9.7]
- 系统版本: [e.g. 1.0.0]

**附加信息**
- 错误日志
- 配置文件
- 测试数据
```

### 功能请求模板

```markdown
**功能描述**
简洁清晰地描述您希望的功能。

**问题背景**
描述这个功能要解决的问题。

**建议的解决方案**
描述您认为可行的解决方案。

**替代方案**
描述您考虑过的其他解决方案。

**附加信息**
添加任何其他相关信息或截图。
```

## 🏆 贡献者认可

我们重视每一个贡献，并通过以下方式认可贡献者：

### 贡献者类型

- **核心维护者** - 长期维护项目的开发者
- **代码贡献者** - 提交代码的开发者
- **文档贡献者** - 改进文档的贡献者
- **测试贡献者** - 提供测试和质量保证的贡献者
- **社区支持者** - 帮助其他用户的贡献者

### 认可方式

1. **README贡献者列表**
2. **发布说明中的致谢**
3. **GitHub贡献者徽章**
4. **学术论文中的致谢**

## 📜 社区准则

### 行为准则

我们采用 [Contributor Covenant](https://www.contributor-covenant.org/) 行为准则：

- **友好包容** - 欢迎不同背景的贡献者
- **尊重他人** - 尊重不同的观点和经验
- **建设性反馈** - 提供有帮助的、建设性的反馈
- **专业态度** - 保持专业和礼貌的交流

### 学术诚信

由于本项目涉及网络安全研究：

- **负责任披露** - 发现的安全问题应负责任地披露
- **研究用途** - 确保研究成果用于防御性目的
- **引用规范** - 正确引用相关研究和数据来源
- **伦理考量** - 遵守研究伦理和法律法规

## 📧 联系方式

如有任何问题或需要帮助：

- **GitHub Issues**: [报告问题](https://github.com/jasontimwong/ais-attack-system/issues)
- **GitHub Discussions**: [参与讨论](https://github.com/jasontimwong/ais-attack-system/discussions)
- **邮件**: jason@example.com

---

感谢您对AIS攻击生成系统的贡献！您的参与将推动海事网络安全研究的发展。 🚢⚓
