# 🚢 AIS Attack Generation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maritime Research](https://img.shields.io/badge/Domain-Maritime%20Cybersecurity-green.svg)](https://github.com/jasontimwong/ais-attack-system)

> 先进的AIS（自动识别系统）攻击生成与可视化系统，用于海事网络安全研究和防御评估

## 🎯 系统概述

本系统是一个完整的AIS攻击生成平台，实现了：

- **多阶段渐进式攻击编排** - Flash-Cross策略的4阶段攻击模式
- **MCDA+模糊逻辑目标选择** - 智能攻击目标筛选算法  
- **MMG约束引擎** - 6自由度船舶动力学模型
- **COLREGs合规性检查** - 国际海上避碰规则实现
- **自动标注管道** - 攻击数据自动标签生成
- **ECDIS可视化质保** - 专业海图显示系统

## 📊 核心成果

### Attack Generator v1
- ✅ **35个验证场景** - 涵盖货船、油轮、集装箱船、客船
- ✅ **98.7%物理一致性** - 轨迹符合船舶动力学
- ✅ **94.3%诱导违规成功率** - 攻击成功触发规避机动
- ✅ **2.1%COLREGs违规率** - 低误报的规则实现

### 性能指标
- 🚀 **处理速度**: 120万AIS消息/小时
- ⚡ **响应延迟**: <10ms
- 🎮 **仿真速度**: 112×实时
- 💾 **内存效率**: <1GB处理1TB数据集

## 🏗️ 系统架构

```
ais-attack-system/
├── core/                    # 核心攻击生成引擎
│   ├── attack_orchestrator/ # 多阶段攻击编排
│   ├── target_selector/     # MCDA目标选择
│   ├── physics_engine/      # MMG约束引擎
│   ├── colregs_validator/   # 避碰规则验证
│   └── auto_labeler/        # 自动标注系统
├── attacks/                 # 9种攻击类型实现
│   ├── flash_cross/         # S1: 闪现横越攻击
│   ├── zone_violation/      # S2: 区域违规
│   ├── ghost_swarm/         # S3: 幽灵船群
│   └── ...                  # S4-S9 其他攻击类型
├── visualization/           # ECDIS可视化系统
│   ├── ecdis_renderer/      # 海图渲染引擎
│   ├── web_interface/       # Web可视化界面
│   └── bridge_integration/  # 船桥系统集成
├── datasets/                # 数据集管理
│   ├── scenarios/           # 35个攻击场景
│   ├── labels/              # 自动生成标签
│   └── statistics/          # 质量统计报告
└── tools/                   # 辅助工具集
    ├── batch_runner/        # 批量执行工具
    ├── validation/          # 数据验证工具
    └── export/              # 格式转换工具
```

## 🚀 快速开始

### 环境要求

```bash
Python 3.8+
Node.js 16+
OpenCPN 5.6+ (可选)
Bridge Command 5.0+ (可选)
```

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/jasontimwong/ais-attack-system.git
cd ais-attack-system

# 2. 安装Python依赖
pip install -r requirements.txt

# 3. 安装Web界面依赖
cd visualization/web_interface
npm install

# 4. 运行系统检查
python tools/system_check.py
```

### 基本使用

```bash
# 生成单个攻击场景
python -m core.attack_orchestrator --scenario s1_flash_cross

# 批量生成所有场景
python tools/batch_runner/run_all_scenarios.py

# 启动Web可视化界面
cd visualization/web_interface && npm run dev

# 生成ECDIS报告
python visualization/ecdis_renderer/create_report.py --scenario s3_ghost_swarm
```

## 📈 攻击类型说明

| 编号 | 攻击类型 | 描述 | 技术特点 |
|------|----------|------|----------|
| S1 | Flash-Cross | 闪现横越攻击 | 4阶段渐进式编排 |
| S2 | Zone Violation | 区域违规 | 位置欺骗+区域入侵 |
| S3 | Ghost Swarm | 幽灵船群 | 8船协调攻击 |
| S4 | Position Offset | 位置偏移 | 1.5海里位移攻击 |
| S5 | Port Spoofing | 港口欺骗 | 港口区域干扰 |
| S6 | Course Disruption | 航向破坏 | 强制规避机动 |
| S7 | Identity Swap | 身份交换 | MMSI身份互换 |
| S8 | Identity Clone | 身份克隆 | 船舶身份复制 |
| S9 | Identity Whitewashing | 身份洗白 | 声誉攻击模式 |

## 🔬 技术创新

### 1. 多阶段渐进式攻击编排
- **并行跟随阶段** (2分钟) - 建立跟踪，建立信任
- **接近启动阶段** (30秒) - 逐渐加速，保持欺骗
- **闪现横越阶段** (45秒) - 快速接近，触发碰撞警报
- **静默消失阶段** (30+秒) - 引起反应后消失

### 2. 智能目标选择算法
```python
vulnerability_score = w1 * isolation_factor + 
                     w2 * predictability_score + 
                     w3 * target_value + 
                     w4 * cascade_potential
```

### 3. 物理约束引擎
- 最大转向率：3°/秒 (IMO标准)
- 速度变化率：0.5节/分钟
- 最小CPA：0.1海里
- 船体动力学：长宽比效应

## 📊 验证结果

### 数据集统计 (Dataset v0.1)
- **总场景数**: 35个已验证场景
- **船舶类型覆盖**: 货船40%，油轮25%，集装箱船20%，客船15%
- **地理分布**: 海峡12个(34%)，港口15个(43%)，TSS 8个(23%)
- **质量指标**: 物理一致性98.7%，COLREGs违规率2.1%

### 性能基准测试
- **验证成功率**: 从0%提升到85.7%
- **处理性能**: 120万AIS消息/小时，延迟<10ms
- **仿真速度**: 112×实时能力
- **跨平台验证**: Bridge Command, OpenCPN, 自定义可视化95%相关性

## 🎮 可视化系统

### ECDIS专业海图显示
- IMO标准海事符号
- 实时CPA/TCPA监控
- COLREGs规则分类
- 近失事故重现

### Web交互界面
- MapLibre + DeckGL渲染
- 60FPS轨迹回放
- 实时攻击效果分析
- 多场景对比视图

## 🛠️ 开发工具

```bash
# 数据验证
python tools/validation/check_data_quality.py

# 格式转换
python tools/export/convert_to_geojson.py
python tools/export/export_to_bridge_command.py

# 性能分析
python tools/profiler/analyze_performance.py

# 批量测试
python tools/batch_runner/test_all_scenarios.py
```

## 📖 文档

- [API参考文档](docs/api_reference.md)
- [算法详细说明](docs/algorithms.md)
- [部署指南](docs/deployment.md)
- [贡献指南](docs/contributing.md)

## 🤝 贡献

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢海事网络安全研究社区的支持
- 基于真实AIS数据集进行验证
- 遵循IMO国际海事组织标准

## 📧 联系方式

- **作者**: Jason Tim Wong
- **GitHub**: [@jasontimwong](https://github.com/jasontimwong)
- **项目链接**: [https://github.com/jasontimwong/ais-attack-system](https://github.com/jasontimwong/ais-attack-system)

---

**⚠️ 免责声明**: 本系统仅用于学术研究和防御性安全评估。请勿用于任何恶意目的。
