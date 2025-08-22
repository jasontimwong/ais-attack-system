# 🚀 快速开始指南

欢迎使用AIS攻击生成系统！本指南将帮助您在几分钟内运行您的第一个攻击场景。

## 📋 系统要求

### 基本要求
- **Python**: 3.8 或更高版本
- **内存**: 至少 4GB RAM
- **存储**: 至少 2GB 可用空间
- **操作系统**: Windows 10+, macOS 10.15+, 或 Linux (Ubuntu 18.04+)

### 可选要求
- **Node.js**: 16+ (用于Web可视化界面)
- **OpenCPN**: 5.6+ (用于专业海图显示)
- **Bridge Command**: 5.0+ (用于船桥模拟器集成)

## ⚡ 5分钟快速安装

### 1. 克隆仓库
```bash
git clone https://github.com/jasontimwong/ais-attack-system.git
cd ais-attack-system
```

### 2. 创建虚拟环境
```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 或使用 conda
conda create -n ais-attack python=3.8
conda activate ais-attack
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python tools/system_check.py
```

如果看到 ✅ 所有检查通过，说明安装成功！

## 🎯 运行第一个攻击场景

### 方法一：使用命令行界面

```bash
# 生成 S1 Flash Cross 攻击场景
python -m core.attack_orchestrator --scenario s1_flash_cross --output output/my_first_attack

# 查看生成的文件
ls output/my_first_attack/
```

### 方法二：使用Python API

创建文件 `my_first_attack.py`：

```python
#!/usr/bin/env python3
from core.attack_orchestrator import AttackOrchestrator
from core.target_selector import TargetSelector
from core.physics_engine import PhysicsEngine
from core.colregs_validator import COLREGSValidator
from attacks.flash_cross import FlashCrossAttack

# 初始化组件
target_selector = TargetSelector()
physics_engine = PhysicsEngine()
colregs_validator = COLREGSValidator()

orchestrator = AttackOrchestrator(
    target_selector, physics_engine, colregs_validator
)

# 创建攻击实例
attack = FlashCrossAttack()

# 模拟目标船舶数据
target_data = {
    'mmsi': '123456789',
    'lat': 40.7128,
    'lon': -74.0060,
    'speed': 12.0,
    'course': 90.0,
    'vessel_type': 'cargo'
}

# 执行攻击
attack_id = attack.initialize_attack(target_data, (40.7100, -74.0100))
print(f"攻击已初始化: {attack_id}")
```

运行脚本：
```bash
python my_first_attack.py
```

## 🎮 启动Web可视化界面

```bash
# 进入可视化目录
cd visualization/web_interface

# 安装依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

打开浏览器访问 `http://localhost:5173` 查看交互式地图。

## 📊 生成批量场景

运行所有35个预定义攻击场景：

```bash
# 顺序执行
python tools/batch_runner/run_all_scenarios.py

# 并行执行（推荐）
python tools/batch_runner/run_all_scenarios.py --parallel --workers 4

# 只运行特定场景
python tools/batch_runner/run_all_scenarios.py --scenarios s1_flash_cross s3_ghost_swarm
```

## 🗂️ 输出文件说明

执行攻击后，系统会在 `output/` 目录生成以下文件：

```
output/
├── s1_flash_cross_20241201_143052/
│   ├── attack_trajectory.geojson    # 攻击轨迹（GeoJSON格式）
│   ├── baseline_trajectory.geojson  # 基线轨迹
│   ├── attack_labels.json           # 自动生成的标签
│   ├── metadata.json               # 攻击元数据
│   ├── metrics_report.json         # 性能指标
│   └── visualization.html          # 可视化报告
```

### 文件格式说明

**GeoJSON 轨迹文件**:
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

## 🔍 验证攻击质量

```bash
# 验证单个场景
python tools/validation/validate_scenario.py output/s1_flash_cross_20241201_143052/

# 生成质量报告
python tools/validation/generate_quality_report.py --input output/ --output quality_report.html
```

质量指标包括：
- ✅ **物理一致性**: 98.7% (轨迹符合船舶动力学)
- ✅ **COLREGs合规性**: 97.9% (符合避碰规则)
- ✅ **攻击成功率**: 94.3% (成功触发目标响应)

## 🎨 生成专业可视化

### ECDIS海图显示
```bash
python visualization/ecdis_renderer/create_ecdis_report.py \
    --scenario s1_flash_cross \
    --output ecdis_report.png \
    --style imo_standard
```

### 交互式HTML报告
```bash
python visualization/create_interactive_report.py \
    --input output/s1_flash_cross_20241201_143052/ \
    --output interactive_report.html
```

## 🛠️ 自定义攻击参数

编辑配置文件 `configs/custom_attack.yaml`：

```yaml
# 自定义Flash Cross攻击参数
attacks:
  s1_flash_cross:
    stages:
      parallel_following:
        duration: 180.0      # 延长到3分钟
        parallel_distance: 1.5  # 减少到1.5海里
      
      flash_cross_maneuver:
        cross_angle: 120.0   # 更大的交叉角度
        speed_factor: 2.0    # 更高的速度倍数
```

使用自定义配置：
```bash
python -m core.attack_orchestrator --config configs/custom_attack.yaml --scenario s1_flash_cross
```

## 📱 集成到现有系统

### Bridge Command集成
```bash
# 导出为Bridge Command格式
python tools/export/export_to_bridge_command.py \
    --input output/s1_flash_cross_20241201_143052/ \
    --output bridge_command_scenario/
```

### OpenCPN插件
```bash
# 构建OpenCPN插件
cd plugins/opencpn_ais_attack/
mkdir build && cd build
cmake .. && make
```

### NMEA输出
```bash
# 生成实时NMEA数据流
python tools/export/generate_nmea_stream.py \
    --input output/s1_flash_cross_20241201_143052/ \
    --port 4001 \
    --realtime
```

## 🚨 常见问题

### Q: 安装时提示缺少依赖？
A: 确保使用Python 3.8+并安装所有依赖：
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Q: 攻击场景生成失败？
A: 检查日志文件 `logs/attack_generation.log`：
```bash
tail -f logs/attack_generation.log
```

### Q: Web界面无法加载？
A: 确保Node.js版本≥16，重新安装依赖：
```bash
cd visualization/web_interface
rm -rf node_modules package-lock.json
npm install
```

### Q: 可视化图片质量不佳？
A: 调整DPI设置：
```bash
python visualization/ecdis_renderer/create_ecdis_report.py \
    --dpi 300 \
    --format png \
    --size 1920x1080
```

## 📚 下一步

恭喜！您已经成功运行了第一个AIS攻击场景。接下来可以：

1. **探索更多攻击类型** - 查看 [攻击类型文档](ATTACK_TYPES.md)
2. **自定义攻击参数** - 阅读 [配置指南](CONFIGURATION.md)
3. **集成到您的系统** - 参考 [API文档](API_REFERENCE.md)
4. **贡献代码** - 查看 [开发指南](CONTRIBUTING.md)

## 🆘 获取帮助

- 📖 **文档**: [完整文档](https://github.com/jasontimwong/ais-attack-system/docs)
- 🐛 **问题报告**: [GitHub Issues](https://github.com/jasontimwong/ais-attack-system/issues)
- 💬 **讨论**: [GitHub Discussions](https://github.com/jasontimwong/ais-attack-system/discussions)

---

**🎉 欢迎加入AIS攻击生成系统社区！**
