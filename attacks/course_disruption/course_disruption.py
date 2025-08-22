#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S6 - 航向扰动造成监视真空 (Course Disruption Creating Surveillance Vacuum)

该攻击通过对船只的航向(COG)和速度(SOG)数据注入随机扰动，
使其在AIS监控中表现为行为不稳定，从而在监视系统中产生"真空"效果。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass
import math
import random

from core.attack_plugin_v2 import StreamingAttackPlugin, DataPatch, PatchType

logger = logging.getLogger(__name__)


@dataclass
class DisruptionState:
    """扰动状态信息"""
    mmsi: int
    start_time: datetime
    phase_start_time: datetime
    current_phase: int  # 当前扰动阶段
    disruption_intensity: float  # 扰动强度(0.0-1.0)
    base_cog: float  # 基础航向
    base_sog: float  # 基础速度
    time_factor: float  # 时间因子(用于sin函数)


class CourseDisruptionAttack(StreamingAttackPlugin):
    """航向扰动造成监视真空攻击实现"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.attack_type = "course_disruption"
        self.category = "information_tampering"
        
        # 攻击参数
        self.cog_disruption_range = params.get('cog_disruption_range', 15.0)  # COG扰动范围(度)
        self.sog_disruption_ratio = params.get('sog_disruption_ratio', 0.15)  # SOG扰动比例
        self.phase_duration = params.get('phase_duration', 7200)  # 每阶段持续时间(秒)
        self.intensity_ramp_time = params.get('intensity_ramp_time', 600)  # 强度渐进时间(秒)
        self.noise_level = params.get('noise_level', 0.3)  # 随机噪声级别(0-1)
        
        # 目标筛选参数
        self.target_vessel_types = params.get('target_vessel_types', [70, 71, 72, 73, 74, 79, 80])  # 货船和油轮
        self.min_speed = params.get('min_speed', 5.0)  # 最小速度(节)
        self.max_speed = params.get('max_speed', 20.0)  # 最大速度(节)
        
        # 攻击状态
        self.disruption_states: Dict[int, DisruptionState] = {}
        self.last_analysis_time: Optional[datetime] = None
        
        # 随机数生成器
        self.rng = np.random.RandomState(params.get('random_seed', 42))
        
    def validate_params(self) -> bool:
        """验证攻击参数"""
        if not (0.0 < self.cog_disruption_range <= 180.0):
            logger.error(f"cog_disruption_range must be between 0 and 180, got {self.cog_disruption_range}")
            return False
            
        if not (0.0 < self.sog_disruption_ratio <= 1.0):
            logger.error(f"sog_disruption_ratio must be between 0 and 1, got {self.sog_disruption_ratio}")
            return False
            
        if not (300 <= self.phase_duration <= 86400):
            logger.error(f"phase_duration must be between 300 and 86400, got {self.phase_duration}")
            return False
            
        if not (0.0 <= self.noise_level <= 1.0):
            logger.error(f"noise_level must be between 0 and 1, got {self.noise_level}")
            return False
            
        return True
        
    def generate_patches(self, 
                        chunk_iter: Iterator[pd.DataFrame],
                        config: Dict[str, Any]) -> Iterator[DataPatch]:
        """生成航向扰动攻击补丁"""
        
        for chunk_id, chunk in enumerate(chunk_iter):
            if chunk.empty:
                continue
                
            # 获取当前时间
            current_time = chunk['BaseDateTime'].iloc[0] if len(chunk) > 0 else datetime.utcnow()
            
            # 分析和选择目标船只
            if (self.last_analysis_time is None or 
                (current_time - self.last_analysis_time).total_seconds() > 300):
                self._analyze_targets(chunk, current_time)
                self.last_analysis_time = current_time
                
            # 更新扰动状态
            self._update_disruption_states(current_time)
            
            # 生成扰动补丁
            if self.disruption_states:
                modifications = self._generate_disruptions(chunk, current_time)
                if modifications is not None and not modifications.empty:
                    yield DataPatch(
                        patch_type=PatchType.MODIFY,
                        chunk_id=chunk_id,
                        modifications=modifications
                    )
                    
    def _analyze_targets(self, chunk: pd.DataFrame, current_time: datetime) -> None:
        """分析和选择攻击目标"""
        # 筛选合适的船只
        suitable_vessels = chunk[
            (chunk['VesselType'].isin(self.target_vessel_types)) &
            (chunk['SOG'] >= self.min_speed) &
            (chunk['SOG'] <= self.max_speed)
        ]
        
        for idx, vessel in suitable_vessels.iterrows():
            mmsi = int(vessel['MMSI'])
            
            # 如果船只不在扰动列表中，添加它
            if mmsi not in self.disruption_states:
                # 随机决定是否攻击这艘船(30%概率)
                if self.rng.random() < 0.3:
                    self._start_disruption(mmsi, vessel, current_time)
                    
    def _start_disruption(self, mmsi: int, vessel: pd.Series, current_time: datetime) -> None:
        """开始对船只进行扰动"""
        state = DisruptionState(
            mmsi=mmsi,
            start_time=current_time,
            phase_start_time=current_time,
            current_phase=0,
            disruption_intensity=0.0,
            base_cog=float(vessel['COG']),
            base_sog=float(vessel['SOG']),
            time_factor=0.0
        )
        
        self.disruption_states[mmsi] = state
        logger.info(f"开始扰动船只 {mmsi}, 基础航向: {state.base_cog}°, 基础速度: {state.base_sog}节")
        
    def _update_disruption_states(self, current_time: datetime) -> None:
        """更新扰动状态"""
        expired_mmsis = []
        
        for mmsi, state in self.disruption_states.items():
            # 更新时间因子
            elapsed = (current_time - state.start_time).total_seconds()
            state.time_factor = elapsed / 60.0  # 转换为分钟
            
            # 更新扰动强度(渐进式)
            if elapsed < self.intensity_ramp_time:
                state.disruption_intensity = elapsed / self.intensity_ramp_time
            else:
                state.disruption_intensity = 1.0
                
            # 检查是否需要切换阶段
            phase_elapsed = (current_time - state.phase_start_time).total_seconds()
            if phase_elapsed > self.phase_duration:
                state.current_phase += 1
                state.phase_start_time = current_time
                # 随机调整基础参数，增加不可预测性
                state.base_cog = (state.base_cog + self.rng.uniform(-30, 30)) % 360
                logger.info(f"船只 {mmsi} 进入扰动阶段 {state.current_phase}")
                
            # 检查是否应该结束扰动(最多3个阶段)
            if state.current_phase >= 3:
                expired_mmsis.append(mmsi)
                
        # 清理过期的扰动
        for mmsi in expired_mmsis:
            del self.disruption_states[mmsi]
            logger.info(f"结束船只 {mmsi} 的扰动")
            
    def _generate_disruptions(self, chunk: pd.DataFrame, current_time: datetime) -> Optional[pd.DataFrame]:
        """生成扰动数据"""
        # 筛选需要扰动的记录
        disrupted_mmsis = set(self.disruption_states.keys())
        mask = chunk['MMSI'].isin(disrupted_mmsis)
        
        if not mask.any():
            return None
            
        # 创建修改数据框
        modifications = pd.DataFrame(index=chunk.index[mask])
        
        for idx in modifications.index:
            mmsi = int(chunk.loc[idx, 'MMSI'])
            if mmsi not in self.disruption_states:
                continue
                
            state = self.disruption_states[mmsi]
            
            # 计算扰动后的COG
            disrupted_cog = self._calculate_disrupted_cog(state)
            
            # 计算扰动后的SOG
            disrupted_sog = self._calculate_disrupted_sog(state)
            
            # 应用修改
            modifications.loc[idx, 'COG'] = disrupted_cog
            modifications.loc[idx, 'SOG'] = disrupted_sog
            
        return modifications if not modifications.empty else None
        
    def _calculate_disrupted_cog(self, state: DisruptionState) -> float:
        """计算扰动后的航向"""
        # 基础正弦波扰动
        sin_factor = math.sin(state.time_factor * 0.1)  # 0.1控制周期
        base_disruption = sin_factor * self.cog_disruption_range * state.disruption_intensity
        
        # 添加随机噪声
        noise = self.rng.uniform(-5, 5) * self.noise_level * state.disruption_intensity
        
        # 计算最终航向
        disrupted_cog = state.base_cog + base_disruption + noise
        
        # 确保在0-360度范围内
        disrupted_cog = disrupted_cog % 360
        if disrupted_cog < 0:
            disrupted_cog += 360
            
        return disrupted_cog
        
    def _calculate_disrupted_sog(self, state: DisruptionState) -> float:
        """计算扰动后的速度"""
        # 基础正弦波扰动(相位偏移，与COG不同步)
        sin_factor = math.sin(state.time_factor * 0.12 + math.pi/4)  # 不同频率和相位
        base_disruption = sin_factor * self.sog_disruption_ratio * state.disruption_intensity
        
        # 添加随机噪声
        noise = self.rng.uniform(-0.05, 0.05) * self.noise_level * state.disruption_intensity
        
        # 计算扰动因子
        disruption_factor = 1.0 + base_disruption + noise
        
        # 计算最终速度
        disrupted_sog = state.base_sog * disruption_factor
        
        # 确保在合理范围内(0-30节)
        disrupted_sog = max(0.0, min(30.0, disrupted_sog))
        
        return disrupted_sog
        
    def get_impact_type(self) -> str:
        """获取攻击影响类型"""
        return "SurveillanceGap"
        
    def get_attack_statistics(self) -> Dict[str, Any]:
        """获取攻击统计信息"""
        active_disruptions = []
        for mmsi, state in self.disruption_states.items():
            active_disruptions.append({
                'mmsi': mmsi,
                'start_time': state.start_time.isoformat(),
                'current_phase': state.current_phase,
                'disruption_intensity': state.disruption_intensity,
                'base_cog': state.base_cog,
                'base_sog': state.base_sog,
                'elapsed_minutes': state.time_factor
            })
            
        return {
            'active_disruptions': len(self.disruption_states),
            'attack_parameters': {
                'cog_disruption_range': self.cog_disruption_range,
                'sog_disruption_ratio': self.sog_disruption_ratio,
                'phase_duration': self.phase_duration,
                'noise_level': self.noise_level
            },
            'target_criteria': {
                'vessel_types': self.target_vessel_types,
                'speed_range': (self.min_speed, self.max_speed)
            },
            'disruption_details': active_disruptions
        }