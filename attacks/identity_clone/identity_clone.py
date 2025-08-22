#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S8 - 一号多船——身份克隆迷踪 (Identity Clone)

该攻击创建多个具有相同身份的船只（克隆体），分布在不同位置，
造成监控系统混乱，无法确定哪个是真实船只。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
import math
import random

from core.attack_plugin_v2 import StreamingAttackPlugin, DataPatch, PatchType

logger = logging.getLogger(__name__)


@dataclass
class CloneVessel:
    """单个克隆体信息"""
    clone_id: str                          # 克隆体唯一标识
    parent_mmsi: int                       # 原始船只MMSI
    clone_index: int                       # 克隆体索引(0,1,2...)
    
    # 位置和运动状态
    current_lat: float                     # 当前纬度
    current_lon: float                     # 当前经度
    current_sog: float                     # 当前速度
    current_cog: float                     # 当前航向
    
    # 轨迹参数
    base_course: float                     # 基础航向
    speed_factor: float                    # 速度系数
    course_offset: float                   # 航向偏移
    
    # 生命周期
    activation_time: datetime              # 激活时间
    deactivation_time: datetime            # 失活时间
    is_active: bool                        # 是否活跃
    
    # 轨迹生成器状态
    trajectory_phase: float = 0.0          # 轨迹相位
    last_update_time: Optional[datetime] = None


@dataclass
class CloneGroup:
    """克隆组管理"""
    group_id: str                          # 组唯一标识
    parent_vessel: Dict[str, Any]          # 原始船只信息
    parent_mmsi: int                       # 原始船只MMSI
    
    # 克隆体列表
    clones: List[CloneVessel] = field(default_factory=list)
    
    # 创建参数
    clone_count: int = 3                   # 克隆数量
    distribution_pattern: str = 'ring'     # 分布模式
    distribution_radius: float = 200.0     # 分布半径(海里)
    
    # 状态
    creation_time: Optional[datetime] = None
    last_conflict_check: Optional[datetime] = None


class IdentityCloneAttack(StreamingAttackPlugin):
    """身份克隆攻击实现"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.attack_type = "identity_clone"
        self.category = "identity_spoofing"
        
        # 克隆参数 - 支持TEST_MODE
        self.clone_count = self._get_param_with_test_mode(params, 'clone_count', 3, 2)  # 每个目标的克隆数
        self.clone_activation_delay = self._get_param_with_test_mode(params, 'clone_activation_delay', 300, 5)  # 激活延迟(秒)
        self.clone_lifetime = self._get_param_with_test_mode(params, 'clone_lifetime', 3600, 300)  # 生命周期(秒)
        
        # 空间分布参数
        self.min_clone_distance = params.get('min_clone_distance', 50.0)  # 最小间距(海里)
        self.distribution_pattern = params.get('distribution_pattern', 'ring')  # ring/sector/random
        self.distribution_radius = params.get('distribution_radius', 200.0)  # 分布半径(海里)
        
        # 行为参数
        self.behavior_variance = params.get('behavior_variance', 0.2)  # 行为差异度
        self.speed_variance = params.get('speed_variance', 0.2)  # 速度变化范围
        self.course_variance = params.get('course_variance', 30.0)  # 航向变化范围(度)
        
        # 目标选择参数
        self.target_vessel_types = params.get('target_vessel_types', [70, 71, 72, 73, 74, 79, 80])
        self.min_vessel_size = params.get('min_vessel_size', 100)  # 最小船长(米)
        self.priority_score_threshold = params.get('priority_score_threshold', 0.6)
        self.max_clone_groups = params.get('max_clone_groups', 5)  # 最大克隆组数
        
        # 克隆组管理
        self.clone_groups: Dict[int, CloneGroup] = {}  # MMSI -> CloneGroup
        self.active_clones: List[CloneVessel] = []  # 所有活跃克隆体
        
        # 随机数生成器
        self.rng = np.random.RandomState(params.get('random_seed', 42))
        
        # 时间管理 - 支持TEST_MODE
        self.last_target_selection: Optional[datetime] = None
        self.target_selection_interval = self._get_param_with_test_mode(params, 'target_selection_interval', 600, 30)  # 目标选择间隔(秒)
        
    def validate_params(self) -> bool:
        """验证攻击参数"""
        import os
        
        # 在TEST_MODE下放宽参数验证
        if os.getenv('AIS_TEST_MODE'):
            logger.info("TEST_MODE: 放宽参数验证约束")
            
        if not (1 <= self.clone_count <= 5):
            logger.error(f"clone_count must be between 1 and 5, got {self.clone_count}")
            return False
            
        if not (10.0 <= self.min_clone_distance <= 200.0):
            logger.error(f"min_clone_distance must be between 10 and 200, got {self.min_clone_distance}")
            return False
            
        if self.distribution_pattern not in ['ring', 'sector', 'random']:
            logger.error(f"Invalid distribution_pattern: {self.distribution_pattern}")
            return False
            
        if not (0.0 <= self.behavior_variance <= 1.0):
            logger.error(f"behavior_variance must be between 0 and 1, got {self.behavior_variance}")
            return False
            
        return True
        
    def generate_patches(self, 
                        chunk_iter: Iterator[pd.DataFrame],
                        config: Dict[str, Any]) -> Iterator[DataPatch]:
        """生成身份克隆攻击补丁"""
        
        for chunk_id, chunk in enumerate(chunk_iter):
            if chunk.empty:
                continue
                
            current_time = chunk['BaseDateTime'].iloc[0] if len(chunk) > 0 else datetime.utcnow()
            
            # 1. 定期选择新的克隆目标
            if self._should_select_targets(current_time):
                self._select_clone_targets(chunk, current_time)
                self.last_target_selection = current_time
                
            # 2. 更新所有活跃克隆体的状态
            self._update_clone_states(current_time)
            
            # 3. 生成克隆体数据
            clone_records = self._generate_clone_records(chunk, current_time)
            
            if not clone_records.empty:
                yield DataPatch(
                    patch_type=PatchType.INSERT,
                    chunk_id=chunk_id,
                    new_records=clone_records
                )
                
            # 4. 检查冲突和生命周期
            self._check_conflicts_and_lifecycle(current_time)
            
    def _should_select_targets(self, current_time: datetime) -> bool:
        """判断是否需要选择新目标"""
        if self.last_target_selection is None:
            return True
        
        elapsed = (current_time - self.last_target_selection).total_seconds()
        return elapsed >= self.target_selection_interval and len(self.clone_groups) < self.max_clone_groups
        
    def _select_clone_targets(self, chunk: pd.DataFrame, current_time: datetime) -> None:
        """选择克隆目标"""
        # 筛选合适的目标船只
        candidates = chunk[
            (chunk['VesselType'].isin(self.target_vessel_types)) &
            (chunk['Length'].fillna(0) >= self.min_vessel_size) &
            (~chunk['MMSI'].isin(self.clone_groups.keys())) &
            (chunk['LAT'].notna()) &
            (chunk['LON'].notna())
        ]
        
        if candidates.empty:
            return
            
        # 计算优先级分数
        scores = []
        for _, vessel in candidates.iterrows():
            score = self._calculate_priority_score(vessel)
            scores.append(score)
            
        candidates['priority_score'] = scores
        
        # 选择高分目标
        selected = candidates[candidates['priority_score'] >= self.priority_score_threshold]
        selected = selected.nlargest(min(3, self.max_clone_groups - len(self.clone_groups)), 'priority_score')
        
        # 为每个选中的目标创建克隆组
        for _, vessel in selected.iterrows():
            self._create_clone_group(vessel, current_time)
            
    def _calculate_priority_score(self, vessel: pd.Series) -> float:
        """计算目标优先级分数"""
        score = 0.0
        
        # 船型权重
        if vessel['VesselType'] in [70, 71, 72, 73, 74]:  # 货船
            score += 0.3
        elif vessel['VesselType'] in [79, 80, 81, 82, 83]:  # 油轮
            score += 0.4
            
        # 尺寸权重
        length = vessel.get('Length', 0)
        if length > 200:
            score += 0.3
        elif length > 150:
            score += 0.2
            
        # 速度权重(活跃船只)
        sog = vessel.get('SOG', 0)
        if 5 <= sog <= 20:
            score += 0.2
            
        # 位置权重(远离岸边)
        # 简化处理，实际应考虑离岸距离
        score += 0.1
        
        return min(score, 1.0)
        
    def _create_clone_group(self, vessel: pd.Series, current_time: datetime) -> None:
        """创建克隆组"""
        mmsi = int(vessel['MMSI'])
        
        # 创建克隆组
        group = CloneGroup(
            group_id=str(uuid.uuid4()),
            parent_vessel=vessel.to_dict(),
            parent_mmsi=mmsi,
            clone_count=self.clone_count,
            distribution_pattern=self.distribution_pattern,
            distribution_radius=self.distribution_radius,
            creation_time=current_time
        )
        
        # 计算克隆体位置
        clone_positions = self._calculate_clone_positions(
            (vessel['LAT'], vessel['LON']),
            self.clone_count,
            self.distribution_pattern,
            self.distribution_radius
        )
        
        # 创建克隆体
        for i, (lat, lon) in enumerate(clone_positions):
            # 计算激活时间(错开激活)
            activation_delay = self.clone_activation_delay + i * 60  # 每个克隆体间隔1分钟
            activation_time = current_time + timedelta(seconds=activation_delay)
            deactivation_time = activation_time + timedelta(seconds=self.clone_lifetime)
            
            # 创建克隆体
            clone = CloneVessel(
                clone_id=f"{group.group_id}-{i}",
                parent_mmsi=mmsi,
                clone_index=i,
                current_lat=lat,
                current_lon=lon,
                current_sog=vessel['SOG'] * (1 + self.rng.uniform(-self.speed_variance, self.speed_variance)),
                current_cog=vessel['COG'] + self.rng.uniform(-self.course_variance, self.course_variance),
                base_course=vessel['COG'],
                speed_factor=1.0 + self.rng.uniform(-self.speed_variance, self.speed_variance),
                course_offset=self.rng.uniform(-self.course_variance, self.course_variance),
                activation_time=activation_time,
                deactivation_time=deactivation_time,
                is_active=False
            )
            
            group.clones.append(clone)
            
        self.clone_groups[mmsi] = group
        logger.info(f"创建克隆组: MMSI={mmsi}, 克隆数={self.clone_count}, 分布模式={self.distribution_pattern}")
        
    def _calculate_clone_positions(self, origin: Tuple[float, float], 
                                 count: int, pattern: str, 
                                 radius: float) -> List[Tuple[float, float]]:
        """计算克隆体初始位置"""
        lat0, lon0 = origin
        positions = []
        
        if pattern == 'ring':
            # 环形分布
            for i in range(count):
                angle = (2 * math.pi * i) / count
                # 添加一些随机扰动
                angle += self.rng.uniform(-0.2, 0.2)
                r = radius * (1 + self.rng.uniform(-0.1, 0.1))
                
                lat, lon = self._calculate_position_offset(lat0, lon0, r, math.degrees(angle))
                positions.append((lat, lon))
                
        elif pattern == 'sector':
            # 扇形分布(120度扇形)
            sector_angle = 120.0
            base_angle = self.rng.uniform(0, 360)
            
            for i in range(count):
                angle = base_angle + (sector_angle * i) / (count - 1) - sector_angle / 2
                angle += self.rng.uniform(-10, 10)
                r = radius * (0.5 + 0.5 * (i + 1) / count)  # 渐进距离
                
                lat, lon = self._calculate_position_offset(lat0, lon0, r, angle)
                positions.append((lat, lon))
                
        else:  # random
            # 随机分布(保证最小间距)
            max_attempts = 100
            
            for i in range(count):
                attempts = 0
                while attempts < max_attempts:
                    angle = self.rng.uniform(0, 360)
                    r = radius * self.rng.uniform(0.3, 1.0)
                    
                    lat, lon = self._calculate_position_offset(lat0, lon0, r, angle)
                    
                    # 检查与其他克隆体的距离
                    valid = True
                    for other_lat, other_lon in positions:
                        dist = self._calculate_distance((lat, lon), (other_lat, other_lon))
                        if dist < self.min_clone_distance:
                            valid = False
                            break
                            
                    if valid:
                        positions.append((lat, lon))
                        break
                        
                    attempts += 1
                    
                if attempts == max_attempts:
                    # 如果找不到合适位置，使用固定偏移
                    angle = (360.0 * i) / count
                    r = radius * 0.7
                    lat, lon = self._calculate_position_offset(lat0, lon0, r, angle)
                    positions.append((lat, lon))
                    
        return positions
        
    def _calculate_position_offset(self, lat: float, lon: float, 
                                 distance: float, bearing: float) -> Tuple[float, float]:
        """计算给定距离和方位的新位置"""
        # 将距离从海里转换为弧度
        distance_rad = distance / 3440.065  # 地球半径(海里)
        
        # 转换为弧度
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        # 计算新位置
        lat2_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance_rad) +
            math.cos(lat_rad) * math.sin(distance_rad) * math.cos(bearing_rad)
        )
        
        lon2_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance_rad) * math.cos(lat_rad),
            math.cos(distance_rad) - math.sin(lat_rad) * math.sin(lat2_rad)
        )
        
        return math.degrees(lat2_rad), math.degrees(lon2_rad)
        
    def _update_clone_states(self, current_time: datetime) -> None:
        """更新所有克隆体状态"""
        # 激活到期的克隆体
        for group in self.clone_groups.values():
            for clone in group.clones:
                if not clone.is_active and current_time >= clone.activation_time:
                    clone.is_active = True
                    self.active_clones.append(clone)
                    logger.info(f"激活克隆体: {clone.clone_id}")
                    
        # 更新活跃克隆体的轨迹
        for clone in self.active_clones:
            if clone.last_update_time is None:
                clone.last_update_time = current_time
                continue
                
            # 计算时间间隔
            time_delta = (current_time - clone.last_update_time).total_seconds()
            if time_delta <= 0:
                continue
                
            # 更新轨迹
            self._update_clone_trajectory(clone, time_delta)
            clone.last_update_time = current_time
            
    def _update_clone_trajectory(self, clone: CloneVessel, time_delta: float) -> None:
        """更新克隆体轨迹"""
        # 更新轨迹相位
        clone.trajectory_phase += time_delta * 0.001  # 缓慢变化
        
        # 计算航向变化(正弦波动)
        course_variation = self.course_variance * 0.5 * math.sin(clone.trajectory_phase)
        new_cog = (clone.base_course + clone.course_offset + course_variation) % 360
        
        # 计算速度变化
        speed_variation = self.speed_variance * 0.3 * math.sin(clone.trajectory_phase * 1.3)
        new_sog = clone.current_sog * (1 + speed_variation)
        new_sog = max(0, min(30, new_sog))  # 限制在合理范围
        
        # 更新位置
        distance = new_sog * time_delta / 3600.0  # 海里
        new_lat, new_lon = self._calculate_position_offset(
            clone.current_lat, clone.current_lon, distance, new_cog
        )
        
        # 添加小幅随机扰动
        new_lat += self.rng.normal(0, 0.0001)
        new_lon += self.rng.normal(0, 0.0001)
        
        # 更新克隆体状态
        clone.current_lat = new_lat
        clone.current_lon = new_lon
        clone.current_sog = new_sog
        clone.current_cog = new_cog
        
    def _generate_clone_records(self, chunk: pd.DataFrame, current_time: datetime) -> pd.DataFrame:
        """生成克隆体AIS记录"""
        records = []
        
        for clone in self.active_clones:
            if not clone.is_active:
                continue
                
            # 获取父船信息
            parent_mmsi = clone.parent_mmsi
            if parent_mmsi not in self.clone_groups:
                continue
                
            parent_info = self.clone_groups[parent_mmsi].parent_vessel
            
            # 创建克隆记录(完全复制身份信息)
            record = {
                'MMSI': parent_mmsi,  # 使用相同的MMSI!
                'BaseDateTime': current_time,
                'LAT': clone.current_lat,
                'LON': clone.current_lon,
                'SOG': clone.current_sog,
                'COG': clone.current_cog,
                'Heading': clone.current_cog,  # 简化处理
                'VesselName': parent_info.get('VesselName', ''),
                'VesselType': parent_info.get('VesselType', 0),
                'Status': parent_info.get('Status', 0),
                'Length': parent_info.get('Length', 0),
                'Width': parent_info.get('Width', 0),
                'Draft': parent_info.get('Draft', 0),
                'CallSign': parent_info.get('CallSign', ''),
                'Destination': parent_info.get('Destination', ''),
                'ETA': parent_info.get('ETA', pd.NaT),
                'IMO': parent_info.get('IMO', ''),
                # 添加克隆标记(用于内部追踪，实际攻击中不会有)
                '_clone_id': clone.clone_id,
                '_is_clone': True
            }
            
            records.append(record)
            
        if records:
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()
            
    def _check_conflicts_and_lifecycle(self, current_time: datetime) -> None:
        """检查冲突和生命周期"""
        # 检查失活的克隆体
        expired_clones = []
        
        for clone in self.active_clones:
            if current_time >= clone.deactivation_time:
                clone.is_active = False
                expired_clones.append(clone)
                logger.info(f"克隆体失活: {clone.clone_id}")
                
        # 移除失活克隆体
        for clone in expired_clones:
            self.active_clones.remove(clone)
            
        # 检查克隆体间冲突(简化处理)
        if len(self.active_clones) > 1:
            for i, clone1 in enumerate(self.active_clones):
                for clone2 in self.active_clones[i+1:]:
                    # 跳过同组克隆体
                    if clone1.parent_mmsi == clone2.parent_mmsi:
                        continue
                        
                    # 检查距离
                    distance = self._calculate_distance(
                        (clone1.current_lat, clone1.current_lon),
                        (clone2.current_lat, clone2.current_lon)
                    )
                    
                    if distance < self.min_clone_distance * 0.5:
                        # 调整航向避免过近
                        clone1.course_offset += 15
                        clone2.course_offset -= 15
                        logger.warning(f"克隆体过近，调整航向: {clone1.clone_id} <-> {clone2.clone_id}")
                        
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间距离(海里)"""
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 3440.065 * c  # 地球半径(海里)
        
    def get_impact_type(self) -> str:
        """获取攻击影响类型"""
        return "IdentityConfusion"
        
    def get_attack_statistics(self) -> Dict[str, Any]:
        """获取攻击统计信息"""
        total_clones = sum(len(group.clones) for group in self.clone_groups.values())
        active_count = len(self.active_clones)
        
        # 统计每个组的状态
        group_stats = []
        for mmsi, group in self.clone_groups.items():
            active_in_group = sum(1 for clone in group.clones if clone.is_active)
            group_stats.append({
                'parent_mmsi': mmsi,
                'vessel_name': group.parent_vessel.get('VesselName', 'Unknown'),
                'total_clones': len(group.clones),
                'active_clones': active_in_group,
                'distribution_pattern': group.distribution_pattern,
                'creation_time': group.creation_time.isoformat() if group.creation_time else None
            })
            
        return {
            'total_clone_groups': len(self.clone_groups),
            'total_clones_created': total_clones,
            'active_clones': active_count,
            'attack_parameters': {
                'clone_count': self.clone_count,
                'distribution_radius': self.distribution_radius,
                'clone_lifetime': self.clone_lifetime,
                'min_clone_distance': self.min_clone_distance
            },
            'distribution_settings': {
                'pattern': self.distribution_pattern,
                'behavior_variance': self.behavior_variance,
                'speed_variance': self.speed_variance,
                'course_variance': self.course_variance
            },
            'group_details': group_stats
        }