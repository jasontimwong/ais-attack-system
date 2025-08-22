#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S4 - 位置偏移诱发碰撞预警 (Position Offset Induced Collision Warning)

该攻击通过修改真实船只的位置信息，使其看起来正在快速接近对方，
从而触发海上交通管理系统的碰撞预警。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.attack_plugin_v2 import StreamingAttackPlugin, DataPatch, PatchType

logger = logging.getLogger(__name__)


@dataclass
class VesselPair:
    """船只配对信息"""
    vessel_a_mmsi: int
    vessel_b_mmsi: int
    initial_distance: float  # 初始距离(海里)
    cpa_time: datetime  # 预计最近会遇时间
    cpa_distance: float  # 预计最近会遇距离(海里)
    attack_start_time: datetime  # 攻击开始时间
    attack_end_time: datetime  # 攻击结束时间


class PositionOffsetAttack(StreamingAttackPlugin):
    """位置偏移攻击实现"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.attack_type = "position_offset"
        self.category = "information_tampering"
        
        # 攻击参数
        self.offset_degree = params.get('offset_degree', 0.05)  # 位置偏移量(度)
        self.trigger_distance = params.get('trigger_distance', 10.0)  # 触发距离(海里)
        self.safety_threshold = params.get('safety_threshold', 5.0)  # 安全阈值(海里)
        self.attack_duration = params.get('attack_duration', 1800)  # 攻击持续时间(秒)
        self.min_vessel_speed = params.get('min_vessel_speed', 8.0)  # 最小船速(节)
        self.max_vessel_speed = params.get('max_vessel_speed', 20.0)  # 最大船速(节)
        
        # 攻击状态
        self.vessel_pairs: List[VesselPair] = []
        self.active_attacks: Dict[int, VesselPair] = {}  # MMSI -> VesselPair
        self.last_analysis_time: Optional[datetime] = None
        
    def validate_params(self) -> bool:
        """验证攻击参数"""
        if not (0.001 <= self.offset_degree <= 0.1):
            logger.error(f"offset_degree must be between 0.001 and 0.1, got {self.offset_degree}")
            return False
            
        if not (1.0 <= self.trigger_distance <= 50.0):
            logger.error(f"trigger_distance must be between 1.0 and 50.0, got {self.trigger_distance}")
            return False
            
        if not (0.5 <= self.safety_threshold <= 10.0):
            logger.error(f"safety_threshold must be between 0.5 and 10.0, got {self.safety_threshold}")
            return False
            
        if not (300 <= self.attack_duration <= 3600):
            logger.error(f"attack_duration must be between 300 and 3600, got {self.attack_duration}")
            return False
            
        return True
        
    def generate_patches(self, 
                        chunk_iter: Iterator[pd.DataFrame],
                        config: Dict[str, Any]) -> Iterator[DataPatch]:
        """生成位置偏移攻击补丁"""
        
        for chunk_id, chunk in enumerate(chunk_iter):
            if chunk.empty:
                continue
                
            # 分析船只配对
            current_time = chunk['BaseDateTime'].iloc[0] if len(chunk) > 0 else datetime.utcnow()
            
            # 每5分钟重新分析一次船只配对
            if (self.last_analysis_time is None or 
                (current_time - self.last_analysis_time).total_seconds() > 300):
                self._analyze_vessel_pairs(chunk, current_time)
                self.last_analysis_time = current_time
                
            # 更新活跃攻击
            self._update_active_attacks(current_time)
            
            # 生成位置偏移补丁
            if self.active_attacks:
                modifications = self._generate_position_modifications(chunk, current_time)
                if modifications is not None and not modifications.empty:
                    yield DataPatch(
                        patch_type=PatchType.MODIFY,
                        chunk_id=chunk_id,
                        modifications=modifications
                    )
                    
    def _analyze_vessel_pairs(self, chunk: pd.DataFrame, current_time: datetime) -> None:
        """分析船只配对"""
        # 筛选合适的船只
        suitable_vessels = self._filter_suitable_vessels(chunk)
        
        if len(suitable_vessels) < 2:
            logger.debug("Not enough suitable vessels for pairing")
            return
            
        # 计算船只间距离和CPA
        new_pairs = []
        for i, vessel_a in suitable_vessels.iterrows():
            for j, vessel_b in suitable_vessels.iterrows():
                if i >= j:  # 避免重复配对
                    continue
                    
                # 计算距离
                distance = self._calculate_distance(
                    (vessel_a['LAT'], vessel_a['LON']),
                    (vessel_b['LAT'], vessel_b['LON'])
                )
                
                # 检查是否在触发距离内
                if distance <= self.trigger_distance:
                    # 计算CPA
                    cpa_time, cpa_distance = self._calculate_cpa(vessel_a, vessel_b)
                    
                    # 检查是否会产生碰撞风险
                    if cpa_distance <= self.safety_threshold:
                        pair = VesselPair(
                            vessel_a_mmsi=int(vessel_a['MMSI']),
                            vessel_b_mmsi=int(vessel_b['MMSI']),
                            initial_distance=distance,
                            cpa_time=cpa_time,
                            cpa_distance=cpa_distance,
                            attack_start_time=current_time,
                            attack_end_time=current_time + timedelta(seconds=self.attack_duration)
                        )
                        new_pairs.append(pair)
                        
        # 更新船只配对列表
        self.vessel_pairs.extend(new_pairs)
        logger.info(f"Found {len(new_pairs)} new vessel pairs for attack")
        
    def _filter_suitable_vessels(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """筛选合适的船只"""
        # 筛选条件：
        # 1. 有效的位置信息
        # 2. 合适的速度范围
        # 3. 货船或油轮类型
        
        filtered = chunk[
            (chunk['LAT'].notna()) & 
            (chunk['LON'].notna()) &
            (chunk['LAT'] != 0) &
            (chunk['LON'] != 0) &
            (chunk['SOG'] >= self.min_vessel_speed) &
            (chunk['SOG'] <= self.max_vessel_speed) &
            (chunk['VesselType'].isin([70, 71, 72, 73, 74, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]))  # 货船和油轮
        ].copy()
        
        return filtered
        
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间距离(海里)"""
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 3440.065 * c  # 地球半径(海里)
        
    def _calculate_cpa(self, vessel_a: pd.Series, vessel_b: pd.Series) -> Tuple[datetime, float]:
        """计算最近会遇点(CPA)"""
        # 位置和速度
        lat1, lon1 = vessel_a['LAT'], vessel_a['LON']
        lat2, lon2 = vessel_b['LAT'], vessel_b['LON']
        
        # 速度向量(节 -> 海里/小时)
        speed1 = vessel_a['SOG']
        course1 = np.radians(vessel_a['COG'])
        speed2 = vessel_b['SOG']
        course2 = np.radians(vessel_b['COG'])
        
        # 转换为海里/小时的速度向量
        vx1 = speed1 * np.sin(course1)
        vy1 = speed1 * np.cos(course1)
        vx2 = speed2 * np.sin(course2)
        vy2 = speed2 * np.cos(course2)
        
        # 相对位置和速度
        dx = self._calculate_distance((lat1, lon1), (lat1, lon2)) * np.sign(lon2 - lon1)
        dy = self._calculate_distance((lat1, lon1), (lat2, lon1)) * np.sign(lat2 - lat1)
        
        dvx = vx2 - vx1
        dvy = vy2 - vy1
        
        # 计算CPA时间
        dv_squared = dvx**2 + dvy**2
        
        if dv_squared == 0:
            # 相对速度为0，船只不相对运动
            cpa_time = vessel_a['BaseDateTime']
            cpa_distance = np.sqrt(dx**2 + dy**2)
        else:
            # CPA时间(小时)
            t_cpa = -(dx * dvx + dy * dvy) / dv_squared
            
            # 确保时间不为负
            t_cpa = max(0, t_cpa)
            
            # CPA位置
            x_cpa = dx + dvx * t_cpa
            y_cpa = dy + dvy * t_cpa
            
            # CPA距离
            cpa_distance = np.sqrt(x_cpa**2 + y_cpa**2)
            
            # CPA时间
            cpa_time = vessel_a['BaseDateTime'] + timedelta(hours=t_cpa)
            
        return cpa_time, cpa_distance
        
    def _update_active_attacks(self, current_time: datetime) -> None:
        """更新活跃攻击"""
        # 开始新的攻击
        for pair in self.vessel_pairs:
            if (pair.attack_start_time <= current_time < pair.attack_end_time and
                pair.vessel_a_mmsi not in self.active_attacks and
                pair.vessel_b_mmsi not in self.active_attacks):
                
                self.active_attacks[pair.vessel_a_mmsi] = pair
                self.active_attacks[pair.vessel_b_mmsi] = pair
                logger.info(f"Started attack on vessel pair: {pair.vessel_a_mmsi} <-> {pair.vessel_b_mmsi}")
                
        # 结束过期的攻击
        expired_mmsis = []
        for mmsi, pair in self.active_attacks.items():
            if current_time >= pair.attack_end_time:
                expired_mmsis.append(mmsi)
                
        for mmsi in expired_mmsis:
            pair = self.active_attacks.pop(mmsi)
            logger.info(f"Ended attack on vessel: {mmsi}")
            
        # 清理过期的船只配对
        self.vessel_pairs = [pair for pair in self.vessel_pairs 
                           if current_time < pair.attack_end_time + timedelta(hours=1)]
                           
    def _generate_position_modifications(self, chunk: pd.DataFrame, current_time: datetime) -> Optional[pd.DataFrame]:
        """生成位置修改"""
        if not self.active_attacks:
            return None
            
        # 筛选需要修改的记录
        attack_mmsis = set(self.active_attacks.keys())
        mask = chunk['MMSI'].isin(attack_mmsis)
        
        if not mask.any():
            return None
            
        # 创建修改数据框
        modifications = pd.DataFrame(index=chunk.index[mask])
        
        for idx in modifications.index:
            mmsi = int(chunk.loc[idx, 'MMSI'])
            if mmsi not in self.active_attacks:
                continue
                
            pair = self.active_attacks[mmsi]
            
            # 计算偏移方向
            if mmsi == pair.vessel_a_mmsi:
                target_mmsi = pair.vessel_b_mmsi
            else:
                target_mmsi = pair.vessel_a_mmsi
                
            # 查找目标船只位置
            target_positions = chunk[chunk['MMSI'] == target_mmsi]
            if target_positions.empty:
                continue
                
            target_pos = target_positions.iloc[0]
            
            # 计算偏移
            current_lat = chunk.loc[idx, 'LAT']
            current_lon = chunk.loc[idx, 'LON']
            target_lat = target_pos['LAT']
            target_lon = target_pos['LON']
            
            # 计算朝向目标的偏移向量
            offset_lat, offset_lon = self._calculate_offset_vector(
                (current_lat, current_lon),
                (target_lat, target_lon),
                self.offset_degree
            )
            
            # 应用偏移
            modifications.loc[idx, 'LAT'] = current_lat + offset_lat
            modifications.loc[idx, 'LON'] = current_lon + offset_lon
            
        return modifications if not modifications.empty else None
        
    def _calculate_offset_vector(self, current_pos: Tuple[float, float], 
                               target_pos: Tuple[float, float], 
                               offset_magnitude: float) -> Tuple[float, float]:
        """计算偏移向量"""
        current_lat, current_lon = current_pos
        target_lat, target_lon = target_pos
        
        # 计算方向向量
        dx = target_lon - current_lon
        dy = target_lat - current_lat
        
        # 归一化
        distance = np.sqrt(dx**2 + dy**2)
        if distance == 0:
            return 0.0, 0.0
            
        unit_x = dx / distance
        unit_y = dy / distance
        
        # 应用偏移量
        offset_lat = unit_y * offset_magnitude
        offset_lon = unit_x * offset_magnitude
        
        return offset_lat, offset_lon
        
    def get_impact_type(self) -> str:
        """获取攻击影响类型"""
        return "CollisionRisk"
        
    def get_attack_statistics(self) -> Dict[str, Any]:
        """获取攻击统计信息"""
        return {
            'total_vessel_pairs': len(self.vessel_pairs),
            'active_attacks': len(self.active_attacks),
            'attack_parameters': {
                'offset_degree': self.offset_degree,
                'trigger_distance': self.trigger_distance,
                'safety_threshold': self.safety_threshold,
                'attack_duration': self.attack_duration
            },
            'vessel_pairs': [
                {
                    'vessel_a_mmsi': pair.vessel_a_mmsi,
                    'vessel_b_mmsi': pair.vessel_b_mmsi,
                    'initial_distance': pair.initial_distance,
                    'cpa_distance': pair.cpa_distance,
                    'attack_start_time': pair.attack_start_time.isoformat(),
                    'attack_end_time': pair.attack_end_time.isoformat()
                }
                for pair in self.vessel_pairs
            ]
        }