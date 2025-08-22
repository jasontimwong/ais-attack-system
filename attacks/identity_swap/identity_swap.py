#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S7 - AIS身份交换"调包"术 (AIS Identity Swap)

该攻击通过两艘船只在接近时交换彼此的AIS身份信息，
实现监管逃避和身份混淆，类似于现实中的"调包"行为。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid
import math

from core.attack_plugin_v2 import StreamingAttackPlugin, DataPatch, PatchType

logger = logging.getLogger(__name__)


@dataclass
class SwapPair:
    """身份交换对信息"""
    pair_id: str                          # 唯一标识
    vessel_a_mmsi: int                    # A船MMSI
    vessel_b_mmsi: int                    # B船MMSI
    
    # 交换时机
    predicted_cpa_time: Optional[datetime]     # 预计最近点时间
    predicted_cpa_distance: Optional[float]    # 预计最近距离
    swap_trigger_distance: float               # 触发交换距离
    
    # 交换状态
    swap_status: str                           # pending/active/completed/aborted
    swap_start_time: Optional[datetime]        # 实际交换时间
    swap_end_time: Optional[datetime]          # 计划结束时间
    swap_duration: int                         # 持续时间(秒)
    
    # 原始身份信息
    original_a_identity: Dict[str, Any]        # A船原始身份
    original_b_identity: Dict[str, Any]        # B船原始身份
    
    # 相似度评分
    similarity_score: float                    # 船只相似度(0-1)
    
    # 最后更新时间
    last_update_time: Optional[datetime]       # 最后状态更新时间


class IdentitySwapAttack(StreamingAttackPlugin):
    """AIS身份交换攻击实现"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.attack_type = "identity_swap"
        self.category = "identity_spoofing"
        
        # 攻击参数 - 支持TEST_MODE
        self.swap_trigger_distance = self._get_param_with_test_mode(params, 'swap_trigger_distance', 0.5, 1.5)  # 触发距离(海里)
        self.swap_duration = self._get_param_with_test_mode(params, 'swap_duration', 1800, 60)  # 交换持续时间(秒)
        self.min_similarity_score = self._get_param_with_test_mode(params, 'min_similarity_score', 0.6, 0.3)  # 最小相似度
        self.analysis_interval = self._get_param_with_test_mode(params, 'analysis_interval', 300, 10)  # 分析间隔(秒)
        self.max_active_swaps = self._get_param_with_test_mode(params, 'max_active_swaps', 5, 3)  # 最大并发交换数
        
        # 目标筛选参数
        self.target_vessel_types = params.get('target_vessel_types', [70, 71, 72, 73, 74, 79, 80])
        self.min_speed = params.get('min_speed', 5.0)  # 最小速度(节)
        self.max_speed = params.get('max_speed', 20.0)  # 最大速度(节)
        self.vessel_type_tolerance = params.get('vessel_type_tolerance', 10)  # 船型差异容忍度
        
        # 交换状态管理
        self.pending_pairs: List[SwapPair] = []  # 待执行交换
        self.active_pairs: Dict[int, SwapPair] = {}  # 活跃交换(MMSI->SwapPair)
        self.completed_pairs: List[SwapPair] = []  # 已完成交换
        self.paired_vessels: set = set()  # 已配对的船只
        
        # 时间管理
        self.last_analysis_time: Optional[datetime] = None
        
        # 交换字段定义
        self.swap_fields = ['VesselName', 'CallSign', 'Destination']
        
    def validate_params(self) -> bool:
        """验证攻击参数"""
        import os
        
        # 在TEST_MODE下放宽参数验证
        if os.getenv('AIS_TEST_MODE'):
            logger.info("TEST_MODE: 放宽参数验证约束")
            if not (0.1 <= self.swap_trigger_distance <= 2.0):
                logger.error(f"swap_trigger_distance must be between 0.1 and 2.0, got {self.swap_trigger_distance}")
                return False
                
            if not (10 <= self.swap_duration <= 7200):  # 放宽最小值
                logger.error(f"swap_duration must be between 10 and 7200, got {self.swap_duration}")
                return False
                
            if not (0.0 <= self.min_similarity_score <= 1.0):
                logger.error(f"min_similarity_score must be between 0 and 1, got {self.min_similarity_score}")
                return False
        else:
            # 生产模式的严格验证
            if not (0.1 <= self.swap_trigger_distance <= 2.0):
                logger.error(f"swap_trigger_distance must be between 0.1 and 2.0, got {self.swap_trigger_distance}")
                return False
                
            if not (300 <= self.swap_duration <= 7200):
                logger.error(f"swap_duration must be between 300 and 7200, got {self.swap_duration}")
                return False
                
            if not (0.0 <= self.min_similarity_score <= 1.0):
                logger.error(f"min_similarity_score must be between 0 and 1, got {self.min_similarity_score}")
                return False
            
        return True
        
    def generate_patches(self, 
                        chunk_iter: Iterator[pd.DataFrame],
                        config: Dict[str, Any]) -> Iterator[DataPatch]:
        """生成身份交换攻击补丁"""
        
        for chunk_id, chunk in enumerate(chunk_iter):
            if chunk.empty:
                continue
                
            current_time = chunk['BaseDateTime'].iloc[0] if len(chunk) > 0 else datetime.utcnow()
            # 1. 定期分析新的配对机会
            if self._should_analyze_pairs(current_time):
                new_pairs = self._find_swap_candidates(chunk, current_time)
                self.pending_pairs.extend(new_pairs)
                self.last_analysis_time = current_time
                
            # 2. 检查接近触发条件
            self._check_proximity_triggers(chunk, current_time)
            
            # 3. 生成身份交换补丁
            modifications = self._generate_swap_modifications(chunk, current_time)
            if modifications is not None and not modifications.empty:
                yield DataPatch(
                    patch_type=PatchType.MODIFY,
                    chunk_id=chunk_id,
                    modifications=modifications
                )
                
            # 4. 处理交换超时和异常
            self._check_swap_timeouts(current_time)
            
    def _should_analyze_pairs(self, current_time: datetime) -> bool:
        """判断是否需要分析新的配对"""
        if self.last_analysis_time is None:
            return True
        return (current_time - self.last_analysis_time).total_seconds() >= self.analysis_interval
        
    def _find_swap_candidates(self, chunk: pd.DataFrame, current_time: datetime) -> List[SwapPair]:
        """寻找潜在的交换对"""
        # 筛选合适的船只
        suitable_vessels = self._filter_suitable_vessels(chunk)
        
        if len(suitable_vessels) < 2:
            return []
            
        new_pairs = []
        
        # 计算所有可能的配对
        for i, vessel_a in suitable_vessels.iterrows():
            for j, vessel_b in suitable_vessels.iterrows():
                if i == j:  # 避免船只与自己配对
                    continue
                    
                mmsi_a = int(vessel_a['MMSI'])
                mmsi_b = int(vessel_b['MMSI'])
                
                # 避免船只与自己配对（检查MMSI）
                if mmsi_a == mmsi_b:
                    continue
                    
                # 避免重复配对（A-B 和 B-A）
                if mmsi_a > mmsi_b:
                    continue
                
                # 检查是否已经配对
                if mmsi_a in self.paired_vessels or mmsi_b in self.paired_vessels:
                    continue
                    
                # 计算相似度
                similarity = self._calculate_similarity(vessel_a, vessel_b)
                if similarity < self.min_similarity_score:
                    continue
                    
                # 预测CPA
                cpa_time, cpa_distance = self._predict_cpa(vessel_a, vessel_b, current_time)
                
                # 创建配对
                if cpa_distance is not None and cpa_distance <= self.swap_trigger_distance * 2:
                    pair = SwapPair(
                        pair_id=str(uuid.uuid4()),
                        vessel_a_mmsi=mmsi_a,
                        vessel_b_mmsi=mmsi_b,
                        predicted_cpa_time=cpa_time,
                        predicted_cpa_distance=cpa_distance,
                        swap_trigger_distance=self.swap_trigger_distance,
                        swap_status='pending',
                        swap_start_time=None,
                        swap_end_time=None,
                        swap_duration=self.swap_duration,
                        original_a_identity={},
                        original_b_identity={},
                        similarity_score=similarity,
                        last_update_time=current_time
                    )
                    
                    new_pairs.append(pair)
                    self.paired_vessels.add(mmsi_a)
                    self.paired_vessels.add(mmsi_b)
                    
                    logger.info(f"创建交换对: {mmsi_a} <-> {mmsi_b}, 相似度: {similarity:.2f}, 预计CPA: {cpa_distance:.2f}海里")
                    
        return new_pairs
        
    def _filter_suitable_vessels(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """筛选合适的船只"""
        filtered = chunk[
            (chunk['VesselType'].isin(self.target_vessel_types)) &
            (chunk['SOG'] >= self.min_speed) &
            (chunk['SOG'] <= self.max_speed) &
            (chunk['LAT'].notna()) &
            (chunk['LON'].notna()) &
            (chunk['VesselName'].notna()) &
            (chunk['CallSign'].notna())
        ].copy()
        
        return filtered
        
    def _calculate_similarity(self, vessel_a: pd.Series, vessel_b: pd.Series) -> float:
        """计算两船相似度"""
        # 船型相似度(40%)
        type_diff = abs(vessel_a['VesselType'] - vessel_b['VesselType'])
        type_similarity = 1.0 - min(type_diff / self.vessel_type_tolerance, 1.0)
        
        # 速度相似度(30%)
        speed_diff = abs(vessel_a['SOG'] - vessel_b['SOG'])
        max_speed_diff = self.max_speed - self.min_speed
        speed_similarity = 1.0 - min(speed_diff / max_speed_diff, 1.0)
        
        # 航向相似度(30%) - 考虑相向或同向
        cog_a = vessel_a['COG']
        cog_b = vessel_b['COG']
        
        # 计算航向差
        heading_diff = abs(cog_a - cog_b)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff
            
        # 相向(180度差)或同向(0度差)都给高分
        if heading_diff < 30 or heading_diff > 150:
            heading_similarity = 1.0
        else:
            heading_similarity = 0.5
            
        # 综合相似度
        similarity = (0.4 * type_similarity + 
                     0.3 * speed_similarity + 
                     0.3 * heading_similarity)
        
        return similarity
        
    def _predict_cpa(self, vessel_a: pd.Series, vessel_b: pd.Series, 
                    current_time: datetime) -> Tuple[Optional[datetime], Optional[float]]:
        """预测最近会遇点(CPA) - 使用正确的坐标系统"""
        # 位置 (degrees)
        lat1, lon1 = vessel_a['LAT'], vessel_a['LON']
        lat2, lon2 = vessel_b['LAT'], vessel_b['LON']
        
        # 速度和航向
        speed1 = vessel_a['SOG']  # knots
        course1 = math.radians(vessel_a['COG'])
        speed2 = vessel_b['SOG']  # knots
        course2 = math.radians(vessel_b['COG'])
        
        # 转换为平面坐标系 (使用Mercator投影近似)
        # 1度经度 ≈ 60 * cos(lat) 海里
        # 1度纬度 ≈ 60 海里
        avg_lat = math.radians((lat1 + lat2) / 2)
        nm_per_deg_lon = 60 * math.cos(avg_lat)
        nm_per_deg_lat = 60
        
        # 位置差异 (海里)
        dx = (lon2 - lon1) * nm_per_deg_lon
        dy = (lat2 - lat1) * nm_per_deg_lat
        
        # 速度向量 (海里/小时)
        # 北为正Y，东为正X
        vx1 = speed1 * math.sin(course1)  # 东向分量
        vy1 = speed1 * math.cos(course1)  # 北向分量
        vx2 = speed2 * math.sin(course2)
        vy2 = speed2 * math.cos(course2)
        
        # 相对速度
        dvx = vx2 - vx1
        dvy = vy2 - vy1
        
        # 计算CPA
        dv_squared = dvx**2 + dvy**2
        
        if dv_squared < 0.01:  # 相对静止
            current_distance = math.sqrt(dx**2 + dy**2)
            return current_time, current_distance
            
        # CPA时间(小时)
        t_cpa = -(dx * dvx + dy * dvy) / dv_squared
        
        if t_cpa < 0:  # 已经开始远离
            current_distance = math.sqrt(dx**2 + dy**2)
            return current_time, current_distance
            
        # CPA位置
        x_cpa = dx + dvx * t_cpa
        y_cpa = dy + dvy * t_cpa
        
        # CPA距离
        cpa_distance = math.sqrt(x_cpa**2 + y_cpa**2)
        
        # CPA时间
        cpa_time = current_time + timedelta(hours=t_cpa)
        
        return cpa_time, cpa_distance
        
    def _check_proximity_triggers(self, chunk: pd.DataFrame, current_time: datetime) -> None:
        """检查接近触发条件"""
        triggered_pairs = []
        
        for pair in self.pending_pairs:
            # 获取两船当前数据
            vessel_a = chunk[chunk['MMSI'] == pair.vessel_a_mmsi]
            vessel_b = chunk[chunk['MMSI'] == pair.vessel_b_mmsi]
            
            if vessel_a.empty or vessel_b.empty:
                continue
                
            # 计算当前距离
            distance = self._calculate_distance(
                (vessel_a['LAT'].iloc[0], vessel_a['LON'].iloc[0]),
                (vessel_b['LAT'].iloc[0], vessel_b['LON'].iloc[0])
            )
            
            # 检查触发条件
            if distance <= pair.swap_trigger_distance:
                if len(self.active_pairs) < self.max_active_swaps:
                    logger.info(f"触发身份交换: {pair.vessel_a_mmsi} <-> {pair.vessel_b_mmsi}, 距离={distance:.2f}海里")
                    self._execute_swap(pair, vessel_a.iloc[0], vessel_b.iloc[0], current_time)
                    triggered_pairs.append(pair)
                else:
                    logger.warning(f"达到最大并发交换数限制，跳过配对 {pair.vessel_a_mmsi} <-> {pair.vessel_b_mmsi}")
                    
        # 从待执行列表中移除已触发的配对
        for pair in triggered_pairs:
            self.pending_pairs.remove(pair)
            
    def _execute_swap(self, pair: SwapPair, vessel_a: pd.Series, 
                     vessel_b: pd.Series, current_time: datetime) -> None:
        """执行身份交换"""
        # 提取原始身份信息
        pair.original_a_identity = {
            field: vessel_a[field] for field in self.swap_fields if field in vessel_a
        }
        pair.original_b_identity = {
            field: vessel_b[field] for field in self.swap_fields if field in vessel_b
        }
        
        # 更新交换状态
        pair.swap_status = 'active'
        pair.swap_start_time = current_time
        pair.swap_end_time = current_time + timedelta(seconds=pair.swap_duration)
        pair.last_update_time = current_time
        
        # 添加到活跃交换
        self.active_pairs[pair.vessel_a_mmsi] = pair
        self.active_pairs[pair.vessel_b_mmsi] = pair
        
        logger.info(f"执行身份交换: {pair.vessel_a_mmsi} <-> {pair.vessel_b_mmsi}, 持续时间: {pair.swap_duration}秒")
        
    def _generate_swap_modifications(self, chunk: pd.DataFrame, 
                                   current_time: datetime) -> Optional[pd.DataFrame]:
        """生成身份交换修改"""
        if not self.active_pairs:
            return None
            
        # 筛选需要修改的记录
        active_mmsis = set(self.active_pairs.keys())
        mask = chunk['MMSI'].isin(active_mmsis)
        
        if not mask.any():
            return None
            
        # 创建修改数据框
        modifications = pd.DataFrame(index=chunk.index[mask])
        
        for idx in modifications.index:
            mmsi = int(chunk.loc[idx, 'MMSI'])
            if mmsi not in self.active_pairs:
                continue
                
            pair = self.active_pairs[mmsi]
            
            # 确定交换方向
            if mmsi == pair.vessel_a_mmsi:
                # A船使用B船身份
                swap_identity = pair.original_b_identity
            else:
                # B船使用A船身份
                swap_identity = pair.original_a_identity
                
            # 应用身份交换
            for field, value in swap_identity.items():
                if field in chunk.columns:
                    modifications.loc[idx, field] = value
                    
        return modifications if not modifications.empty else None
        
    def _check_swap_timeouts(self, current_time: datetime) -> None:
        """检查交换超时"""
        expired_mmsis = []
        
        for mmsi, pair in self.active_pairs.items():
            if current_time >= pair.swap_end_time:
                # 标记为完成
                pair.swap_status = 'completed'
                pair.last_update_time = current_time
                self.completed_pairs.append(pair)
                expired_mmsis.append(mmsi)
                
        # 清理过期的活跃交换
        for mmsi in expired_mmsis:
            if mmsi in self.active_pairs:
                pair = self.active_pairs[mmsi]
                del self.active_pairs[mmsi]
                
                # 同时清理配对的另一艘船
                if mmsi == pair.vessel_a_mmsi and pair.vessel_b_mmsi in self.active_pairs:
                    del self.active_pairs[pair.vessel_b_mmsi]
                elif mmsi == pair.vessel_b_mmsi and pair.vessel_a_mmsi in self.active_pairs:
                    del self.active_pairs[pair.vessel_a_mmsi]
                    
                # 从已配对集合中移除
                self.paired_vessels.discard(pair.vessel_a_mmsi)
                self.paired_vessels.discard(pair.vessel_b_mmsi)
                
                logger.info(f"身份交换完成: {pair.vessel_a_mmsi} <-> {pair.vessel_b_mmsi}")
                
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
        active_swaps = []
        for mmsi, pair in self.active_pairs.items():
            if mmsi == pair.vessel_a_mmsi:  # 避免重复
                active_swaps.append({
                    'pair_id': pair.pair_id,
                    'vessel_a_mmsi': pair.vessel_a_mmsi,
                    'vessel_b_mmsi': pair.vessel_b_mmsi,
                    'swap_start_time': pair.swap_start_time.isoformat() if pair.swap_start_time else None,
                    'swap_end_time': pair.swap_end_time.isoformat() if pair.swap_end_time else None,
                    'similarity_score': pair.similarity_score
                })
                
        return {
            'pending_swaps': len(self.pending_pairs),
            'active_swaps': len(active_swaps),
            'completed_swaps': len(self.completed_pairs),
            'attack_parameters': {
                'swap_trigger_distance': self.swap_trigger_distance,
                'swap_duration': self.swap_duration,
                'min_similarity_score': self.min_similarity_score
            },
            'target_criteria': {
                'vessel_types': self.target_vessel_types,
                'speed_range': (self.min_speed, self.max_speed)
            },
            'active_swap_details': active_swaps
        }