#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S3 - 幽灵渔船群攻击
在指定海域生成协调的虚假渔船群，通过同步的出现和消失行为干扰监视系统
"""

import logging
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from datetime import datetime, timedelta

from core.attack_plugin_v2 import StreamingAttackPlugin, DataPatch, PatchType

logger = logging.getLogger(__name__)


class GhostSwarmAttack(StreamingAttackPlugin):
    """幽灵渔船群攻击插件"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """初始化攻击插件"""
        super().__init__(params or {})
        
        # 群体参数
        self.swarm_size = 8
        self.swarm_area_radius = 2.0  # 海里
        self.vessel_separation = 0.15  # 最小船间距（海里）
        
        # 运动参数
        self.base_speed = 4.5  # 节
        self.speed_variation = 1.5
        self.heading_variation = 45.0
        self.update_interval = 45  # 秒
        
        # 时间参数
        self.appearance_duration = 2400  # 40分钟
        self.disappearance_duration = 1200  # 20分钟
        self.sync_probability = 0.75
        
        # 中心位置
        self.swarm_area_center = (30.8, -88.5)
        
        # 内部状态
        self.ghost_vessels = []
        self.vessel_states = {}
        self.swarm_visible = True
        self.next_state_change = None
        self.last_update_time = None
        self.spatial_grid = {}
        self.generated_messages = []
        
        # OU过程参数
        self.ou_mu = 0.0
        self.ou_theta = 0.15
        self.ou_sigma = 0.3
        self.ou_dt = self.update_interval / 3600.0
        
    def validate_params(self) -> bool:
        """验证参数"""
        return True
        
    def get_impact_type(self) -> str:
        """获取影响类型"""
        return "false_target"
        
    def generate_patches(self, 
                        chunk_iter: Iterator[pd.DataFrame],
                        config: Dict[str, Any]) -> Iterator[DataPatch]:
        """生成攻击补丁"""
        # 初始化配置
        if 'swarm_config' in self.params:
            swarm_config = self.params['swarm_config']
            self.swarm_size = swarm_config.get('size', 8)
            self.swarm_area_radius = swarm_config.get('area_radius', 2.0)
            self.vessel_separation = swarm_config.get('min_separation', 0.15)
            
        if 'motion_config' in self.params:
            motion_config = self.params['motion_config']
            self.base_speed = motion_config.get('base_speed', 4.5)
            self.speed_variation = motion_config.get('speed_variation', 1.5)
            self.heading_variation = motion_config.get('heading_variation', 45.0)
            self.update_interval = motion_config.get('update_interval', 45)
            
        if 'timing_config' in self.params:
            timing_config = self.params['timing_config']
            self.appearance_duration = timing_config.get('appearance_duration', 2400)
            self.disappearance_duration = timing_config.get('disappearance_duration', 1200)
            self.sync_probability = timing_config.get('sync_probability', 0.75)
            
        if 'center_position' in self.params:
            center = self.params['center_position']
            self.swarm_area_center = (center['lat'], center['lon'])
            
        # 初始化幽灵船只群
        self._initialize_swarm()
        
        # 设置时间窗口
        start_time = config.get('start_time')
        end_time = config.get('end_time')
        
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        current_time = start_time
        self.next_state_change = current_time + timedelta(seconds=self.appearance_duration)
        
        # 生成整个时间窗口的幽灵船只消息
        ghost_messages = []
        
        while current_time < end_time:
            # 检查是否需要改变群体状态
            if current_time >= self.next_state_change:
                self._update_swarm_visibility(current_time)
                
            # 更新位置并生成消息
            if self.swarm_visible and any(state['visible'] for state in self.vessel_states.values()):
                if self.last_update_time is None or \
                   (current_time - self.last_update_time).total_seconds() >= self.update_interval:
                    self._update_all_positions(current_time)
                    messages = self._generate_messages(current_time)
                    ghost_messages.extend(messages)
                    self.last_update_time = current_time
                    
            current_time += timedelta(seconds=self.update_interval)
            
        # 转换为DataFrame
        if ghost_messages:
            ghost_df = pd.DataFrame(ghost_messages)
            
            # 按时间排序
            ghost_df = ghost_df.sort_values('BaseDateTime').reset_index(drop=True)
            
            # 生成INSERT补丁
            yield DataPatch(
                patch_type=PatchType.INSERT,
                new_records=ghost_df
            )
            
        logger.info(f"Generated {len(ghost_messages)} ghost vessel messages")
        
    def _initialize_swarm(self) -> None:
        """初始化幽灵渔船群"""
        logger.info(f"初始化幽灵渔船群: {self.swarm_size}艘")
        
        self.ghost_vessels = []
        self.vessel_states = {}
        
        # 生成每艘船的初始状态
        for i in range(self.swarm_size):
            mmsi = 900000000 + random.randint(1000, 9999) * 100 + i
            vessel = {
                'mmsi': mmsi,
                'vessel_name': f'FISHING_{i+1:02d}',
                'vessel_type': 30,  # 渔船
                'call_sign': f'GHOST{i+1:02d}',
                'length': random.randint(20, 40),
                'width': random.randint(6, 10),
                'draft': random.uniform(2.0, 4.0)
            }
            self.ghost_vessels.append(vessel)
            
            # 初始化船只状态
            position = self._generate_initial_position()
            self.vessel_states[mmsi] = {
                'position': position,
                'speed': self.base_speed + random.uniform(-self.speed_variation, self.speed_variation),
                'heading': random.uniform(0, 360),
                'course': random.uniform(0, 360),
                'visible': True,
                'ou_state': 0.0,  # OU过程状态
                'route_points': [position]
            }
            
    def _generate_initial_position(self) -> Tuple[float, float]:
        """生成初始位置，确保满足最小间距要求"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # 在活动区域内随机生成位置
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, self.swarm_area_radius)
            
            # 限制纬度范围以避免极点问题
            safe_center_lat = np.clip(self.swarm_area_center[0], -85, 85)
            lat_offset = distance * np.cos(angle) / 60.0  # 1海里约等于1/60度
            lon_offset = distance * np.sin(angle) / (60.0 * np.cos(np.radians(safe_center_lat)))
            
            lat = self.swarm_area_center[0] + lat_offset
            lon = self.swarm_area_center[1] + lon_offset
            
            # 检查与其他船只的距离
            position = (lat, lon)
            if self._check_minimum_separation(position):
                # 添加到空间索引
                grid_key = self._get_grid_key(position)
                if grid_key not in self.spatial_grid:
                    self.spatial_grid[grid_key] = []
                self.spatial_grid[grid_key].append(position)
                return position
                
        # 如果无法找到合适位置，返回中心附近的位置
        logger.warning("无法找到满足间距要求的位置，使用默认位置")
        return self.swarm_area_center
        
    def _check_minimum_separation(self, position: Tuple[float, float]) -> bool:
        """检查与其他船只的最小间距"""
        for mmsi, state in self.vessel_states.items():
            other_pos = state['position']
            distance = self._calculate_distance(position, other_pos)
            if distance < self.vessel_separation:
                return False
        return True
        
    def _get_grid_key(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """获取空间网格键"""
        # 每个网格单元大约0.5海里
        lat_key = int(position[0] * 120)  # 0.5/60 ≈ 1/120
        lon_key = int(position[1] * 120)
        return (lat_key, lon_key)
        
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间距离（海里）"""
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 3440.065 * c  # 地球半径(海里)
        
    def _update_swarm_visibility(self, current_time: datetime) -> None:
        """更新群体可见性"""
        if self.swarm_visible:
            # 准备消失
            self.swarm_visible = False
            self.next_state_change = current_time + timedelta(seconds=self.disappearance_duration)
            logger.info(f"渔船群消失，下次出现时间: {self.next_state_change}")
            
            # 根据同步概率决定是否所有船只同时消失
            if random.random() < self.sync_probability:
                # 同步消失
                for mmsi in self.vessel_states:
                    self.vessel_states[mmsi]['visible'] = False
            else:
                # 随机消失
                for mmsi in self.vessel_states:
                    self.vessel_states[mmsi]['visible'] = random.random() < 0.3
        else:
            # 准备出现
            self.swarm_visible = True
            self.next_state_change = current_time + timedelta(seconds=self.appearance_duration)
            logger.info(f"渔船群出现，下次消失时间: {self.next_state_change}")
            
            # 根据同步概率决定是否所有船只同时出现
            if random.random() < self.sync_probability:
                # 同步出现
                for mmsi in self.vessel_states:
                    self.vessel_states[mmsi]['visible'] = True
            else:
                # 随机出现
                for mmsi in self.vessel_states:
                    self.vessel_states[mmsi]['visible'] = random.random() < 0.7
                    
    def _update_all_positions(self, current_time: datetime) -> None:
        """更新所有船只位置"""
        # 清空空间索引
        self.spatial_grid.clear()
        
        # 更新每艘船的位置
        for vessel in self.ghost_vessels:
            mmsi = vessel['mmsi']
            state = self.vessel_states[mmsi]
            
            if state['visible']:
                # 更新位置
                new_position = self._update_vessel_position(mmsi, state)
                state['position'] = new_position
                state['route_points'].append(new_position)
                
                # 更新空间索引
                grid_key = self._get_grid_key(new_position)
                if grid_key not in self.spatial_grid:
                    self.spatial_grid[grid_key] = []
                self.spatial_grid[grid_key].append((mmsi, new_position))
                
                # 限制轨迹长度
                if len(state['route_points']) > 100:
                    state['route_points'] = state['route_points'][-100:]
                    
    def _update_vessel_position(self, mmsi: str, state: Dict) -> Tuple[float, float]:
        """使用随机游走更新单个船只位置"""
        # 当前位置
        lat, lon = state['position']
        
        # 更新OU过程
        dt = self.update_interval / 3600.0  # 转换为小时
        state['ou_state'] += self.ou_theta * (self.ou_mu - state['ou_state']) * dt + \
                            self.ou_sigma * np.sqrt(dt) * random.gauss(0, 1)
        
        # 更新航向
        heading_change = state['ou_state'] * self.heading_variation
        state['heading'] = (state['heading'] + heading_change) % 360
        state['course'] = state['heading']  # 简化：航向等于航迹向
        
        # 更新速度
        speed_factor = 1.0 + random.gauss(0, 0.1)
        state['speed'] = np.clip(
            self.base_speed * speed_factor,
            self.base_speed - self.speed_variation,
            self.base_speed + self.speed_variation
        )
        
        # 计算新位置
        distance = state['speed'] * dt  # 海里
        heading_rad = np.radians(state['heading'])
        
        # 限制纬度范围以避免极点问题
        safe_lat = np.clip(lat, -85, 85)
        
        lat_new = lat + (distance / 60.0) * np.cos(heading_rad)
        lon_new = lon + (distance / (60.0 * np.cos(np.radians(safe_lat)))) * np.sin(heading_rad)
        
        # 检查边界和碰撞
        new_position = self._constrain_position((lat_new, lon_new), mmsi)
        
        return new_position
        
    def _constrain_position(self, position: Tuple[float, float], mmsi: str) -> Tuple[float, float]:
        """约束位置在活动区域内并避免碰撞"""
        lat, lon = position
        
        # 检查是否超出活动区域
        distance_from_center = self._calculate_distance(position, self.swarm_area_center)
        if distance_from_center > self.swarm_area_radius:
            # 将位置拉回到边界内
            angle = np.arctan2(lon - self.swarm_area_center[1], lat - self.swarm_area_center[0])
            safe_center_lat = np.clip(self.swarm_area_center[0], -85, 85)
            lat = self.swarm_area_center[0] + self.swarm_area_radius * 0.9 * np.cos(angle) / 60.0
            lon = self.swarm_area_center[1] + self.swarm_area_radius * 0.9 * np.sin(angle) / (60.0 * np.cos(np.radians(safe_center_lat)))
            
        # 检查碰撞
        final_position = self._check_collision_and_adjust((lat, lon), mmsi)
        
        return final_position
        
    def _check_collision_and_adjust(self, position: Tuple[float, float], mmsi: str) -> Tuple[float, float]:
        """检查碰撞并调整位置"""
        # 检查附近的网格单元
        grid_key = self._get_grid_key(position)
        nearby_vessels = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                check_key = (grid_key[0] + di, grid_key[1] + dj)
                if check_key in self.spatial_grid:
                    nearby_vessels.extend(self.spatial_grid[check_key])
                    
        # 检查与附近船只的距离
        for other_mmsi, other_pos in nearby_vessels:
            if other_mmsi != mmsi:
                distance = self._calculate_distance(position, other_pos)
                if distance < self.vessel_separation:
                    # 调整位置以保持最小间距
                    angle = np.arctan2(
                        position[1] - other_pos[1],
                        position[0] - other_pos[0]
                    )
                    safe_other_lat = np.clip(other_pos[0], -85, 85)
                    lat = other_pos[0] + self.vessel_separation * 1.1 * np.cos(angle) / 60.0
                    lon = other_pos[1] + self.vessel_separation * 1.1 * np.sin(angle) / (60.0 * np.cos(np.radians(safe_other_lat)))
                    position = (lat, lon)
                    
        return position
        
    def _generate_messages(self, current_time: datetime) -> List[Dict]:
        """生成AIS消息"""
        messages = []
        
        for vessel in self.ghost_vessels:
            mmsi = vessel['mmsi']
            state = self.vessel_states[mmsi]
            
            if state['visible']:
                # 生成位置报告（消息类型1）
                position_msg = {
                    'BaseDateTime': current_time,
                    'MMSI': mmsi,
                    'LAT': state['position'][0],
                    'LON': state['position'][1],
                    'SOG': state['speed'],
                    'COG': state['course'],
                    'Heading': state['heading'],
                    'VesselName': vessel['vessel_name'],
                    'IMO': '',
                    'CallSign': vessel['call_sign'],
                    'VesselType': vessel['vessel_type'],
                    'Status': 0,  # 正在使用引擎航行
                    'Length': vessel['length'],
                    'Width': vessel['width'],
                    'Draft': vessel['draft'],
                    'Cargo': 30,  # 渔船
                    'TransceiverClass': 'A'
                }
                messages.append(position_msg)
                
                # 偶尔生成静态信息（消息类型5）
                if random.random() < 0.05:  # 5%概率
                    static_msg = position_msg.copy()
                    static_msg['MessageType'] = 5
                    messages.append(static_msg)
                    
        return messages
        
    def get_attack_vessels(self) -> List[int]:
        """获取攻击船只列表"""
        return [v['mmsi'] for v in self.ghost_vessels]
        
    def get_swarm_status(self) -> Dict[str, Any]:
        """获取群体状态"""
        visible_count = sum(1 for state in self.vessel_states.values() if state['visible'])
        
        return {
            'total_vessels': len(self.ghost_vessels),
            'visible_vessels': visible_count,
            'swarm_visible': self.swarm_visible,
            'center_position': self.swarm_area_center,
            'area_radius': self.swarm_area_radius,
            'vessel_positions': {
                mmsi: state['position'] 
                for mmsi, state in self.vessel_states.items()
                if state['visible']
            }
        }