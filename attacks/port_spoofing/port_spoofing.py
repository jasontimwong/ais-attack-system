#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S5 - 位置造假掩盖禁运港停靠 (Port Spoofing to Conceal Embargo Port Docking)

该攻击通过伪造AIS位置信息来掩盖船只在禁运港口的停靠行为，
使其在监控系统中显示为正常的海上航行轨迹。
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
class EmbargoPort:
    """禁运港口信息"""
    name: str
    lat: float
    lon: float
    radius: float  # 港口范围半径(海里)
    country: str = ""
    port_type: str = ""


@dataclass
class DockingEvent:
    """停靠事件信息"""
    mmsi: int
    port: EmbargoPort
    start_time: datetime
    end_time: datetime
    original_positions: List[Tuple[float, float, datetime]]  # 原始位置记录
    spoofed_positions: List[Tuple[float, float, datetime]]   # 伪造位置记录
    last_sea_position: Tuple[float, float]  # 进港前最后海上位置
    next_sea_position: Tuple[float, float]  # 出港后首个海上位置


class PortSpoofingAttack(StreamingAttackPlugin):
    """位置造假掩盖禁运港停靠攻击实现"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.attack_type = "port_spoofing"
        self.category = "information_tampering"
        
        # 攻击参数
        self.spoofing_distance = params.get('spoofing_distance', 5.0)  # 伪造距离(海里)
        self.min_docking_time = params.get('min_docking_time', 1800)  # 最小停靠时间(秒)
        self.speed_threshold = params.get('speed_threshold', 3.0)  # 停靠速度阈值(节)
        self.trajectory_smoothness = params.get('trajectory_smoothness', 0.8)  # 轨迹平滑度
        self.detection_radius_multiplier = params.get('detection_radius_multiplier', 1.2)  # 检测半径倍数
        
        logger.info(f"[PortSpoofing] Initialized with parameters:")
        logger.info(f"  - spoofing_distance: {self.spoofing_distance} nm")
        logger.info(f"  - min_docking_time: {self.min_docking_time} seconds ({self.min_docking_time/60:.1f} minutes)")
        logger.info(f"  - speed_threshold: {self.speed_threshold} knots")
        logger.info(f"  - detection_radius_multiplier: {self.detection_radius_multiplier}")
        
        # 禁运港口配置
        self.embargo_ports = self._load_embargo_ports(params.get('embargo_ports', []))
        
        # 攻击状态
        self.active_dockings: Dict[int, DockingEvent] = {}  # MMSI -> DockingEvent
        self.vessel_histories: Dict[int, List[Dict]] = {}  # MMSI -> 历史位置记录
        self.last_analysis_time: Optional[datetime] = None
        
    def _load_embargo_ports(self, ports_config: List[Dict]) -> List[EmbargoPort]:
        """加载禁运港口配置"""
        ports = []
        
        # 默认禁运港口（示例）
        default_ports = [
            {
                'name': 'Embargo_Port_1',
                'lat': 30.7,
                'lon': -88.1,
                'radius': 2.0,
                'country': 'Example',
                'port_type': 'Commercial'
            },
            {
                'name': 'Port_A',
                'lat': 31.2,
                'lon': 121.5,
                'radius': 2.0,
                'country': 'Example',
                'port_type': 'Commercial'
            },
            {
                'name': 'Port_B', 
                'lat': 30.8,
                'lon': 120.9,
                'radius': 1.5,
                'country': 'Example',
                'port_type': 'Oil'
            }
        ]
        
        # 使用配置中的港口或默认港口
        port_configs = ports_config if ports_config else default_ports
        
        for port_config in port_configs:
            port = EmbargoPort(
                name=port_config['name'],
                lat=port_config['lat'],
                lon=port_config['lon'],
                radius=port_config['radius'],
                country=port_config.get('country', ''),
                port_type=port_config.get('port_type', '')
            )
            ports.append(port)
            logger.info(f"加载禁运港口: {port.name} ({port.lat}, {port.lon}), 半径: {port.radius}海里")
        
        return ports
        
    def validate_params(self) -> bool:
        """验证攻击参数"""
        if not (1.0 <= self.spoofing_distance <= 20.0):
            logger.error(f"spoofing_distance must be between 1.0 and 20.0, got {self.spoofing_distance}")
            return False
            
        if not (300 <= self.min_docking_time <= 86400):
            logger.error(f"min_docking_time must be between 300 and 86400, got {self.min_docking_time}")
            return False
            
        if not (0.5 <= self.speed_threshold <= 10.0):
            logger.error(f"speed_threshold must be between 0.5 and 10.0, got {self.speed_threshold}")
            return False
            
        if not self.embargo_ports:
            logger.error("No embargo ports configured")
            return False
            
        return True
        
    def generate_patches(self, 
                        chunk_iter: Iterator[pd.DataFrame],
                        config: Dict[str, Any]) -> Iterator[DataPatch]:
        """生成位置造假攻击补丁"""
        
        for chunk_id, chunk in enumerate(chunk_iter):
            if chunk.empty:
                continue
                
            # 获取当前时间
            current_time = chunk['BaseDateTime'].iloc[0] if len(chunk) > 0 else datetime.utcnow()
            logger.debug(f"[PortSpoofing] Processing chunk {chunk_id} at time {current_time}, chunk size: {len(chunk)}")
            
            # 更新船只历史记录
            self._update_vessel_histories(chunk)
            
            # 分析停靠事件
            if (self.last_analysis_time is None or 
                (current_time - self.last_analysis_time).total_seconds() > 300):
                logger.debug(f"[PortSpoofing] Analyzing docking events at {current_time}")
                self._analyze_docking_events(chunk, current_time)
                self.last_analysis_time = current_time
            else:
                logger.debug(f"[PortSpoofing] Skipping analysis, last analyzed {(current_time - self.last_analysis_time).total_seconds():.1f}s ago")
                
            # 更新活跃停靠事件
            self._update_active_dockings(current_time)
            logger.debug(f"[PortSpoofing] Active dockings count: {len(self.active_dockings)}")
            
            # 生成位置伪造补丁
            if self.active_dockings:
                logger.debug(f"[PortSpoofing] Generating spoofing for {len(self.active_dockings)} active dockings")
                modifications = self._generate_position_spoofing(chunk, current_time)
                if modifications is not None and not modifications.empty:
                    logger.info(f"[PortSpoofing] Generated {len(modifications)} position modifications")
                    yield DataPatch(
                        patch_type=PatchType.MODIFY,
                        chunk_id=chunk_id,
                        modifications=modifications
                    )
                else:
                    logger.debug(f"[PortSpoofing] No modifications generated for this chunk")
            else:
                logger.debug(f"[PortSpoofing] No active dockings to spoof")
                    
    def _update_vessel_histories(self, chunk: pd.DataFrame) -> None:
        """更新船只历史记录"""
        unique_mmsis = chunk['MMSI'].unique()
        logger.debug(f"[PortSpoofing] Updating vessel histories for {len(unique_mmsis)} vessels in chunk")
        
        for idx, row in chunk.iterrows():
            mmsi = int(row['MMSI'])
            
            if mmsi not in self.vessel_histories:
                self.vessel_histories[mmsi] = []
                logger.debug(f"[PortSpoofing] Creating new history for MMSI {mmsi}")
                
            # 添加当前位置记录
            position_record = {
                'timestamp': row['BaseDateTime'],
                'lat': row['LAT'],
                'lon': row['LON'],
                'sog': row['SOG'],
                'cog': row['COG']
            }
            
            self.vessel_histories[mmsi].append(position_record)
            
            # 限制历史记录长度
            if len(self.vessel_histories[mmsi]) > 100:
                self.vessel_histories[mmsi] = self.vessel_histories[mmsi][-100:]
                
        # Log detailed info for vessel 636092000 if present
        if 636092000 in self.vessel_histories:
            history = self.vessel_histories[636092000]
            if history:
                latest = history[-1]
                logger.debug(f"[PortSpoofing] MMSI 636092000: History length = {len(history)}, "
                           f"Latest position = ({latest['lat']:.4f}, {latest['lon']:.4f}), "
                           f"SOG = {latest['sog']:.1f}, Time = {latest['timestamp']}")
                
    def _analyze_docking_events(self, chunk: pd.DataFrame, current_time: datetime) -> None:
        """分析停靠事件"""
        logger.debug(f"[PortSpoofing] Analyzing docking events for {len(self.vessel_histories)} vessels")
        
        for mmsi, history in self.vessel_histories.items():
            if len(history) < 10:  # 需要足够的历史数据
                logger.debug(f"[PortSpoofing] MMSI {mmsi}: Insufficient history ({len(history)} records)")
                continue
                
            # 检查是否在禁运港停靠
            current_pos = history[-1]
            logger.debug(f"[PortSpoofing] MMSI {mmsi}: Current position ({current_pos['lat']:.4f}, {current_pos['lon']:.4f}), SOG: {current_pos['sog']:.1f}")
            
            port = self._find_port_in_range(current_pos['lat'], current_pos['lon'])
            
            if port:
                logger.debug(f"[PortSpoofing] MMSI {mmsi}: Found near port {port.name} ({port.lat}, {port.lon})")
                
                if mmsi not in self.active_dockings:
                    # 检查是否满足停靠条件
                    logger.debug(f"[PortSpoofing] MMSI {mmsi}: Checking docking conditions at port {port.name}")
                    if self._is_docking_at_port(mmsi, port, current_time):
                        logger.info(f"[PortSpoofing] MMSI {mmsi}: Docking conditions met at port {port.name}")
                        self._start_docking_event(mmsi, port, current_time)
                    else:
                        logger.debug(f"[PortSpoofing] MMSI {mmsi}: Docking conditions NOT met at port {port.name}")
                else:
                    logger.debug(f"[PortSpoofing] MMSI {mmsi}: Already has active docking event")
            else:
                logger.debug(f"[PortSpoofing] MMSI {mmsi}: Not near any embargo port")
                    
    def _find_port_in_range(self, lat: float, lon: float) -> Optional[EmbargoPort]:
        """查找范围内的禁运港口"""
        logger.debug(f"[PortSpoofing] Finding port near ({lat:.4f}, {lon:.4f})")
        
        for port in self.embargo_ports:
            distance = self._calculate_distance((lat, lon), (port.lat, port.lon))
            detection_radius = port.radius * self.detection_radius_multiplier
            
            logger.debug(f"[PortSpoofing] Port {port.name} ({port.lat}, {port.lon}): "
                        f"distance = {distance:.2f} nm, detection radius = {detection_radius:.2f} nm")
            
            if distance <= detection_radius:
                logger.info(f"[PortSpoofing] Found vessel within range of embargo port {port.name}")
                return port
                
        logger.debug(f"[PortSpoofing] No embargo port found near position")
        return None
        
    def _is_docking_at_port(self, mmsi: int, port: EmbargoPort, current_time: datetime) -> bool:
        """判断是否在港口停靠"""
        history = self.vessel_histories[mmsi]
        logger.debug(f"[PortSpoofing] MMSI {mmsi}: Checking docking conditions with {len(history)} history records")
        
        # 检查最近的位置记录
        recent_positions = [h for h in history[-20:] 
                          if (current_time - h['timestamp']).total_seconds() <= 1800]
        
        logger.debug(f"[PortSpoofing] MMSI {mmsi}: Recent positions within 30 min: {len(recent_positions)}")
        
        if len(recent_positions) < 5:
            logger.debug(f"[PortSpoofing] MMSI {mmsi}: Not enough recent positions (need 5, have {len(recent_positions)})")
            return False
            
        # 检查速度条件
        low_speed_count = sum(1 for pos in recent_positions 
                             if pos['sog'] <= self.speed_threshold)
        speed_ratio = low_speed_count / len(recent_positions)
        
        logger.debug(f"[PortSpoofing] MMSI {mmsi}: Low speed positions: {low_speed_count}/{len(recent_positions)} "
                    f"(ratio: {speed_ratio:.2f}, threshold: 0.7, speed_threshold: {self.speed_threshold} knots)")
        
        if low_speed_count < len(recent_positions) * 0.7:
            logger.debug(f"[PortSpoofing] MMSI {mmsi}: Failed speed condition - not enough low speed records")
            return False
            
        # 检查是否在港口范围内停留足够时间
        port_positions = []
        for pos in recent_positions:
            distance = self._calculate_distance(
                (pos['lat'], pos['lon']),
                (port.lat, port.lon)
            )
            if distance <= port.radius:
                port_positions.append(pos)
                logger.debug(f"[PortSpoofing] MMSI {mmsi}: Position at {pos['timestamp']} is {distance:.2f} nm from port (within {port.radius} nm)")
            else:
                logger.debug(f"[PortSpoofing] MMSI {mmsi}: Position at {pos['timestamp']} is {distance:.2f} nm from port (outside {port.radius} nm)")
                
        logger.debug(f"[PortSpoofing] MMSI {mmsi}: Positions within port radius: {len(port_positions)}")
        
        if not port_positions:
            logger.debug(f"[PortSpoofing] MMSI {mmsi}: No positions found within port radius")
            return False
            
        # 检查停留时间
        time_span = (port_positions[-1]['timestamp'] - port_positions[0]['timestamp']).total_seconds()
        
        logger.debug(f"[PortSpoofing] MMSI {mmsi}: Time span in port: {time_span:.0f}s "
                    f"(required: {self.min_docking_time}s)")
        
        result = time_span >= self.min_docking_time
        if result:
            logger.info(f"[PortSpoofing] MMSI {mmsi}: Docking conditions satisfied! Time in port: {time_span:.0f}s")
        else:
            logger.debug(f"[PortSpoofing] MMSI {mmsi}: Not enough time in port yet")
            
        return result
        
    def _start_docking_event(self, mmsi: int, port: EmbargoPort, current_time: datetime) -> None:
        """开始停靠事件"""
        history = self.vessel_histories[mmsi]
        
        # 找到进港前的最后海上位置
        last_sea_pos = None
        for pos in reversed(history):
            distance = self._calculate_distance(
                (pos['lat'], pos['lon']),
                (port.lat, port.lon)
            )
            if distance > port.radius * 1.5:
                last_sea_pos = (pos['lat'], pos['lon'])
                break
                
        if not last_sea_pos:
            # 如果找不到，生成一个合理的海上位置
            last_sea_pos = self._generate_sea_position(port, 'before')
            
        # 创建停靠事件
        docking_event = DockingEvent(
            mmsi=mmsi,
            port=port,
            start_time=current_time,
            end_time=current_time + timedelta(hours=8),  # 预估停靠8小时
            original_positions=[],
            spoofed_positions=[],
            last_sea_position=last_sea_pos,
            next_sea_position=self._generate_sea_position(port, 'after')
        )
        
        self.active_dockings[mmsi] = docking_event
        logger.info(f"开始停靠事件: 船只 {mmsi} 在港口 {port.name}")
        
    def _generate_sea_position(self, port: EmbargoPort, direction: str) -> Tuple[float, float]:
        """生成海上位置"""
        # 在港口外围生成位置
        angle = random.uniform(0, 2 * math.pi)
        distance = self.spoofing_distance
        
        # 根据方向调整角度
        if direction == 'before':
            angle += math.pi  # 来港方向
        elif direction == 'after':
            pass  # 离港方向
            
        # 计算位置偏移
        lat_offset = distance * math.cos(angle) / 60.0
        lon_offset = distance * math.sin(angle) / (60.0 * math.cos(math.radians(port.lat)))
        
        return (port.lat + lat_offset, port.lon + lon_offset)
        
    def _update_active_dockings(self, current_time: datetime) -> None:
        """更新活跃停靠事件"""
        expired_mmsis = []
        
        for mmsi, docking in self.active_dockings.items():
            # 检查是否应该结束停靠
            if self._should_end_docking(mmsi, docking, current_time):
                expired_mmsis.append(mmsi)
                
        # 清理过期的停靠事件
        for mmsi in expired_mmsis:
            docking = self.active_dockings.pop(mmsi)
            logger.info(f"结束停靠事件: 船只 {mmsi} 离开港口 {docking.port.name}")
            
    def _should_end_docking(self, mmsi: int, docking: DockingEvent, current_time: datetime) -> bool:
        """判断是否应该结束停靠"""
        # 检查是否已经离开港口
        if mmsi not in self.vessel_histories:
            return False
            
        history = self.vessel_histories[mmsi]
        if not history:
            return False
            
        # 检查最近位置是否还在港口范围内
        recent_pos = history[-1]
        distance = self._calculate_distance(
            (recent_pos['lat'], recent_pos['lon']),
            (docking.port.lat, docking.port.lon)
        )
        
        # 如果距离港口超过检测半径且速度较高，认为已离港
        if (distance > docking.port.radius * self.detection_radius_multiplier and
            recent_pos['sog'] > self.speed_threshold * 2):
            return True
            
        # 检查是否超过最大停靠时间
        max_docking_time = 24 * 3600  # 24小时
        if (current_time - docking.start_time).total_seconds() > max_docking_time:
            return True
            
        return False
        
    def _generate_position_spoofing(self, chunk: pd.DataFrame, current_time: datetime) -> Optional[pd.DataFrame]:
        """生成位置伪造"""
        if not self.active_dockings:
            logger.debug(f"[PortSpoofing] No active dockings to generate spoofing")
            return None
            
        # 筛选需要伪造的记录
        docking_mmsis = set(self.active_dockings.keys())
        logger.debug(f"[PortSpoofing] Looking for MMSIs to spoof: {docking_mmsis}")
        
        mask = chunk['MMSI'].isin(docking_mmsis)
        matching_count = mask.sum()
        
        logger.debug(f"[PortSpoofing] Found {matching_count} records to spoof in this chunk")
        
        if not mask.any():
            logger.debug(f"[PortSpoofing] No matching MMSIs found in chunk. Chunk MMSIs: {chunk['MMSI'].unique()}")
            return None
            
        # 创建修改数据框
        modifications = pd.DataFrame(index=chunk.index[mask])
        logger.debug(f"[PortSpoofing] Creating modifications for {len(modifications)} records")
        
        for idx in modifications.index:
            mmsi = int(chunk.loc[idx, 'MMSI'])
            if mmsi not in self.active_dockings:
                logger.warning(f"[PortSpoofing] MMSI {mmsi} not in active_dockings, skipping")
                continue
                
            docking = self.active_dockings[mmsi]
            original_lat = chunk.loc[idx, 'LAT']
            original_lon = chunk.loc[idx, 'LON']
            
            logger.debug(f"[PortSpoofing] MMSI {mmsi}: Original position ({original_lat:.4f}, {original_lon:.4f})")
            
            # 生成伪造位置
            spoofed_lat, spoofed_lon = self._generate_spoofed_position(
                docking, current_time
            )
            
            logger.debug(f"[PortSpoofing] MMSI {mmsi}: Spoofed position ({spoofed_lat:.4f}, {spoofed_lon:.4f})")
            
            # 应用伪造位置
            modifications.loc[idx, 'LAT'] = spoofed_lat
            modifications.loc[idx, 'LON'] = spoofed_lon
            
            # 调整速度和航向以匹配伪造轨迹
            spoofed_sog, spoofed_cog = self._calculate_spoofed_motion(
                docking, (spoofed_lat, spoofed_lon), current_time
            )
            
            modifications.loc[idx, 'SOG'] = spoofed_sog
            modifications.loc[idx, 'COG'] = spoofed_cog
            
            logger.info(f"[PortSpoofing] MMSI {mmsi}: Position spoofed from ({original_lat:.4f}, {original_lon:.4f}) "
                       f"to ({spoofed_lat:.4f}, {spoofed_lon:.4f}), SOG: {spoofed_sog:.1f}, COG: {spoofed_cog:.1f}")
            
        result = modifications if not modifications.empty else None
        logger.debug(f"[PortSpoofing] Returning {len(modifications) if result is not None else 0} modifications")
        return result
        
    def _generate_spoofed_position(self, docking: DockingEvent, current_time: datetime) -> Tuple[float, float]:
        """生成伪造位置"""
        # 计算停靠进度
        total_time = (docking.end_time - docking.start_time).total_seconds()
        elapsed_time = (current_time - docking.start_time).total_seconds()
        progress = max(0, min(1, elapsed_time / total_time))
        
        # 使用贝塞尔曲线生成自然轨迹
        start_pos = docking.last_sea_position
        end_pos = docking.next_sea_position
        
        # 控制点（在航线中间偏移）
        mid_lat = (start_pos[0] + end_pos[0]) / 2
        mid_lon = (start_pos[1] + end_pos[1]) / 2
        
        # 添加随机偏移使轨迹更自然
        offset_scale = self.spoofing_distance * 0.3
        lat_offset = (random.random() - 0.5) * offset_scale / 60.0
        lon_offset = (random.random() - 0.5) * offset_scale / (60.0 * math.cos(math.radians(mid_lat)))
        
        control_lat = mid_lat + lat_offset
        control_lon = mid_lon + lon_offset
        
        # 贝塞尔曲线插值
        t = progress
        spoofed_lat = (1-t)**2 * start_pos[0] + 2*(1-t)*t * control_lat + t**2 * end_pos[0]
        spoofed_lon = (1-t)**2 * start_pos[1] + 2*(1-t)*t * control_lon + t**2 * end_pos[1]
        
        return spoofed_lat, spoofed_lon
        
    def _calculate_spoofed_motion(self, docking: DockingEvent, position: Tuple[float, float], 
                                 current_time: datetime) -> Tuple[float, float]:
        """计算伪造的运动参数"""
        # 获取历史伪造位置
        if not docking.spoofed_positions:
            # 第一个位置，使用合理的默认值
            return 8.0, 90.0  # 8节，东向
            
        # 计算与上一个位置的差异
        last_pos = docking.spoofed_positions[-1]
        time_diff = (current_time - last_pos[2]).total_seconds()
        
        if time_diff <= 0:
            return 8.0, 90.0
            
        # 计算距离和航向
        distance = self._calculate_distance(
            (last_pos[0], last_pos[1]),
            position
        )
        
        # 计算速度（海里/小时）
        speed = distance / (time_diff / 3600.0)
        speed = max(3.0, min(15.0, speed))  # 限制在合理范围内
        
        # 计算航向
        bearing = self._calculate_bearing(
            (last_pos[0], last_pos[1]),
            position
        )
        
        # 记录当前伪造位置
        docking.spoofed_positions.append((position[0], position[1], current_time))
        
        # 限制历史记录长度
        if len(docking.spoofed_positions) > 50:
            docking.spoofed_positions = docking.spoofed_positions[-50:]
            
        return speed, bearing
        
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间距离(海里)"""
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 3440.065 * c  # 地球半径(海里)
        
    def _calculate_bearing(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间方位角"""
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlon = lon2 - lon1
        
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        bearing = np.arctan2(y, x)
        return (np.degrees(bearing) + 360) % 360
        
    def get_impact_type(self) -> str:
        """获取攻击影响类型"""
        return "PositionDeception"
        
    def get_attack_statistics(self) -> Dict[str, Any]:
        """获取攻击统计信息"""
        return {
            'total_embargo_ports': len(self.embargo_ports),
            'active_dockings': len(self.active_dockings),
            'attack_parameters': {
                'spoofing_distance': self.spoofing_distance,
                'min_docking_time': self.min_docking_time,
                'speed_threshold': self.speed_threshold,
                'trajectory_smoothness': self.trajectory_smoothness
            },
            'embargo_ports': [
                {
                    'name': port.name,
                    'location': (port.lat, port.lon),
                    'radius': port.radius,
                    'country': port.country,
                    'port_type': port.port_type
                }
                for port in self.embargo_ports
            ],
            'active_docking_events': [
                {
                    'mmsi': mmsi,
                    'port_name': docking.port.name,
                    'start_time': docking.start_time.isoformat(),
                    'end_time': docking.end_time.isoformat(),
                    'last_sea_position': docking.last_sea_position,
                    'next_sea_position': docking.next_sea_position
                }
                for mmsi, docking in self.active_dockings.items()
            ]
        }