"""
Consequence evaluation module for assessing attack impacts - Streaming Version.
Designed for memory-efficient processing of large datasets.
"""

import math
import logging
from typing import Dict, Any, List, Optional, Iterator, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

from .attack_plugin_v2 import AttackEvent, DataPatch, PatchType

logger = logging.getLogger(__name__)


class StreamingConsequenceEvaluator:
    """
    Evaluates consequences of attacks using streaming data processing.
    Specifically optimized for false target CPA calculations.
    """
    
    # CPA calculation parameters
    CPA_TIME_WINDOW = 1800  # 30 minutes window for CPA calculation
    COLLISION_RISK_THRESHOLD = 0.3  # nautical miles for high severity
    MEDIUM_RISK_THRESHOLD = 1.0  # nautical miles for medium severity
    NEAR_MISS_THRESHOLD = 2.0  # nautical miles
    
    # Chunk parameters for memory efficiency
    MAX_RECORDS_IN_MEMORY = 50000  # Limit records held in memory
    
    def __init__(self):
        self.vessel_cache = {}  # Cache recent vessel positions
        self.cpa_results = {}  # Store CPA calculations per event
        
    def evaluate_streaming(self, 
                          baseline_chunks: Iterator[pd.DataFrame],
                          attacked_chunks: Iterator[pd.DataFrame],
                          events: List[AttackEvent],
                          patches: List[DataPatch]) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate attack impacts using streaming data.
        
        Args:
            baseline_chunks: Iterator of baseline data chunks
            attacked_chunks: Iterator of attacked data chunks  
            events: List of attack events
            patches: List of data patches applied
            
        Returns:
            Dictionary mapping event_id to impact metrics
        """
        logger.info(f"Starting streaming evaluation for {len(events)} events")
        
        # Initialize results
        results = {event.event_id: {} for event in events}
        
        # For S1 attacks, we need to evaluate CPA with false targets
        s1_events = [e for e in events if e.attack_type == 'S1']
        
        if s1_events:
            # Process false target CPA calculations
            logger.info(f"Evaluating {len(s1_events)} false target events")
            self._evaluate_false_targets_streaming(
                baseline_chunks, patches, s1_events, results
            )
        
        # For S2 attacks, we need to evaluate tampering impact
        s2_events = [e for e in events if e.attack_type == 'S2']
        
        if s2_events:
            # Process tampering impact calculations
            logger.info(f"Evaluating {len(s2_events)} tampering events")
            self._evaluate_tampering_streaming(
                baseline_chunks, attacked_chunks, patches, s2_events, results
            )
        
        # Add severity classifications based on attack type
        for event_id, metrics in results.items():
            event = next((e for e in events if e.event_id == event_id), None)
            if event:
                if event.attack_type == 'S1' and 'min_cpa_nm' in metrics:
                    metrics['severity'] = self._classify_severity(metrics['min_cpa_nm'])
                    logger.info(f"Event {event_id}: min_cpa_nm={metrics['min_cpa_nm']:.3f}, severity={metrics['severity']}")
                elif event.attack_type == 'S2' and 'max_pos_offset_m' in metrics:
                    metrics['severity'] = self._classify_tampering_severity(metrics['max_pos_offset_m'])
                    logger.info(f"Event {event_id}: max_pos_offset_m={metrics['max_pos_offset_m']:.1f}, severity={metrics['severity']}")
        
        return results
    
    def _evaluate_false_targets_streaming(self,
                                        baseline_chunks: Iterator[pd.DataFrame],
                                        patches: List[DataPatch],
                                        events: List[AttackEvent],
                                        results: Dict[int, Dict[str, Any]]):
        """Evaluate false target impacts using streaming approach."""
        
        # Group patches by event
        patches_by_event = defaultdict(list)
        for patch in patches:
            if patch.patch_type == PatchType.INSERT and patch.new_records is not None:
                # Determine which event this patch belongs to
                for event in events:
                    if not patch.new_records.empty:
                        patch_time = patch.new_records['BaseDateTime'].iloc[0]
                        if event.start_ts <= patch_time <= event.end_ts:
                            patches_by_event[event.event_id].append(patch)
                            break
        
        # Extract false target trajectories from patches
        false_targets = {}
        for event in events:
            event_patches = patches_by_event[event.event_id]
            if event_patches:
                # Combine all new records for this event
                all_records = pd.concat([p.new_records for p in event_patches], ignore_index=True)
                if not all_records.empty:
                    false_mmsi = all_records['MMSI'].iloc[0]
                    false_targets[event.event_id] = {
                        'mmsi': false_mmsi,
                        'trajectory': all_records,
                        'start_time': event.start_ts,
                        'end_time': event.end_ts
                    }
        
        # Process baseline chunks to calculate CPA
        chunk_count = 0
        vessel_buffer = defaultdict(list)  # Buffer recent positions per vessel
        
        for chunk in baseline_chunks:
            chunk_count += 1
            
            # For each false target, calculate CPA with vessels in this chunk
            for event_id, target_info in false_targets.items():
                self._process_chunk_for_cpa(
                    chunk, target_info, event_id, vessel_buffer, results
                )
            
            # Clean up old positions from buffer to manage memory
            self._cleanup_vessel_buffer(vessel_buffer)
            
            if chunk_count % 10 == 0:
                logger.info(f"Processed {chunk_count} chunks for CPA calculation")
        
        # Finalize CPA results
        for event_id in false_targets:
            if event_id not in results:
                results[event_id] = {}
            
            # Find minimum CPA across all vessels
            if event_id in self.cpa_results and self.cpa_results[event_id]:
                min_cpa_info = min(self.cpa_results[event_id], key=lambda x: x['cpa_distance'])
                results[event_id]['min_cpa_nm'] = round(min_cpa_info['cpa_distance'], 3)
                results[event_id]['cpa_vessel_mmsi'] = min_cpa_info['vessel_mmsi']
                results[event_id]['cpa_time'] = min_cpa_info['cpa_time'].isoformat()
                
                # Count risk vessels
                collision_risk = sum(1 for r in self.cpa_results[event_id] 
                                   if r['cpa_distance'] < self.COLLISION_RISK_THRESHOLD)
                near_miss = sum(1 for r in self.cpa_results[event_id] 
                               if r['cpa_distance'] < self.NEAR_MISS_THRESHOLD)
                
                results[event_id]['collision_risk_vessels'] = collision_risk
                results[event_id]['near_miss_vessels'] = near_miss
                
                # Add metrics for added messages
                event_patches = patches_by_event[event_id]
                total_added_messages = sum(len(p.new_records) for p in event_patches if p.new_records is not None)
                results[event_id]['added_messages'] = total_added_messages
                
                # Add validators_failed (placeholder - would need actual validation logic)
                results[event_id]['validators_failed'] = []
            else:
                results[event_id]['min_cpa_nm'] = float('inf')
                results[event_id]['cpa_vessel_mmsi'] = None
                results[event_id]['collision_risk_vessels'] = 0
                results[event_id]['near_miss_vessels'] = 0
                results[event_id]['added_messages'] = 0
                results[event_id]['validators_failed'] = []
    
    def _process_chunk_for_cpa(self, chunk: pd.DataFrame, 
                              target_info: Dict[str, Any],
                              event_id: int,
                              vessel_buffer: Dict[int, List],
                              results: Dict[int, Dict[str, Any]]):
        """Process a chunk to calculate CPA with false target."""
        
        false_trajectory = target_info['trajectory']
        false_mmsi = target_info['mmsi']
        start_time = target_info['start_time']
        end_time = target_info['end_time']
        
        # Filter chunk to relevant time window
        mask = (chunk['BaseDateTime'] >= start_time - timedelta(seconds=self.CPA_TIME_WINDOW)) & \
               (chunk['BaseDateTime'] <= end_time + timedelta(seconds=self.CPA_TIME_WINDOW))
        relevant_chunk = chunk[mask]
        
        if relevant_chunk.empty:
            return
        
        # Limit to nearest vessels for efficiency (sample if too many)
        if len(relevant_chunk) > 1000:
            # Sample vessels to process
            unique_vessels = relevant_chunk['MMSI'].unique()
            if len(unique_vessels) > 50:
                sampled_vessels = np.random.choice(unique_vessels, 50, replace=False)
                relevant_chunk = relevant_chunk[relevant_chunk['MMSI'].isin(sampled_vessels)]
        
        # Group by vessel MMSI
        for mmsi, vessel_data in relevant_chunk.groupby('MMSI'):
            if mmsi == false_mmsi:
                continue
            
            # Limit vessel data points for efficiency
            if len(vessel_data) > 20:
                # Sample every nth record to reduce computation
                vessel_data = vessel_data.iloc[::max(1, len(vessel_data) // 20)]
            
            # Calculate CPA using vector-based approach
            cpa_info = self._calculate_vector_cpa(
                false_trajectory, vessel_data, mmsi
            )
            
            if cpa_info:
                if event_id not in self.cpa_results:
                    self.cpa_results[event_id] = []
                self.cpa_results[event_id].append(cpa_info)
    
    def _calculate_vector_cpa(self, 
                            target_trajectory: pd.DataFrame,
                            vessel_trajectory: pd.DataFrame,
                            vessel_mmsi: int) -> Optional[Dict[str, Any]]:
        """Calculate CPA between target and vessel trajectories using vector-based approach."""
        
        min_cpa_nm = float('inf')
        cpa_time = None
        cpa_target_time = None
        
        # Ensure trajectories are sorted by time
        target_trajectory = target_trajectory.sort_values('BaseDateTime')
        vessel_trajectory = vessel_trajectory.sort_values('BaseDateTime')
        
        # Sample trajectories for efficiency if needed
        if len(target_trajectory) > 30:
            target_trajectory = target_trajectory.iloc[::max(1, len(target_trajectory) // 30)]
        if len(vessel_trajectory) > 30:
            vessel_trajectory = vessel_trajectory.iloc[::max(1, len(vessel_trajectory) // 30)]
        
        # For each pair of time-synchronized positions
        for _, target_row in target_trajectory.iterrows():
            target_time = target_row['BaseDateTime']
            
            # Find vessel positions near this time
            time_window = timedelta(minutes=5)  # 5 minute window for matching
            time_mask = (vessel_trajectory['BaseDateTime'] >= target_time - time_window) & \
                       (vessel_trajectory['BaseDateTime'] <= target_time + time_window)
            nearby_vessel_data = vessel_trajectory[time_mask]
            
            if nearby_vessel_data.empty:
                continue
            
            # Use closest time match
            time_diffs = abs((nearby_vessel_data['BaseDateTime'] - target_time).dt.total_seconds())
            closest_idx = time_diffs.idxmin()
            vessel_row = nearby_vessel_data.loc[closest_idx]
            
            # Calculate CPA using vector formula
            cpa_result = self._compute_vector_cpa(
                target_row['LAT'], target_row['LON'], 
                target_row['SOG'], target_row['COG'],
                vessel_row['LAT'], vessel_row['LON'],
                vessel_row['SOG'], vessel_row['COG'],
                window_len_sec=self.CPA_TIME_WINDOW
            )
            
            if cpa_result and cpa_result['cpa_nm'] < min_cpa_nm:
                min_cpa_nm = cpa_result['cpa_nm']
                cpa_time = target_time + timedelta(seconds=cpa_result['t_cpa'])
                cpa_target_time = target_time
        
        if min_cpa_nm < float('inf'):
            return {
                'vessel_mmsi': vessel_mmsi,
                'cpa_distance': float(min_cpa_nm),
                'cpa_time': cpa_time
            }
        
        return None
    
    def _compute_vector_cpa(self, lat_a: float, lon_a: float, sog_a: float, cog_a: float,
                           lat_b: float, lon_b: float, sog_b: float, cog_b: float,
                           window_len_sec: float) -> Optional[Dict[str, float]]:
        """
        Compute CPA using vector-based calculation with proper coordinate transformation.
        
        Args:
            lat_a, lon_a: Position of vessel A (degrees)
            sog_a: Speed of vessel A (knots)
            cog_a: Course of vessel A (degrees)
            lat_b, lon_b: Position of vessel B (degrees)
            sog_b: Speed of vessel B (knots)
            cog_b: Course of vessel B (degrees)
            window_len_sec: Time window for CPA calculation (seconds)
            
        Returns:
            Dictionary with 'cpa_nm' and 't_cpa' or None if calculation fails
        """
        # Convert to local tangent plane centered at vessel A
        # This provides accurate distance calculations for nearby vessels
        
        # Reference point (vessel A position)
        ref_lat = lat_a
        ref_lon = lon_a
        
        # Convert positions to local tangent plane (meters)
        # X-axis points East, Y-axis points North
        lat_to_m = 111320.0  # meters per degree latitude (constant)
        lon_to_m = 111320.0 * np.cos(np.radians(ref_lat))  # meters per degree longitude at reference latitude
        
        # Position of vessel A in local tangent plane (origin)
        pos_a = np.array([0.0, 0.0])
        
        # Position of vessel B in local tangent plane
        delta_lat = lat_b - ref_lat
        delta_lon = lon_b - ref_lon
        pos_b = np.array([
            delta_lon * lon_to_m,  # East component
            delta_lat * lat_to_m   # North component
        ])
        
        # Convert speeds from knots to m/s
        speed_a_ms = sog_a * 0.514444
        speed_b_ms = sog_b * 0.514444
        
        # Velocity vectors in local tangent plane
        # COG is clockwise from North, so we need to convert to standard math angles
        # Math angle = 90 - COG (converts from navigation to math convention)
        angle_a = np.radians(90 - cog_a)
        angle_b = np.radians(90 - cog_b)
        
        vel_a = np.array([
            speed_a_ms * np.cos(angle_a),  # East component
            speed_a_ms * np.sin(angle_a)   # North component
        ])
        vel_b = np.array([
            speed_b_ms * np.cos(angle_b),  # East component
            speed_b_ms * np.sin(angle_b)   # North component
        ])
        
        # Relative position and velocity
        rel_p = pos_b - pos_a
        rel_v = vel_b - vel_a
        
        # Check if vessels are moving relative to each other
        rel_v_norm_sq = np.dot(rel_v, rel_v)
        if rel_v_norm_sq < 1e-6:  # Nearly stationary relative to each other
            # CPA is current distance
            cpa_dist_m = np.linalg.norm(rel_p)
            cpa_nm = cpa_dist_m / 1852.0
            return {'cpa_nm': cpa_nm, 't_cpa': 0.0}
        
        # Time to CPA
        t_cpa = -np.dot(rel_p, rel_v) / rel_v_norm_sq
        
        # Clip to time window
        t_cpa = np.clip(t_cpa, 0, window_len_sec)
        
        # CPA distance
        cpa_dist_m = np.linalg.norm(rel_p + rel_v * t_cpa)
        cpa_nm = cpa_dist_m / 1852.0
        
        return {'cpa_nm': cpa_nm, 't_cpa': t_cpa}
    
    def _calculate_distances_vectorized(self, lat1: float, lon1: float,
                                      lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
        """Calculate distances using vectorized haversine formula."""
        R = 3440.065  # Earth radius in nautical miles
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lats2)
        delta_lat = np.radians(lats2 - lat1)
        delta_lon = np.radians(lons2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) *
             np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _cleanup_vessel_buffer(self, vessel_buffer: Dict[int, List]):
        """Remove old entries from vessel buffer to manage memory."""
        current_time = datetime.now()
        
        for mmsi in list(vessel_buffer.keys()):
            # Remove entries older than CPA time window
            vessel_buffer[mmsi] = [
                r for r in vessel_buffer[mmsi]
                if (current_time - r.get('BaseDateTime', current_time)).total_seconds() < self.CPA_TIME_WINDOW * 2
            ]
            
            # Remove vessel if no recent data
            if not vessel_buffer[mmsi]:
                del vessel_buffer[mmsi]
    
    def _classify_severity(self, min_cpa_nm: float) -> str:
        """
        Classify severity based on minimum CPA distance.
        
        Args:
            min_cpa_nm: Minimum CPA in nautical miles
            
        Returns:
            Severity level: 'high', 'medium', or 'low'
        """
        if min_cpa_nm < self.COLLISION_RISK_THRESHOLD:
            return 'high'
        elif min_cpa_nm < self.MEDIUM_RISK_THRESHOLD:
            return 'medium'
        else:
            return 'low'