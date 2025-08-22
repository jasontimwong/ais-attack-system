"""
Streaming data validator with record-level and track-level validation.
"""

from typing import Iterator, Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    validator: str
    severity: ValidationSeverity
    reason: str
    chunk_id: Optional[int] = None
    record_index: Optional[int] = None
    mmsi: Optional[int] = None
    timestamp: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration for validators."""
    # Physical thresholds
    max_speed_kn: float = 60.0
    max_acceleration_kn_per_min: float = 5.0
    max_position_jump_nm: float = 50.0
    max_rot: float = 127.0
    min_rot: float = -128.0
    
    # Logical thresholds
    stationary_speed_threshold_kn: float = 0.5
    identity_conflict_distance_nm: float = 1.0
    identity_conflict_time_window_sec: int = 30
    
    # Severity thresholds
    severity_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'position_jump_nm': {'low': 10, 'medium': 30, 'high': 50},
        'speed_kn': {'low': 40, 'medium': 60, 'high': 80},
        'acceleration_kn_per_min': {'low': 3, 'medium': 5, 'high': 10}
    })
    
    # Validation overrides
    skip_land_check: bool = False
    skip_speed_check: bool = False
    skip_identity_check: bool = False
    
    # Track validation parameters
    track_history_size: int = 100  # Records to keep per vessel
    track_time_window_min: int = 30  # Time window for track analysis


class RecordValidator:
    """Validates individual AIS records."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.issues: List[ValidationIssue] = []
    
    def validate_chunk(self, chunk: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
        """
        Validate records in a chunk.
        Returns DataFrame with additional validation columns.
        """
        # Add validation columns
        chunk['valid'] = True
        chunk['validation_issues'] = None
        
        # Run validators
        self._validate_positions(chunk, chunk_id)
        self._validate_speeds(chunk, chunk_id)
        self._validate_courses(chunk, chunk_id)
        self._validate_rot(chunk, chunk_id)
        self._validate_nav_status(chunk, chunk_id)
        
        return chunk
    
    def _validate_positions(self, chunk: pd.DataFrame, chunk_id: int):
        """Validate position values."""
        # Latitude validation
        invalid_lat = (chunk['LAT'] < -90) | (chunk['LAT'] > 90)
        if invalid_lat.any():
            for idx in chunk[invalid_lat].index:
                self.issues.append(ValidationIssue(
                    validator='position',
                    severity=ValidationSeverity.ERROR,
                    reason='Invalid latitude',
                    chunk_id=chunk_id,
                    record_index=idx,
                    mmsi=chunk.loc[idx, 'MMSI'],
                    details={'lat': chunk.loc[idx, 'LAT']}
                ))
                chunk.loc[idx, 'valid'] = False
        
        # Longitude validation
        invalid_lon = (chunk['LON'] < -180) | (chunk['LON'] > 180)
        if invalid_lon.any():
            for idx in chunk[invalid_lon].index:
                self.issues.append(ValidationIssue(
                    validator='position',
                    severity=ValidationSeverity.ERROR,
                    reason='Invalid longitude',
                    chunk_id=chunk_id,
                    record_index=idx,
                    mmsi=chunk.loc[idx, 'MMSI'],
                    details={'lon': chunk.loc[idx, 'LON']}
                ))
                chunk.loc[idx, 'valid'] = False
        
        # Null island check
        null_island = (chunk['LAT'] == 0) & (chunk['LON'] == 0)
        if null_island.any():
            for idx in chunk[null_island].index:
                self.issues.append(ValidationIssue(
                    validator='position',
                    severity=ValidationSeverity.WARNING,
                    reason='Position at null island (0,0)',
                    chunk_id=chunk_id,
                    record_index=idx,
                    mmsi=chunk.loc[idx, 'MMSI']
                ))
    
    def _validate_speeds(self, chunk: pd.DataFrame, chunk_id: int):
        """Validate speed values."""
        if self.config.skip_speed_check:
            return
        
        # Check for negative speeds
        negative_speed = chunk['SOG'] < 0
        if negative_speed.any():
            for idx in chunk[negative_speed].index:
                self.issues.append(ValidationIssue(
                    validator='speed',
                    severity=ValidationSeverity.ERROR,
                    reason='Negative speed',
                    chunk_id=chunk_id,
                    record_index=idx,
                    mmsi=chunk.loc[idx, 'MMSI'],
                    details={'sog': chunk.loc[idx, 'SOG']}
                ))
                chunk.loc[idx, 'valid'] = False
        
        # Check for excessive speeds
        thresholds = self.config.severity_thresholds['speed_kn']
        
        for severity, threshold in [
            (ValidationSeverity.WARNING, thresholds['low']),
            (ValidationSeverity.ERROR, thresholds['high'])
        ]:
            excessive_speed = chunk['SOG'] > threshold
            if excessive_speed.any():
                for idx in chunk[excessive_speed].index:
                    self.issues.append(ValidationIssue(
                        validator='speed',
                        severity=severity,
                        reason=f'Speed exceeds {threshold} knots',
                        chunk_id=chunk_id,
                        record_index=idx,
                        mmsi=chunk.loc[idx, 'MMSI'],
                        details={'sog': chunk.loc[idx, 'SOG']}
                    ))
    
    def _validate_courses(self, chunk: pd.DataFrame, chunk_id: int):
        """Validate course values."""
        invalid_course = (chunk['COG'] < 0) | (chunk['COG'] >= 360)
        if invalid_course.any():
            for idx in chunk[invalid_course].index:
                self.issues.append(ValidationIssue(
                    validator='course',
                    severity=ValidationSeverity.WARNING,
                    reason='Course out of range',
                    chunk_id=chunk_id,
                    record_index=idx,
                    mmsi=chunk.loc[idx, 'MMSI'],
                    details={'cog': chunk.loc[idx, 'COG']}
                ))
    
    def _validate_rot(self, chunk: pd.DataFrame, chunk_id: int):
        """Validate rate of turn values."""
        if 'ROT' not in chunk.columns:
            return
        invalid_rot = (chunk['ROT'] < self.config.min_rot) | (chunk['ROT'] > self.config.max_rot)
        if invalid_rot.any():
            for idx in chunk[invalid_rot].index:
                self.issues.append(ValidationIssue(
                    validator='rot',
                    severity=ValidationSeverity.WARNING,
                    reason='Rate of turn out of range',
                    chunk_id=chunk_id,
                    record_index=idx,
                    mmsi=chunk.loc[idx, 'MMSI'],
                    details={'rot': chunk.loc[idx, 'ROT']}
                ))
    
    def _validate_nav_status(self, chunk: pd.DataFrame, chunk_id: int):
        """Validate navigation status consistency."""
        # Map status strings to codes
        stationary_statuses = {'At anchor', 'Moored', 'Aground'}
        
        # Check stationary status with movement
        for status in stationary_statuses:
            mask = (chunk['Status'] == status) & (chunk['SOG'] > self.config.stationary_speed_threshold_kn)
            if mask.any():
                for idx in chunk[mask].index:
                    self.issues.append(ValidationIssue(
                        validator='nav_status',
                        severity=ValidationSeverity.WARNING,
                        reason=f'Vessel moving while status is "{status}"',
                        chunk_id=chunk_id,
                        record_index=idx,
                        mmsi=chunk.loc[idx, 'MMSI'],
                        details={'status': status, 'sog': chunk.loc[idx, 'SOG']}
                    ))


class TrackValidator:
    """Validates vessel tracks across multiple records."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.vessel_tracks: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.track_history_size)
        )
        self.issues: List[ValidationIssue] = []
    
    def update_tracks(self, chunk: pd.DataFrame, chunk_id: int):
        """Update vessel track history with new chunk data."""
        for mmsi, vessel_data in chunk.groupby('MMSI'):
            # Sort by time
            vessel_data = vessel_data.sort_values('BaseDateTime')
            
            # Add to track history
            for _, record in vessel_data.iterrows():
                self.vessel_tracks[mmsi].append({
                    'timestamp': record['BaseDateTime'],
                    'lat': record['LAT'],
                    'lon': record['LON'],
                    'sog': record['SOG'],
                    'cog': record['COG'],
                    'chunk_id': chunk_id,
                    'index': record.name
                })
            
            # Validate updated track
            self._validate_vessel_track(mmsi, chunk_id)
    
    def _validate_vessel_track(self, mmsi: int, chunk_id: int):
        """Validate a single vessel's track."""
        track = list(self.vessel_tracks[mmsi])
        if len(track) < 2:
            return
        
        # Check consecutive pairs
        for i in range(1, len(track)):
            prev = track[i - 1]
            curr = track[i]
            
            # Time difference
            time_diff = (curr['timestamp'] - prev['timestamp']).total_seconds()
            if time_diff <= 0:
                continue  # Skip if out of order
            
            # Position jump check
            distance_nm = self._haversine_distance(
                prev['lat'], prev['lon'],
                curr['lat'], curr['lon']
            )
            
            if time_diff > 0:
                implied_speed = (distance_nm / time_diff) * 3600  # knots
                
                # Check for impossible speed
                thresholds = self.config.severity_thresholds['position_jump_nm']
                
                if distance_nm > thresholds['high']:
                    self.issues.append(ValidationIssue(
                        validator='track_continuity',
                        severity=ValidationSeverity.ERROR,
                        reason=f'Position jump of {distance_nm:.1f} nm',
                        chunk_id=chunk_id,
                        mmsi=mmsi,
                        timestamp=curr['timestamp'],
                        details={
                            'distance_nm': distance_nm,
                            'time_diff_sec': time_diff,
                            'implied_speed_kn': implied_speed
                        }
                    ))
                elif distance_nm > thresholds['medium']:
                    self.issues.append(ValidationIssue(
                        validator='track_continuity',
                        severity=ValidationSeverity.WARNING,
                        reason=f'Large position change of {distance_nm:.1f} nm',
                        chunk_id=chunk_id,
                        mmsi=mmsi,
                        timestamp=curr['timestamp'],
                        details={
                            'distance_nm': distance_nm,
                            'implied_speed_kn': implied_speed
                        }
                    ))
            
            # Acceleration check
            if prev['sog'] is not None and curr['sog'] is not None:
                speed_change = abs(curr['sog'] - prev['sog'])
                minutes = time_diff / 60
                
                if minutes > 0:
                    acceleration = speed_change / minutes
                    
                    acc_thresholds = self.config.severity_thresholds['acceleration_kn_per_min']
                    if acceleration > acc_thresholds['high']:
                        self.issues.append(ValidationIssue(
                            validator='acceleration',
                            severity=ValidationSeverity.ERROR,
                            reason=f'Extreme acceleration: {acceleration:.1f} kn/min',
                            chunk_id=chunk_id,
                            mmsi=mmsi,
                            timestamp=curr['timestamp'],
                            details={
                                'acceleration_kn_per_min': acceleration,
                                'speed_change_kn': speed_change
                            }
                        ))
    
    def check_identity_conflicts(self, chunk: pd.DataFrame, chunk_id: int) -> List[ValidationIssue]:
        """Check for identity conflicts (same MMSI at different locations)."""
        if self.config.skip_identity_check:
            return []
        
        conflicts = []
        
        # Group by timestamp (rounded to nearest 30 seconds)
        chunk['time_bucket'] = chunk['BaseDateTime'].dt.round('30s')
        
        for time_bucket, time_group in chunk.groupby('time_bucket'):
            # Check each MMSI in this time window
            for mmsi, mmsi_group in time_group.groupby('MMSI'):
                if len(mmsi_group) > 1:
                    # Calculate pairwise distances
                    positions = mmsi_group[['LAT', 'LON']].values
                    max_distance = 0
                    
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            dist = self._haversine_distance(
                                positions[i][0], positions[i][1],
                                positions[j][0], positions[j][1]
                            )
                            max_distance = max(max_distance, dist)
                    
                    if max_distance > self.config.identity_conflict_distance_nm:
                        conflicts.append(ValidationIssue(
                            validator='identity',
                            severity=ValidationSeverity.ERROR,
                            reason='MMSI appears at multiple distant locations',
                            chunk_id=chunk_id,
                            mmsi=mmsi,
                            timestamp=time_bucket,
                            details={
                                'max_distance_nm': max_distance,
                                'num_positions': len(mmsi_group)
                            }
                        ))
        
        return conflicts
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in nautical miles."""
        R = 3440.065  # Earth radius in nautical miles
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) *
             np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c


class StreamingDataValidator:
    """
    Main validator that coordinates record and track validation.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.record_validator = RecordValidator(self.config)
        self.track_validator = TrackValidator(self.config)
        self.all_issues: List[ValidationIssue] = []
        self.stats = {
            'chunks_validated': 0,
            'records_validated': 0,
            'issues_by_severity': defaultdict(int),
            'issues_by_validator': defaultdict(int)
        }
    
    def validate_stream(self, chunk_iter: Iterator[pd.DataFrame]) -> Iterator[Tuple[pd.DataFrame, List[ValidationIssue]]]:
        """
        Validate data stream chunk by chunk.
        
        Yields:
            Tuples of (validated_chunk, issues_for_chunk)
        """
        chunk_id = 0
        
        for chunk in chunk_iter:
            # Clear per-chunk issues
            self.record_validator.issues = []
            self.track_validator.issues = []
            
            # Record validation
            validated_chunk = self.record_validator.validate_chunk(chunk, chunk_id)
            
            # Track validation
            self.track_validator.update_tracks(validated_chunk, chunk_id)
            
            # Identity conflict check
            identity_issues = self.track_validator.check_identity_conflicts(validated_chunk, chunk_id)
            
            # Collect all issues
            chunk_issues = (
                self.record_validator.issues +
                self.track_validator.issues +
                identity_issues
            )
            
            # Update statistics
            self._update_stats(chunk, chunk_issues)
            
            # Store issues
            self.all_issues.extend(chunk_issues)
            
            yield validated_chunk, chunk_issues
            chunk_id += 1
    
    def _update_stats(self, chunk: pd.DataFrame, issues: List[ValidationIssue]):
        """Update validation statistics."""
        self.stats['chunks_validated'] += 1
        self.stats['records_validated'] += len(chunk)
        
        for issue in issues:
            self.stats['issues_by_severity'][issue.severity.value] += 1
            self.stats['issues_by_validator'][issue.validator] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'chunks_validated': self.stats['chunks_validated'],
            'records_validated': self.stats['records_validated'],
            'total_issues': len(self.all_issues),
            'issues_by_severity': dict(self.stats['issues_by_severity']),
            'issues_by_validator': dict(self.stats['issues_by_validator']),
            'error_count': self.stats['issues_by_severity'].get('error', 0),
            'warning_count': self.stats['issues_by_severity'].get('warning', 0)
        }
    
    def assess_severity(self, event_issues: List[ValidationIssue]) -> str:
        """
        Assess overall severity for an attack event based on its validation issues.
        
        Returns:
            'low', 'medium', or 'high'
        """
        if any(issue.severity == ValidationSeverity.ERROR for issue in event_issues):
            return 'high'
        elif any(issue.severity == ValidationSeverity.WARNING for issue in event_issues):
            return 'medium'
        else:
            return 'low'