"""
Attack plugin base class and manager with patch-based approach.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PatchType(Enum):
    """Types of data patches."""
    MODIFY = "modify"      # Modify existing records
    INSERT = "insert"      # Insert new records
    DELETE = "delete"      # Delete records


@dataclass
class DataPatch:
    """
    Represents a change to the data.
    Can be modifications, insertions, or deletions.
    """
    patch_type: PatchType
    chunk_id: Optional[int] = None      # Which chunk this applies to
    
    # For MODIFY patches
    modifications: Optional[pd.DataFrame] = None  # DataFrame with same index as original
    
    # For INSERT patches  
    new_records: Optional[pd.DataFrame] = None    # New records to insert
    
    # For DELETE patches
    delete_indices: Optional[List[int]] = None    # Indices to delete
    delete_mask: Optional[pd.Series] = None       # Boolean mask for deletion
    
    def apply_to_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply this patch to a data chunk."""
        if self.patch_type == PatchType.MODIFY:
            if self.modifications is not None:
                # Update only the columns present in modifications
                for col in self.modifications.columns:
                    if col in chunk.columns:
                        # Use index alignment for updates
                        chunk.loc[self.modifications.index, col] = self.modifications[col]
            return chunk
            
        elif self.patch_type == PatchType.INSERT:
            if self.new_records is not None:
                # Append new records
                return pd.concat([chunk, self.new_records], ignore_index=True)
            return chunk
            
        elif self.patch_type == PatchType.DELETE:
            if self.delete_mask is not None:
                return chunk[~self.delete_mask]
            elif self.delete_indices is not None:
                return chunk.drop(self.delete_indices, errors='ignore')
            return chunk
        
        return chunk


@dataclass
class AttackEvent:
    """Enhanced event log with streaming support."""
    event_id: int
    attack_type: str
    impact_type: str
    start_ts: datetime
    end_ts: datetime
    target_mmsi: Optional[int] = None
    secondary_mmsi: Optional[int] = None
    
    # Metrics will be populated during evaluation
    metrics: Dict[str, Any] = None
    
    # Track affected chunks for efficient processing
    affected_chunks: List[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'attack_type': self.attack_type,
            'impact_type': self.impact_type,
            'start_ts': self.start_ts.isoformat(),
            'end_ts': self.end_ts.isoformat(),
            'target_mmsi': self.target_mmsi,
            'secondary_mmsi': self.secondary_mmsi,
            'metrics': self.metrics or {},
            'affected_chunks': self.affected_chunks or []
        }


class StreamingAttackPlugin(ABC):
    """
    Abstract base class for streaming attack plugins.
    Plugins generate patches instead of modifying data directly.
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.patches: List[DataPatch] = []
        self.event: Optional[AttackEvent] = None
    
    @abstractmethod
    def generate_patches(self, 
                        chunk_iter: Iterator[pd.DataFrame],
                        config: Dict[str, Any]) -> Iterator[DataPatch]:
        """
        Generate patches for the attack by processing data chunks.
        
        Args:
            chunk_iter: Iterator of DataFrame chunks
            config: Attack configuration including time window, targets, etc.
            
        Yields:
            DataPatch objects representing changes
        """
        pass
    
    @abstractmethod
    def validate_params(self) -> bool:
        """Validate plugin parameters."""
        pass
    
    def create_event(self, config: Dict[str, Any]) -> AttackEvent:
        """Create event log for this attack."""
        return AttackEvent(
            event_id=config.get('event_id', 0),
            attack_type=config.get('attack_type'),
            impact_type=self.get_impact_type(),
            start_ts=config.get('start_time'),
            end_ts=config.get('end_time'),
            target_mmsi=config.get('target'),
            secondary_mmsi=config.get('target_b')
        )
    
    @abstractmethod 
    def get_impact_type(self) -> str:
        """Get the impact type for this attack."""
        pass
    
    def filter_chunk_by_time(self, chunk: pd.DataFrame,
                           start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Filter chunk to time window."""
        mask = (chunk['BaseDateTime'] >= start_time) & (chunk['BaseDateTime'] <= end_time)
        return chunk[mask]
    
    def filter_chunk_by_mmsi(self, chunk: pd.DataFrame, mmsi: int) -> pd.DataFrame:
        """Filter chunk to specific vessel."""
        return chunk[chunk['MMSI'] == mmsi]
    
    def _get_param_with_test_mode(self, params: Dict[str, Any], key: str, 
                                 default_value: Any, test_value: Any) -> Any:
        """
        Get parameter with TEST_MODE support.
        
        Args:
            params: Parameter dictionary
            key: Parameter key
            default_value: Default value for production
            test_value: Value to use in TEST_MODE
            
        Returns:
            test_value if AIS_TEST_MODE is set, otherwise params.get(key, default_value)
        """
        import os
        if os.getenv('AIS_TEST_MODE'):
            logger.info(f"TEST_MODE: Using {key}={test_value} (instead of {params.get(key, default_value)})")
            return test_value
        return params.get(key, default_value)


class StreamingAttackManager:
    """
    Manages attack plugins and coordinates patch application in streaming fashion.
    """
    
    def __init__(self):
        self.plugins: Dict[str, type] = {}
        self._events_counter = 0
    
    def register_plugin(self, attack_type: str, plugin_class: type):
        """Register an attack plugin."""
        if not issubclass(plugin_class, StreamingAttackPlugin):
            raise ValueError(f"{plugin_class} must inherit from StreamingAttackPlugin")
        
        self.plugins[attack_type] = plugin_class
        logger.info(f"Registered plugin: {attack_type} -> {plugin_class.__name__}")
    
    def apply_attacks_streaming(self,
                              baseline_iter: Iterator[pd.DataFrame],
                              attack_iter: Iterator[pd.DataFrame],
                              scenarios: List[Dict[str, Any]]) -> Iterator[Tuple[pd.DataFrame, List[AttackEvent]]]:
        """
        Apply attacks in streaming fashion.
        
        Args:
            baseline_iter: Iterator of baseline chunks (for reading)
            attack_iter: Iterator of attack chunks (for writing)
            scenarios: List of attack scenarios
            
        Yields:
            Tuples of (modified_chunk, events_for_chunk)
        """
        # First pass: Generate all patches
        all_patches = []
        all_events = []
        
        for scenario in scenarios:
            attack_type = scenario.get('type')
            if attack_type not in self.plugins:
                raise ValueError(f"Unknown attack type: {attack_type}")
            
            # Create plugin instance
            plugin_class = self.plugins[attack_type]
            params = scenario.get('params', {})
            plugin = plugin_class(params)
            
            if not plugin.validate_params():
                raise ValueError(f"Invalid parameters for {attack_type}")
            
            # Prepare config
            config = self._prepare_config(scenario)
            
            # Create event
            event = plugin.create_event(config)
            all_events.append(event)
            
            # Generate patches
            logger.info(f"Generating patches for {attack_type} attack")
            patches = list(plugin.generate_patches(baseline_iter, config))
            all_patches.extend(patches)
            
            # Track affected chunks in event
            event.affected_chunks = list(set(p.chunk_id for p in patches if p.chunk_id is not None))
        
        # Second pass: Apply patches to chunks
        chunk_id = 0
        for chunk in attack_iter:
            # Find patches for this chunk
            chunk_patches = [p for p in all_patches 
                           if p.chunk_id is None or p.chunk_id == chunk_id]
            
            # Apply patches
            modified_chunk = chunk
            for patch in chunk_patches:
                modified_chunk = patch.apply_to_chunk(modified_chunk)
            
            # Find events affecting this chunk
            chunk_events = [e for e in all_events if chunk_id in (e.affected_chunks or [])]
            
            yield modified_chunk, chunk_events
            chunk_id += 1
    
    def generate_patches_only(self,
                            data_iter: Iterator[pd.DataFrame],
                            scenarios: List[Dict[str, Any]]) -> Tuple[List[DataPatch], List[AttackEvent]]:
        """
        Generate patches without applying them.
        Useful for validation and dry-run mode.
        """
        all_patches = []
        all_events = []
        
        for scenario in scenarios:
            attack_type = scenario.get('type')
            if attack_type not in self.plugins:
                raise ValueError(f"Unknown attack type: {attack_type}")
            
            # Create plugin and generate patches
            plugin_class = self.plugins[attack_type]
            params = scenario.get('params', {})
            plugin = plugin_class(params)
            
            if not plugin.validate_params():
                raise ValueError(f"Invalid parameters for {attack_type}")
            
            config = self._prepare_config(scenario)
            event = plugin.create_event(config)
            
            # Collect patches
            patches = list(plugin.generate_patches(data_iter, config))
            
            # Track affected chunks in event
            event.affected_chunks = list(set(p.chunk_id for p in patches if p.chunk_id is not None))
            
            all_patches.extend(patches)
            all_events.append(event)
        
        return all_patches, all_events
    
    def _prepare_config(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for plugin."""
        # Parse timestamps - support both formats
        if 'window' in scenario and isinstance(scenario['window'], list):
            start_time = self._parse_time(scenario['window'][0])
            end_time = self._parse_time(scenario['window'][1])
        else:
            start_time = self._parse_time(scenario.get('start_time'))
            end_time = self._parse_time(scenario.get('end_time'))
        
        if start_time >= end_time:
            raise ValueError("Start time must be before end time")
        
        return {
            'event_id': scenario.get('id', self._next_event_id()),
            'attack_type': scenario.get('type'),
            'start_time': start_time,
            'end_time': end_time,
            'target': scenario.get('target'),
            'target_b': scenario.get('target_b'),
            **scenario
        }
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime."""
        if isinstance(time_str, datetime):
            return time_str
        
        formats = [
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse time: {time_str}")
    
    def _next_event_id(self) -> int:
        """Generate next event ID."""
        self._events_counter += 1
        return self._events_counter