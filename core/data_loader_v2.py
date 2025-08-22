"""
Streaming Data Loader for large AIS CSV files.
Supports chunked reading, filtering, and caching.
"""

import csv
import logging
from datetime import datetime
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import pyarrow.parquet as pq

from .data_structures import AISRecord

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for data loading."""
    chunk_size: int = 1_000_000
    time_range: Optional[Tuple[datetime, datetime]] = None
    geo_bbox: Optional[Dict[str, float]] = None  # {min_lat, max_lat, min_lon, max_lon}
    target_mmsis: Optional[List[int]] = None
    cache_dir: Optional[Path] = None
    skip_validation: bool = False  # Skip all validation checks
    

class StreamingDataLoader:
    """
    Streaming data loader for large AIS CSV files.
    Processes data in chunks to minimize memory usage.
    """
    
    def __init__(self, config: Optional[LoaderConfig] = None):
        self.config = config or LoaderConfig()
        self.stats = {
            'total_records': 0,
            'valid_records': 0,
            'filtered_records': 0,
            'invalid_records': 0,
            'chunks_processed': 0,
            'unique_vessels': set()
        }
        
        # Setup cache directory if specified
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_csv_stream(self, filepath: str) -> Iterator[pd.DataFrame]:
        """
        Load AIS data from CSV file as a stream of chunks.
        
        Args:
            filepath: Path to the CSV file
            
        Yields:
            DataFrame chunks containing AIS records
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        logger.info(f"Starting streaming load from {filepath}")
        logger.info(f"Chunk size: {self.config.chunk_size:,} records")
        
        # Define column types for efficient parsing
        dtype_map = {
            'MMSI': 'int32',
            'LAT': 'float32',
            'LON': 'float32', 
            'SOG': 'float32',
            'COG': 'float32',
            'Heading': 'float32',
            'VesselName': 'str',
            'IMO': 'str',
            'CallSign': 'str',
            'VesselType': 'Int32',  # Nullable integer
            'Status': 'str',
            'Length': 'float32',
            'Width': 'float32',
            'Draft': 'float32',
            'Cargo': 'str',
            'TransceiverClass': 'str'
        }
        
        # Read in chunks
        chunk_iter = pd.read_csv(
            filepath,
            chunksize=self.config.chunk_size,
            dtype=dtype_map,
            parse_dates=['BaseDateTime'],
            date_format='%Y-%m-%dT%H:%M:%S',
            na_values=['', 'NA', 'null'],
            engine='c'  # C engine is faster
        )
        
        for chunk_num, chunk in enumerate(chunk_iter):
            self.stats['chunks_processed'] += 1
            
            # Clean and filter chunk
            logger.debug(f"Chunk {chunk_num + 1}: {len(chunk):,} raw records")
            filtered_chunk = self._process_chunk(chunk, chunk_num)
            
            if not filtered_chunk.empty:
                logger.debug(f"Chunk {chunk_num + 1}: {len(filtered_chunk):,} records after all filters")
                yield filtered_chunk
            else:
                logger.debug(f"Chunk {chunk_num + 1}: Empty after filtering")
            
            # Log progress
            if chunk_num % 10 == 0 or chunk_num == 0:
                logger.info(f"Processed {self.stats['chunks_processed']} chunks, "
                          f"{self.stats['valid_records']:,} valid records")
    
    def _process_chunk(self, chunk: pd.DataFrame, chunk_num: int) -> pd.DataFrame:
        """Process a single chunk: clean, validate, and filter."""
        initial_size = len(chunk)
        self.stats['total_records'] += initial_size
        
        # Drop rows with invalid timestamps (unless validation is skipped)
        if not self.config.skip_validation:
            chunk = chunk.dropna(subset=['BaseDateTime'])
        
        # Apply time range filter if specified
        if self.config.time_range:
            start_time, end_time = self.config.time_range
            before_time_filter = len(chunk)
            mask = (chunk['BaseDateTime'] >= start_time) & (chunk['BaseDateTime'] <= end_time)
            chunk = chunk[mask]
            time_filtered = before_time_filter - len(chunk)
            if time_filtered > 0:
                logger.debug(f"Chunk {chunk_num + 1}: Time filter removed {time_filtered} records")
            self.stats['filtered_records'] += time_filtered
        
        # Apply geographic bounding box filter if specified
        if self.config.geo_bbox:
            bbox = self.config.geo_bbox
            before_geo_filter = len(chunk)
            mask = (
                (chunk['LAT'] >= bbox['min_lat']) & 
                (chunk['LAT'] <= bbox['max_lat']) &
                (chunk['LON'] >= bbox['min_lon']) & 
                (chunk['LON'] <= bbox['max_lon'])
            )
            chunk = chunk[mask]
            geo_filtered = before_geo_filter - len(chunk)
            if geo_filtered > 0:
                logger.debug(f"Chunk {chunk_num + 1}: Geo filter removed {geo_filtered} records")
            self.stats['filtered_records'] += geo_filtered
        
        # Apply MMSI filter if specified
        if self.config.target_mmsis:
            chunk = chunk[chunk['MMSI'].isin(self.config.target_mmsis)]
        
        # Validate remaining records (unless validation is skipped)
        if not self.config.skip_validation:
            valid_mask = self._validate_records(chunk)
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                logger.debug(f"Chunk {chunk_num}: {invalid_count} invalid records removed")
                self.stats['invalid_records'] += invalid_count
                chunk = chunk[valid_mask]
        
        # Update statistics
        self.stats['valid_records'] += len(chunk)
        self.stats['unique_vessels'].update(chunk['MMSI'].unique())
        
        # Cache chunk if configured
        if self.config.cache_dir and not chunk.empty:
            self._cache_chunk(chunk, chunk_num)
        
        return chunk
    
    def _validate_records(self, chunk: pd.DataFrame) -> pd.Series:
        """Validate records in chunk, return boolean mask."""
        valid = pd.Series(True, index=chunk.index)
        
        # MMSI validation
        valid &= (chunk['MMSI'] > 0) & (chunk['MMSI'] <= 999999999)
        
        # Position validation
        valid &= (chunk['LAT'] >= -90) & (chunk['LAT'] <= 90)
        valid &= (chunk['LON'] >= -180) & (chunk['LON'] <= 180)
        
        # Speed validation
        valid &= (chunk['SOG'] >= 0) & (chunk['SOG'] <= 102.3)
        
        # Course validation
        valid &= (chunk['COG'] >= 0) & (chunk['COG'] <= 360)
        
        # Heading validation (if column exists)
        if 'Heading' in chunk.columns:
            valid &= (chunk['Heading'] >= 0) & (chunk['Heading'] <= 360)
        
        return valid
    
    def _cache_chunk(self, chunk: pd.DataFrame, chunk_num: int):
        """Cache chunk to parquet for faster subsequent access."""
        cache_file = self.config.cache_dir / f"chunk_{chunk_num:06d}.parquet"
        chunk.to_parquet(cache_file, engine='pyarrow', compression='snappy')
    
    def load_from_cache(self) -> Iterator[pd.DataFrame]:
        """Load data from cached parquet files."""
        if not self.config.cache_dir:
            raise ValueError("No cache directory configured")
        
        cache_files = sorted(self.config.cache_dir.glob("chunk_*.parquet"))
        logger.info(f"Loading from {len(cache_files)} cached chunks")
        
        for cache_file in cache_files:
            yield pd.read_parquet(cache_file, engine='pyarrow')
    
    def partition_by_vessel(self, chunk_iter: Iterator[pd.DataFrame], 
                          output_dir: Path) -> Dict[int, Path]:
        """
        Partition data by vessel MMSI for efficient per-vessel processing.
        
        Args:
            chunk_iter: Iterator of DataFrame chunks
            output_dir: Directory to write vessel partitions
            
        Returns:
            Dictionary mapping MMSI to partition file path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        vessel_files = {}
        vessel_writers = {}
        
        try:
            for chunk in chunk_iter:
                # Group by MMSI
                for mmsi, vessel_data in chunk.groupby('MMSI'):
                    if mmsi not in vessel_files:
                        # Create new parquet writer for this vessel
                        vessel_file = output_dir / f"vessel_{mmsi}.parquet"
                        vessel_files[mmsi] = vessel_file
                        vessel_writers[mmsi] = pq.ParquetWriter(
                            vessel_file, 
                            vessel_data.to_arrow().schema,
                            compression='snappy'
                        )
                    
                    # Write vessel data
                    vessel_writers[mmsi].write_table(vessel_data.to_arrow())
        
        finally:
            # Close all writers
            for writer in vessel_writers.values():
                writer.close()
        
        logger.info(f"Partitioned data for {len(vessel_files)} vessels")
        return vessel_files
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return {
            'total_records': self.stats['total_records'],
            'valid_records': self.stats['valid_records'],
            'filtered_records': self.stats['filtered_records'],
            'invalid_records': self.stats['invalid_records'],
            'chunks_processed': self.stats['chunks_processed'],
            'unique_vessels': len(self.stats['unique_vessels']),
            'vessel_mmsis': sorted(list(self.stats['unique_vessels']))
        }
    
    @staticmethod
    def estimate_memory_usage(filepath: str, sample_size: int = 10000) -> Dict[str, Any]:
        """
        Estimate memory usage by sampling the file.
        
        Args:
            filepath: Path to CSV file
            sample_size: Number of rows to sample
            
        Returns:
            Dictionary with memory estimates
        """
        # Read sample
        sample_df = pd.read_csv(filepath, nrows=sample_size)
        
        # Calculate memory usage
        memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)
        
        # Estimate total rows (rough)
        file_size = Path(filepath).stat().st_size
        avg_row_size = file_size / sample_size  # Very rough estimate
        estimated_rows = int(file_size / avg_row_size)
        
        return {
            'memory_per_row_bytes': int(memory_per_row),
            'estimated_total_rows': estimated_rows,
            'estimated_memory_gb': round((memory_per_row * estimated_rows) / (1024**3), 2),
            'recommended_chunk_size': min(5_000_000, max(100_000, int(2e9 / memory_per_row)))  # Target ~2GB per chunk
        }