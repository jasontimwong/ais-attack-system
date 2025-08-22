#!/usr/bin/env python3
"""
Data Quality Checker for AIS Attack Generation System

This tool validates the quality and integrity of generated attack data.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

class DataQualityChecker:
    """
    Comprehensive data quality validation for AIS attack scenarios
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.quality_report = {
            'timestamp': datetime.now().isoformat(),
            'checks_performed': [],
            'issues_found': [],
            'quality_score': 0.0,
            'recommendations': []
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load quality check configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'physics': {
                'max_speed': 50.0,  # knots
                'max_acceleration': 2.0,  # knots/minute
                'max_turn_rate': 10.0,  # degrees/second
                'min_cpa': 0.01  # nautical miles
            },
            'temporal': {
                'max_time_gap': 300,  # seconds
                'min_message_interval': 1.0  # seconds
            },
            'spatial': {
                'max_position_jump': 1.0,  # nautical miles
                'coordinate_precision': 6  # decimal places
            },
            'quality_thresholds': {
                'excellent': 0.95,
                'good': 0.85,
                'fair': 0.70,
                'poor': 0.50
            }
        }
    
    def check_trajectory_file(self, file_path: str) -> Dict:
        """
        Check quality of a single trajectory file
        
        Args:
            file_path: Path to trajectory file (GeoJSON or CSV)
            
        Returns:
            Quality check results
        """
        try:
            # Load trajectory data
            if file_path.endswith('.geojson'):
                trajectory = self._load_geojson_trajectory(file_path)
            elif file_path.endswith('.csv'):
                trajectory = self._load_csv_trajectory(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            if trajectory.empty:
                return {
                    'file': file_path,
                    'valid': False,
                    'error': 'Empty trajectory data',
                    'quality_score': 0.0
                }
            
            # Perform quality checks
            results = {
                'file': file_path,
                'valid': True,
                'trajectory_points': len(trajectory),
                'time_span': self._calculate_time_span(trajectory),
                'checks': {}
            }
            
            # Run individual checks
            results['checks']['temporal_consistency'] = self._check_temporal_consistency(trajectory)
            results['checks']['spatial_consistency'] = self._check_spatial_consistency(trajectory)
            results['checks']['physics_compliance'] = self._check_physics_compliance(trajectory)
            results['checks']['data_completeness'] = self._check_data_completeness(trajectory)
            results['checks']['coordinate_validity'] = self._check_coordinate_validity(trajectory)
            results['checks']['duplicate_detection'] = self._check_duplicates(trajectory)
            
            # Calculate overall quality score
            results['quality_score'] = self._calculate_quality_score(results['checks'])
            results['quality_level'] = self._get_quality_level(results['quality_score'])
            
            return results
            
        except Exception as e:
            return {
                'file': file_path,
                'valid': False,
                'error': str(e),
                'quality_score': 0.0
            }
    
    def _load_geojson_trajectory(self, file_path: str) -> pd.DataFrame:
        """Load trajectory from GeoJSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        points = []
        for feature in data.get('features', []):
            if feature['geometry']['type'] == 'Point':
                coords = feature['geometry']['coordinates']
                props = feature['properties']
                
                points.append({
                    'timestamp': props.get('timestamp'),
                    'lat': coords[1],
                    'lon': coords[0],
                    'speed': props.get('speed'),
                    'course': props.get('course'),
                    'mmsi': props.get('mmsi')
                })
        
        df = pd.DataFrame(points)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def _load_csv_trajectory(self, file_path: str) -> pd.DataFrame:
        """Load trajectory from CSV file"""
        df = pd.read_csv(file_path)
        
        # Standardize column names
        column_mapping = {
            'latitude': 'lat',
            'longitude': 'lon',
            'sog': 'speed',
            'cog': 'course'
        }
        
        df = df.rename(columns=column_mapping)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def _calculate_time_span(self, trajectory: pd.DataFrame) -> Optional[float]:
        """Calculate trajectory time span in seconds"""
        if 'timestamp' in trajectory.columns and len(trajectory) > 1:
            time_span = (trajectory['timestamp'].max() - trajectory['timestamp'].min()).total_seconds()
            return time_span
        return None
    
    def _check_temporal_consistency(self, trajectory: pd.DataFrame) -> Dict:
        """Check temporal consistency of trajectory"""
        if 'timestamp' not in trajectory.columns:
            return {'valid': False, 'error': 'No timestamp column found'}
        
        issues = []
        
        # Check for time gaps
        time_diffs = trajectory['timestamp'].diff().dt.total_seconds()
        max_gap = time_diffs.max()
        large_gaps = (time_diffs > self.config['temporal']['max_time_gap']).sum()
        
        if large_gaps > 0:
            issues.append(f"{large_gaps} time gaps > {self.config['temporal']['max_time_gap']}s")
        
        # Check for minimum intervals
        min_interval = time_diffs.min()
        if min_interval < self.config['temporal']['min_message_interval']:
            issues.append(f"Minimum interval {min_interval:.1f}s too small")
        
        # Check for duplicate timestamps
        duplicates = trajectory['timestamp'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate timestamps")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'max_time_gap': max_gap,
            'min_interval': min_interval,
            'duplicate_timestamps': duplicates
        }
    
    def _check_spatial_consistency(self, trajectory: pd.DataFrame) -> Dict:
        """Check spatial consistency of trajectory"""
        required_cols = ['lat', 'lon']
        if not all(col in trajectory.columns for col in required_cols):
            return {'valid': False, 'error': 'Missing coordinate columns'}
        
        issues = []
        
        # Calculate position jumps
        position_jumps = []
        for i in range(1, len(trajectory)):
            lat1, lon1 = trajectory.iloc[i-1][['lat', 'lon']]
            lat2, lon2 = trajectory.iloc[i][['lat', 'lon']]
            
            # Haversine distance in nautical miles
            distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            position_jumps.append(distance)
        
        if position_jumps:
            max_jump = max(position_jumps)
            large_jumps = sum(1 for jump in position_jumps 
                            if jump > self.config['spatial']['max_position_jump'])
            
            if large_jumps > 0:
                issues.append(f"{large_jumps} position jumps > {self.config['spatial']['max_position_jump']}nm")
        
        # Check coordinate bounds (basic sanity check)
        lat_bounds = (-90, 90)
        lon_bounds = (-180, 180)
        
        invalid_lat = ((trajectory['lat'] < lat_bounds[0]) | 
                      (trajectory['lat'] > lat_bounds[1])).sum()
        invalid_lon = ((trajectory['lon'] < lon_bounds[0]) | 
                      (trajectory['lon'] > lon_bounds[1])).sum()
        
        if invalid_lat > 0:
            issues.append(f"{invalid_lat} invalid latitude values")
        if invalid_lon > 0:
            issues.append(f"{invalid_lon} invalid longitude values")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'max_position_jump': max(position_jumps) if position_jumps else 0,
            'invalid_coordinates': invalid_lat + invalid_lon
        }
    
    def _check_physics_compliance(self, trajectory: pd.DataFrame) -> Dict:
        """Check physics compliance of trajectory"""
        required_cols = ['speed', 'course', 'timestamp']
        if not all(col in trajectory.columns for col in required_cols):
            return {'valid': False, 'error': 'Missing required columns for physics check'}
        
        issues = []
        
        # Check speed limits
        max_speed = trajectory['speed'].max()
        if max_speed > self.config['physics']['max_speed']:
            issues.append(f"Maximum speed {max_speed:.1f} kts exceeds limit")
        
        # Check acceleration
        time_diffs = trajectory['timestamp'].diff().dt.total_seconds()
        speed_diffs = trajectory['speed'].diff()
        
        # Convert to knots per minute
        accelerations = (speed_diffs / (time_diffs / 60)).abs()
        max_acceleration = accelerations.max()
        
        if max_acceleration > self.config['physics']['max_acceleration']:
            issues.append(f"Maximum acceleration {max_acceleration:.1f} kts/min exceeds limit")
        
        # Check turn rates
        course_diffs = trajectory['course'].diff().abs()
        # Handle course wrap-around
        course_diffs = np.minimum(course_diffs, 360 - course_diffs)
        
        turn_rates = (course_diffs / time_diffs).abs()  # degrees per second
        max_turn_rate = turn_rates.max()
        
        if max_turn_rate > self.config['physics']['max_turn_rate']:
            issues.append(f"Maximum turn rate {max_turn_rate:.1f} deg/s exceeds limit")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'max_speed': max_speed,
            'max_acceleration': max_acceleration,
            'max_turn_rate': max_turn_rate
        }
    
    def _check_data_completeness(self, trajectory: pd.DataFrame) -> Dict:
        """Check data completeness"""
        issues = []
        
        # Check for missing values
        missing_counts = trajectory.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            issues.append(f"{total_missing} missing values found")
        
        # Check for required columns
        required_columns = ['timestamp', 'lat', 'lon', 'speed', 'course']
        missing_columns = [col for col in required_columns if col not in trajectory.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        completeness_ratio = 1.0 - (total_missing / (len(trajectory) * len(trajectory.columns)))
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'completeness_ratio': completeness_ratio,
            'missing_values': dict(missing_counts)
        }
    
    def _check_coordinate_validity(self, trajectory: pd.DataFrame) -> Dict:
        """Check coordinate precision and validity"""
        if 'lat' not in trajectory.columns or 'lon' not in trajectory.columns:
            return {'valid': False, 'error': 'Missing coordinate columns'}
        
        issues = []
        
        # Check precision (number of decimal places)
        lat_precision = trajectory['lat'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
        lon_precision = trajectory['lon'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
        
        low_precision_lat = (lat_precision < 4).sum()
        low_precision_lon = (lon_precision < 4).sum()
        
        if low_precision_lat > 0:
            issues.append(f"{low_precision_lat} latitude values with low precision")
        if low_precision_lon > 0:
            issues.append(f"{low_precision_lon} longitude values with low precision")
        
        # Check for obviously invalid coordinates (e.g., 0,0 or repeated values)
        zero_coords = ((trajectory['lat'] == 0) & (trajectory['lon'] == 0)).sum()
        if zero_coords > 0:
            issues.append(f"{zero_coords} zero coordinate pairs")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'avg_lat_precision': lat_precision.mean(),
            'avg_lon_precision': lon_precision.mean(),
            'zero_coordinates': zero_coords
        }
    
    def _check_duplicates(self, trajectory: pd.DataFrame) -> Dict:
        """Check for duplicate records"""
        issues = []
        
        # Check for exact duplicates
        exact_duplicates = trajectory.duplicated().sum()
        if exact_duplicates > 0:
            issues.append(f"{exact_duplicates} exact duplicate records")
        
        # Check for duplicate positions
        if 'lat' in trajectory.columns and 'lon' in trajectory.columns:
            position_duplicates = trajectory[['lat', 'lon']].duplicated().sum()
            if position_duplicates > 0:
                issues.append(f"{position_duplicates} duplicate positions")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'exact_duplicates': exact_duplicates,
            'position_duplicates': position_duplicates if 'lat' in trajectory.columns else 0
        }
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in nautical miles"""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in nautical miles
        r = 3440.065
        
        return c * r
    
    def _calculate_quality_score(self, checks: Dict) -> float:
        """Calculate overall quality score based on check results"""
        weights = {
            'temporal_consistency': 0.2,
            'spatial_consistency': 0.25,
            'physics_compliance': 0.25,
            'data_completeness': 0.15,
            'coordinate_validity': 0.1,
            'duplicate_detection': 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for check_name, weight in weights.items():
            if check_name in checks:
                check_result = checks[check_name]
                if isinstance(check_result, dict) and 'valid' in check_result:
                    score = 1.0 if check_result['valid'] else 0.0
                    
                    # Adjust score based on specific metrics
                    if check_name == 'data_completeness' and 'completeness_ratio' in check_result:
                        score = check_result['completeness_ratio']
                    
                    total_score += score * weight
                    total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level description based on score"""
        thresholds = self.config['quality_thresholds']
        
        if score >= thresholds['excellent']:
            return 'Excellent'
        elif score >= thresholds['good']:
            return 'Good'
        elif score >= thresholds['fair']:
            return 'Fair'
        elif score >= thresholds['poor']:
            return 'Poor'
        else:
            return 'Critical'
    
    def check_directory(self, directory_path: str) -> Dict:
        """
        Check quality of all trajectory files in a directory
        
        Args:
            directory_path: Path to directory containing trajectory files
            
        Returns:
            Comprehensive quality report
        """
        directory = Path(directory_path)
        if not directory.exists():
            return {'error': f"Directory not found: {directory_path}"}
        
        # Find trajectory files
        trajectory_files = []
        for ext in ['*.geojson', '*.csv']:
            trajectory_files.extend(directory.glob(ext))
        
        if not trajectory_files:
            return {'error': f"No trajectory files found in {directory_path}"}
        
        # Check each file
        file_results = []
        total_score = 0.0
        valid_files = 0
        
        for file_path in trajectory_files:
            result = self.check_trajectory_file(str(file_path))
            file_results.append(result)
            
            if result.get('valid', False):
                total_score += result.get('quality_score', 0.0)
                valid_files += 1
        
        # Calculate overall statistics
        avg_quality = total_score / valid_files if valid_files > 0 else 0.0
        
        return {
            'directory': directory_path,
            'total_files': len(trajectory_files),
            'valid_files': valid_files,
            'invalid_files': len(trajectory_files) - valid_files,
            'average_quality_score': avg_quality,
            'overall_quality_level': self._get_quality_level(avg_quality),
            'file_results': file_results
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AIS Attack Data Quality Checker")
    parser.add_argument("path", help="Path to trajectory file or directory")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--output", "-o", help="Output report file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = DataQualityChecker(args.config)
    
    # Check path
    path = Path(args.path)
    if path.is_file():
        results = checker.check_trajectory_file(str(path))
    elif path.is_dir():
        results = checker.check_directory(str(path))
    else:
        print(f"Error: Path not found: {args.path}")
        sys.exit(1)
    
    # Display results
    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
    
    if path.is_file():
        # Single file results
        print(f"Quality Check Results for: {results['file']}")
        print(f"Valid: {'✅' if results['valid'] else '❌'}")
        print(f"Quality Score: {results.get('quality_score', 0.0):.3f}")
        print(f"Quality Level: {results.get('quality_level', 'Unknown')}")
        
        if args.verbose and 'checks' in results:
            print("\nDetailed Check Results:")
            for check_name, check_result in results['checks'].items():
                status = '✅' if check_result.get('valid', False) else '❌'
                print(f"  {status} {check_name.replace('_', ' ').title()}")
                
                if not check_result.get('valid', False) and 'issues' in check_result:
                    for issue in check_result['issues']:
                        print(f"    - {issue}")
    
    else:
        # Directory results
        print(f"Quality Check Results for Directory: {results['directory']}")
        print(f"Total Files: {results['total_files']}")
        print(f"Valid Files: {results['valid_files']}")
        print(f"Invalid Files: {results['invalid_files']}")
        print(f"Average Quality Score: {results['average_quality_score']:.3f}")
        print(f"Overall Quality Level: {results['overall_quality_level']}")
        
        if args.verbose:
            print("\nFile-by-File Results:")
            for file_result in results['file_results']:
                status = '✅' if file_result.get('valid', False) else '❌'
                score = file_result.get('quality_score', 0.0)
                filename = Path(file_result['file']).name
                print(f"  {status} {filename}: {score:.3f}")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {args.output}")

if __name__ == "__main__":
    main()
