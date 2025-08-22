#!/usr/bin/env python3
"""
GeoJSON Converter for AIS Attack Generation System

This tool converts various data formats to GeoJSON for visualization and analysis.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import csv

class GeoJSONConverter:
    """
    Convert AIS trajectory data to GeoJSON format
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.txt', '.nmea']
    
    def convert_csv_to_geojson(self, csv_path: str, output_path: str = None) -> str:
        """
        Convert CSV trajectory data to GeoJSON
        
        Args:
            csv_path: Path to CSV file
            output_path: Output GeoJSON file path
            
        Returns:
            Path to generated GeoJSON file
        """
        # Read CSV data
        df = pd.read_csv(csv_path)
        
        # Standardize column names
        column_mapping = {
            'latitude': 'lat',
            'longitude': 'lon',
            'sog': 'speed',
            'cog': 'course',
            'time': 'timestamp',
            'datetime': 'timestamp'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Validate required columns
        required_columns = ['lat', 'lon']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = Path(csv_path).with_suffix('.geojson')
        
        # Convert to GeoJSON
        geojson = self._dataframe_to_geojson(df)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2, default=str)
        
        return str(output_path)
    
    def convert_json_to_geojson(self, json_path: str, output_path: str = None) -> str:
        """
        Convert JSON trajectory data to GeoJSON
        
        Args:
            json_path: Path to JSON file
            output_path: Output GeoJSON file path
            
        Returns:
            Path to generated GeoJSON file
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of trajectory points
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'trajectory' in data:
                df = pd.DataFrame(data['trajectory'])
            elif 'features' in data:
                # Already GeoJSON format
                if output_path is None:
                    output_path = Path(json_path).with_suffix('.geojson')
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return str(output_path)
            else:
                # Assume single trajectory record
                df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = Path(json_path).with_suffix('.geojson')
        
        # Convert to GeoJSON
        geojson = self._dataframe_to_geojson(df)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2, default=str)
        
        return str(output_path)
    
    def convert_nmea_to_geojson(self, nmea_path: str, output_path: str = None) -> str:
        """
        Convert NMEA data to GeoJSON
        
        Args:
            nmea_path: Path to NMEA file
            output_path: Output GeoJSON file path
            
        Returns:
            Path to generated GeoJSON file
        """
        trajectory_points = []
        
        with open(nmea_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse NMEA sentence
                    point = self._parse_nmea_sentence(line)
                    if point:
                        trajectory_points.append(point)
                except Exception as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        if not trajectory_points:
            raise ValueError("No valid NMEA data found")
        
        # Convert to DataFrame
        df = pd.DataFrame(trajectory_points)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = Path(nmea_path).with_suffix('.geojson')
        
        # Convert to GeoJSON
        geojson = self._dataframe_to_geojson(df)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2, default=str)
        
        return str(output_path)
    
    def _dataframe_to_geojson(self, df: pd.DataFrame) -> Dict:
        """
        Convert DataFrame to GeoJSON format
        
        Args:
            df: DataFrame with trajectory data
            
        Returns:
            GeoJSON dictionary
        """
        features = []
        
        for idx, row in df.iterrows():
            # Extract coordinates
            if 'lat' not in row or 'lon' not in row:
                continue
            
            lat, lon = row['lat'], row['lon']
            
            # Skip invalid coordinates
            if pd.isna(lat) or pd.isna(lon):
                continue
            
            # Create feature properties
            properties = {}
            for col, value in row.items():
                if col not in ['lat', 'lon'] and not pd.isna(value):
                    # Convert numpy types to native Python types
                    if isinstance(value, np.integer):
                        value = int(value)
                    elif isinstance(value, np.floating):
                        value = float(value)
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    elif pd.isna(value):
                        continue
                    
                    properties[col] = value
            
            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)]
                },
                "properties": properties
            }
            
            features.append(feature)
        
        # Create GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generated_by": "AIS Attack System GeoJSON Converter",
                "generated_at": datetime.now().isoformat(),
                "total_points": len(features)
            }
        }
        
        return geojson
    
    def _parse_nmea_sentence(self, sentence: str) -> Optional[Dict]:
        """
        Parse NMEA sentence to extract position data
        
        Args:
            sentence: NMEA sentence string
            
        Returns:
            Dictionary with parsed data or None
        """
        if not sentence.startswith('$'):
            return None
        
        # Remove checksum if present
        if '*' in sentence:
            sentence = sentence.split('*')[0]
        
        fields = sentence.split(',')
        
        if len(fields) < 3:
            return None
        
        sentence_type = fields[0][3:6]  # Extract sentence type (e.g., GGA, RMC)
        
        if sentence_type == 'GGA':
            return self._parse_gga_sentence(fields)
        elif sentence_type == 'RMC':
            return self._parse_rmc_sentence(fields)
        elif sentence_type == 'VDM' or sentence_type == 'VDO':
            return self._parse_ais_sentence(fields)
        
        return None
    
    def _parse_gga_sentence(self, fields: List[str]) -> Optional[Dict]:
        """Parse NMEA GGA sentence"""
        if len(fields) < 15:
            return None
        
        try:
            # Extract time
            time_str = fields[1]
            if len(time_str) >= 6:
                hours = int(time_str[:2])
                minutes = int(time_str[2:4])
                seconds = float(time_str[4:])
            else:
                hours = minutes = seconds = 0
            
            # Extract latitude
            lat_str = fields[2]
            lat_dir = fields[3]
            if lat_str and lat_dir:
                lat_deg = int(lat_str[:2])
                lat_min = float(lat_str[2:])
                lat = lat_deg + lat_min / 60.0
                if lat_dir == 'S':
                    lat = -lat
            else:
                return None
            
            # Extract longitude
            lon_str = fields[4]
            lon_dir = fields[5]
            if lon_str and lon_dir:
                lon_deg = int(lon_str[:3])
                lon_min = float(lon_str[3:])
                lon = lon_deg + lon_min / 60.0
                if lon_dir == 'W':
                    lon = -lon
            else:
                return None
            
            return {
                'lat': lat,
                'lon': lon,
                'timestamp': f"{hours:02d}:{minutes:02d}:{seconds:06.3f}",
                'source': 'NMEA_GGA'
            }
            
        except (ValueError, IndexError):
            return None
    
    def _parse_rmc_sentence(self, fields: List[str]) -> Optional[Dict]:
        """Parse NMEA RMC sentence"""
        if len(fields) < 13:
            return None
        
        try:
            # Check if data is valid
            if fields[2] != 'A':  # 'A' = valid, 'V' = invalid
                return None
            
            # Extract time and date
            time_str = fields[1]
            date_str = fields[9]
            
            if len(time_str) >= 6 and len(date_str) >= 6:
                hours = int(time_str[:2])
                minutes = int(time_str[2:4])
                seconds = float(time_str[4:])
                
                day = int(date_str[:2])
                month = int(date_str[2:4])
                year = 2000 + int(date_str[4:6])
                
                timestamp = f"{year}-{month:02d}-{day:02d}T{hours:02d}:{minutes:02d}:{seconds:06.3f}Z"
            else:
                timestamp = None
            
            # Extract latitude
            lat_str = fields[3]
            lat_dir = fields[4]
            if lat_str and lat_dir:
                lat_deg = int(lat_str[:2])
                lat_min = float(lat_str[2:])
                lat = lat_deg + lat_min / 60.0
                if lat_dir == 'S':
                    lat = -lat
            else:
                return None
            
            # Extract longitude
            lon_str = fields[5]
            lon_dir = fields[6]
            if lon_str and lon_dir:
                lon_deg = int(lon_str[:3])
                lon_min = float(lon_str[3:])
                lon = lon_deg + lon_min / 60.0
                if lon_dir == 'W':
                    lon = -lon
            else:
                return None
            
            # Extract speed and course
            speed = float(fields[7]) if fields[7] else 0.0  # Speed in knots
            course = float(fields[8]) if fields[8] else 0.0  # Course in degrees
            
            return {
                'lat': lat,
                'lon': lon,
                'speed': speed,
                'course': course,
                'timestamp': timestamp,
                'source': 'NMEA_RMC'
            }
            
        except (ValueError, IndexError):
            return None
    
    def _parse_ais_sentence(self, fields: List[str]) -> Optional[Dict]:
        """Parse NMEA AIS sentence (simplified)"""
        # This is a simplified AIS parser
        # In practice, you would use a proper AIS decoder library
        try:
            if len(fields) < 6:
                return None
            
            # Extract basic AIS information
            # This is a placeholder - actual AIS decoding is complex
            return {
                'source': 'NMEA_AIS',
                'raw_sentence': ','.join(fields)
            }
            
        except Exception:
            return None
    
    def convert_file(self, input_path: str, output_path: str = None) -> str:
        """
        Convert file to GeoJSON based on file extension
        
        Args:
            input_path: Path to input file
            output_path: Output GeoJSON file path
            
        Returns:
            Path to generated GeoJSON file
        """
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        file_ext = input_file.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if file_ext == '.csv':
            return self.convert_csv_to_geojson(input_path, output_path)
        elif file_ext == '.json':
            return self.convert_json_to_geojson(input_path, output_path)
        elif file_ext in ['.nmea', '.txt']:
            return self.convert_nmea_to_geojson(input_path, output_path)
        else:
            raise ValueError(f"Conversion not implemented for: {file_ext}")
    
    def convert_directory(self, input_dir: str, output_dir: str = None) -> List[str]:
        """
        Convert all supported files in a directory to GeoJSON
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            
        Returns:
            List of generated GeoJSON file paths
        """
        input_path = Path(input_dir)
        
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"Invalid input directory: {input_dir}")
        
        if output_dir is None:
            output_dir = input_path / "geojson_output"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        converted_files = []
        
        # Find all supported files
        for ext in self.supported_formats:
            for file_path in input_path.glob(f"*{ext}"):
                try:
                    output_file = output_path / f"{file_path.stem}.geojson"
                    result_path = self.convert_file(str(file_path), str(output_file))
                    converted_files.append(result_path)
                    print(f"✅ Converted: {file_path.name} -> {Path(result_path).name}")
                except Exception as e:
                    print(f"❌ Failed to convert {file_path.name}: {e}")
        
        return converted_files

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Convert trajectory data to GeoJSON format")
    parser.add_argument("input", help="Input file or directory path")
    parser.add_argument("--output", "-o", help="Output file or directory path")
    parser.add_argument("--recursive", "-r", action="store_true", 
                       help="Process directories recursively")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    converter = GeoJSONConverter()
    
    try:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Convert single file
            output_path = converter.convert_file(str(input_path), args.output)
            print(f"✅ Converted: {input_path} -> {output_path}")
            
        elif input_path.is_dir():
            # Convert directory
            converted_files = converter.convert_directory(str(input_path), args.output)
            
            print(f"\n📊 Conversion Summary:")
            print(f"Total files converted: {len(converted_files)}")
            
            if args.verbose:
                print("\nConverted files:")
                for file_path in converted_files:
                    print(f"  • {file_path}")
        
        else:
            print(f"❌ Error: Path not found: {args.input}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
