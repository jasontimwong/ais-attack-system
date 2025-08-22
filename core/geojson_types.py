"""GeoJSON type definitions for AIS data visualization."""

from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from dataclasses import dataclass, asdict
import json
import math
import numpy as np


# GeoJSON coordinate types
Coordinate = List[float]  # [longitude, latitude]
LinearRing = List[Coordinate]  # Closed ring of coordinates
Position = Union[Coordinate, List[Coordinate]]  # Single or multiple positions


class GeoJSONGeometry(TypedDict):
    """Base GeoJSON geometry structure."""
    type: Literal["Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon"]
    coordinates: Union[Coordinate, List[Coordinate], List[List[Coordinate]]]


class PointGeometry(TypedDict):
    """GeoJSON Point geometry."""
    type: Literal["Point"]
    coordinates: Coordinate


class LineStringGeometry(TypedDict):
    """GeoJSON LineString geometry."""
    type: Literal["LineString"]
    coordinates: List[Coordinate]


class PolygonGeometry(TypedDict):
    """GeoJSON Polygon geometry."""
    type: Literal["Polygon"]
    coordinates: List[LinearRing]


class VesselProperties(TypedDict):
    """Properties for vessel trajectory features."""
    mmsi: str
    vessel_name: Optional[str]
    vessel_type: Literal["baseline", "attack"]
    attack_type: Optional[Literal["ghost", "spoof", "zone_violation"]]
    time_start: str  # ISO 8601 format
    time_end: str
    min_speed: float  # knots
    max_speed: float
    avg_speed: float
    total_distance: float  # nautical miles
    point_count: int  # Number of points in trajectory


class ZoneProperties(TypedDict):
    """Properties for guard zone features."""
    zone_id: str
    zone_type: Literal["restricted", "warning", "prohibited"]
    zone_name: Optional[str]
    restrictions: Optional[str]


class Feature(TypedDict):
    """GeoJSON Feature structure."""
    type: Literal["Feature"]
    geometry: GeoJSONGeometry
    properties: Dict[str, Any]
    id: Optional[Union[str, int]]


class FeatureCollection(TypedDict):
    """GeoJSON FeatureCollection structure."""
    type: Literal["FeatureCollection"]
    features: List[Feature]
    metadata: Optional[Dict[str, Any]]


@dataclass
class BoundingBox:
    """Geographic bounding box."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    
    def to_list(self) -> List[float]:
        """Convert to GeoJSON bbox format [min_lon, min_lat, max_lon, max_lat]."""
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]
    
    def contains(self, lon: float, lat: float) -> bool:
        """Check if a point is within the bounding box."""
        return (self.min_lon <= lon <= self.max_lon and 
                self.min_lat <= lat <= self.max_lat)


@dataclass
class TrajectoryPoint:
    """Single point in a vessel trajectory."""
    timestamp: float  # Unix timestamp
    longitude: float
    latitude: float
    speed: Optional[float] = None  # knots
    heading: Optional[float] = None  # degrees
    
    def to_coordinate(self) -> Coordinate:
        """Convert to GeoJSON coordinate format."""
        return [self.longitude, self.latitude]
    
    def is_valid(self) -> bool:
        """Check if coordinates are valid."""
        return (-180 <= self.longitude <= 180 and 
                -90 <= self.latitude <= 90)


def create_point_feature(
    point: TrajectoryPoint,
    properties: Optional[Dict[str, Any]] = None
) -> Feature:
    """Create a Point feature from a TrajectoryPoint."""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": point.to_coordinate()
        },
        "properties": properties or {}
    }


def create_linestring_feature(
    points: List[TrajectoryPoint],
    properties: Optional[Dict[str, Any]] = None
) -> Feature:
    """Create a LineString feature from a list of TrajectoryPoints."""
    coordinates = [p.to_coordinate() for p in points if p.is_valid()]
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
        },
        "properties": properties or {}
    }


def create_polygon_feature(
    coordinates: List[List[float]],
    properties: Optional[Dict[str, Any]] = None
) -> Feature:
    """Create a Polygon feature from coordinates."""
    # Ensure the polygon is closed
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates]
        },
        "properties": properties or {}
    }


def create_feature_collection(
    features: List[Feature],
    metadata: Optional[Dict[str, Any]] = None
) -> FeatureCollection:
    """Create a FeatureCollection from a list of features."""
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": metadata
    }


def validate_geojson(data: Dict[str, Any]) -> bool:
    """Basic validation of GeoJSON structure."""
    if "type" not in data:
        return False
    
    if data["type"] == "FeatureCollection":
        return "features" in data and isinstance(data["features"], list)
    elif data["type"] == "Feature":
        return "geometry" in data and "properties" in data
    else:
        return "coordinates" in data


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN and Infinity values."""
    
    def encode(self, o):
        """Override encode to handle NaN values in the serialized string."""
        # First, let the default encoder do its work
        result = super().encode(o)
        # Replace NaN and Infinity in the result string
        result = result.replace('NaN', 'null')
        result = result.replace('Infinity', 'null')
        result = result.replace('-Infinity', 'null')
        return result
    
    def iterencode(self, o, _one_shot=False):
        """Override iterencode for streaming encoding."""
        for chunk in super().iterencode(o, _one_shot):
            # Replace NaN and Infinity in each chunk
            chunk = chunk.replace('NaN', 'null')
            chunk = chunk.replace('Infinity', 'null')
            chunk = chunk.replace('-Infinity', 'null')
            yield chunk


def save_geojson(data: Union[Feature, FeatureCollection], filepath: str) -> None:
    """Save GeoJSON data to file with NaN/Infinity handling."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)


def load_geojson(filepath: str) -> Union[Feature, FeatureCollection]:
    """Load GeoJSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not validate_geojson(data):
        raise ValueError(f"Invalid GeoJSON structure in {filepath}")
    
    return data