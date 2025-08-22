"""Constants for AIS data processing."""

# Coordinate validation ranges
LAT_MIN = -90.0
LAT_MAX = 90.0
LON_MIN = -180.0
LON_MAX = 180.0

# Speed thresholds (knots)
MIN_VALID_SPEED = 0.0
MAX_VALID_SPEED = 50.0  # Most vessels don't exceed 50 knots
STATIONARY_SPEED_THRESHOLD = 0.5

# Time thresholds (seconds)
MAX_TIME_GAP = 3600  # 1 hour - larger gaps indicate missing data
MIN_TIME_INTERVAL = 1  # Minimum 1 second between points
INTERPOLATION_TIME_GAP = 300  # 5 minutes

# Sampling parameters
DEFAULT_SAMPLE_RATE = 0.01  # Keep 1% of points
S1_TIME_INTERVAL = 60  # 1 minute for S1 scenarios
S2_TIME_INTERVAL = 300  # 5 minutes for S2 scenarios
DOUGLAS_PEUCKER_TOLERANCE = 0.001  # ~100m at equator

# Trajectory thresholds
MIN_TRAJECTORY_POINTS = 10  # Minimum points for valid trajectory
MIN_TRAJECTORY_DISTANCE = 0.1  # Minimum distance in nautical miles
SIGNIFICANT_HEADING_CHANGE = 30.0  # degrees
SIGNIFICANT_SPEED_CHANGE = 5.0  # knots

# File size limits
MAX_GEOJSON_SIZE_MB = 50  # Maximum output file size
CHUNK_SIZE = 10000  # Rows to process at once

# Zone types
ZONE_TYPES = ["restricted", "warning", "prohibited"]

# Vessel types
VESSEL_TYPES = ["baseline", "attack"]
ATTACK_TYPES = ["ghost", "spoof", "zone_violation"]

# Column name variations
REQUIRED_COLUMNS = ["mmsi", "latitude", "longitude", "timestamp"]
OPTIONAL_COLUMNS = ["speed", "heading", "vessel_name", "vessel_type"]

# Default values
DEFAULT_VESSEL_NAME = "Unknown Vessel"
DEFAULT_SPEED = 0.0
DEFAULT_HEADING = 0.0

# Processing settings
MULTIPROCESSING_WORKERS = 4  # Default number of workers
PROGRESS_UPDATE_INTERVAL = 1000  # Update progress every N rows

# Output formats
COORDINATE_PRECISION = 6  # Decimal places for coordinates
SPEED_PRECISION = 1  # Decimal places for speed
DISTANCE_PRECISION = 2  # Decimal places for distance

# Error messages
ERROR_INVALID_COORDINATES = "Invalid coordinates: lat={}, lon={}"
ERROR_MISSING_COLUMN = "Missing required column: {}"
ERROR_EMPTY_TRAJECTORY = "Empty trajectory for vessel {}"
ERROR_CONVERSION_FAILED = "Failed to convert {} to GeoJSON: {}"