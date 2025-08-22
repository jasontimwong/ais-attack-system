"""
Impact Exporter for Impact Simulator.

Exports impact.csv and impact_events.json showing vessel avoidance maneuvers.
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import os

from .traffic_snapshot_builder import TrafficSnapshot
from .maneuver_simulator import ManeuverEvent

logger = logging.getLogger(__name__)


class ImpactExporter:
    """
    Exports impact simulation results to CSV and JSON formats.
    
    Generates:
    - impact.csv: Complete vessel trajectories with maneuvers applied
    - impact_events.json: Event log of all maneuvers triggered
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize impact exporter.
        
        Args:
            output_dir: Directory to write output files
        """
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def export_impact_data(self, snapshots: Dict[datetime, TrafficSnapshot], 
                          maneuver_events: List[ManeuverEvent],
                          original_attack_csv: str = None) -> Tuple[str, str]:
        """
        Export complete impact simulation results.
        
        Args:
            snapshots: Traffic snapshots with maneuvers applied
            maneuver_events: List of all maneuver events
            original_attack_csv: Path to original attack.csv for metadata
            
        Returns:
            Tuple of (impact.csv path, impact_events.json path)
        """
        logger.info("Exporting impact simulation results")
        
        # Export impact CSV
        impact_csv_path = self._export_impact_csv(snapshots, original_attack_csv)
        
        # Export impact events JSON
        impact_events_path = self._export_impact_events(maneuver_events)
        
        logger.info(f"Exported impact data to {self.output_dir}")
        return impact_csv_path, impact_events_path
    
    def _export_impact_csv(self, snapshots: Dict[datetime, TrafficSnapshot], 
                          original_attack_csv: str = None) -> str:
        """
        Export impact.csv with complete vessel trajectories.
        
        Args:
            snapshots: Traffic snapshots with maneuvers applied
            original_attack_csv: Path to original attack.csv for reference
            
        Returns:
            Path to exported impact.csv
        """
        csv_path = os.path.join(self.output_dir, 'impact.csv')
        
        # Convert snapshots to records
        records = []
        for timestamp in sorted(snapshots.keys()):
            snapshot = snapshots[timestamp]
            
            for vessel in snapshot.get_all_vessels():
                record = self._vessel_to_csv_record(vessel, timestamp)
                records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Add metadata from original attack CSV if available
        if original_attack_csv and os.path.exists(original_attack_csv):
            df = self._enrich_with_original_metadata(df, original_attack_csv)
        
        # Sort by timestamp and MMSI
        df = df.sort_values(['BaseDateTime', 'MMSI'])
        
        # Export to CSV
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Exported {len(df)} impact records to {csv_path}")
        return csv_path
    
    def _vessel_to_csv_record(self, vessel, timestamp: datetime) -> Dict:
        """Convert VesselSnapshot to CSV record format."""
        return {
            'MMSI': vessel.mmsi,
            'BaseDateTime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'LAT': vessel.lat,
            'LON': vessel.lon,
            'SOG': vessel.sog,
            'COG': vessel.cog,
            'Heading': vessel.heading,
            'VesselName': vessel.vessel_name,
            'IMO': '',  # Not available in simulation
            'CallSign': '',  # Not available in simulation
            'VesselType': 0,  # Default value
            'Status': 0,  # Default value
            'Length': '',  # Not available in simulation
            'Width': '',  # Not available in simulation
            'Draft': '',  # Not available in simulation
            'Cargo': '',  # Not available in simulation
            'TransceiverClass': 'A',  # Default value
            'ROT': '',  # Not available in simulation
            'SpecialManoeuvre': '',  # Not available in simulation
            'Spare': '',  # Not available in simulation
            'RAIM': '',  # Not available in simulation
            'CommState': ''  # Not available in simulation
        }
    
    def _enrich_with_original_metadata(self, df: pd.DataFrame, original_csv: str) -> pd.DataFrame:
        """
        Enrich impact data with metadata from original attack.csv.
        
        Args:
            df: Impact DataFrame
            original_csv: Path to original attack.csv
            
        Returns:
            Enriched DataFrame
        """
        try:
            # Load original data
            original_df = pd.read_csv(original_csv)
            original_df['BaseDateTime'] = pd.to_datetime(original_df['BaseDateTime'])
            
            # Create lookup for vessel metadata
            vessel_metadata = {}
            for _, row in original_df.iterrows():
                mmsi = row['MMSI']
                if mmsi not in vessel_metadata:
                    vessel_metadata[mmsi] = {
                        'VesselName': row.get('VesselName', ''),
                        'IMO': row.get('IMO', ''),
                        'CallSign': row.get('CallSign', ''),
                        'VesselType': row.get('VesselType', 0),
                        'Length': row.get('Length', ''),
                        'Width': row.get('Width', ''),
                        'Draft': row.get('Draft', ''),
                        'Cargo': row.get('Cargo', ''),
                        'TransceiverClass': row.get('TransceiverClass', 'A')
                    }
            
            # Enrich impact data
            for mmsi, metadata in vessel_metadata.items():
                mask = df['MMSI'] == mmsi
                for field, value in metadata.items():
                    if field in df.columns:
                        df.loc[mask, field] = value
            
            logger.info(f"Enriched impact data with metadata for {len(vessel_metadata)} vessels")
            
        except Exception as e:
            logger.warning(f"Could not enrich with original metadata: {e}")
        
        return df
    
    def _export_impact_events(self, maneuver_events: List[ManeuverEvent]) -> str:
        """
        Export impact_events.json with maneuver event log.
        
        Args:
            maneuver_events: List of all maneuver events
            
        Returns:
            Path to exported impact_events.json
        """
        json_path = os.path.join(self.output_dir, 'impact_events.json')
        
        # Create events data structure
        events_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_events': len(maneuver_events),
                'event_types': self._get_event_type_summary(maneuver_events)
            },
            'events': [event.to_dict() for event in maneuver_events]
        }
        
        # Export to JSON
        with open(json_path, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        logger.info(f"Exported {len(maneuver_events)} impact events to {json_path}")
        return json_path
    
    def _get_event_type_summary(self, maneuver_events: List[ManeuverEvent]) -> Dict[str, int]:
        """Get summary of event types."""
        event_types = {}
        for event in maneuver_events:
            event_type = event.maneuver_type
            event_types[event_type] = event_types.get(event_type, 0) + 1
        return event_types
    
    def export_simulation_summary(self, snapshots: Dict[datetime, TrafficSnapshot], 
                                maneuver_events: List[ManeuverEvent],
                                original_attack_csv: str = None) -> str:
        """
        Export simulation summary with key metrics.
        
        Args:
            snapshots: Traffic snapshots with maneuvers applied
            maneuver_events: List of all maneuver events
            original_attack_csv: Path to original attack.csv
            
        Returns:
            Path to exported summary.json
        """
        summary_path = os.path.join(self.output_dir, 'simulation_summary.json')
        
        # Calculate summary metrics
        total_vessels = len(set(vessel.mmsi for snapshot in snapshots.values() 
                              for vessel in snapshot.get_all_vessels()))
        
        vessels_with_maneuvers = len(set(event.vessel_mmsi for event in maneuver_events))
        
        maneuver_types = self._get_event_type_summary(maneuver_events)
        
        # Time range
        timestamps = sorted(snapshots.keys())
        time_range = {
            'start': timestamps[0].isoformat() if timestamps else None,
            'end': timestamps[-1].isoformat() if timestamps else None,
            'duration_hours': (timestamps[-1] - timestamps[0]).total_seconds() / 3600 if len(timestamps) > 1 else 0
        }
        
        summary_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'original_attack_csv': original_attack_csv,
                'simulation_parameters': {
                    'cpa_threshold_nm': 1.0,
                    'tcpa_threshold_seconds': 900,
                    'time_step_seconds': 10,
                    'max_turn_rate_deg_per_min': 15.0,
                    'standard_turn_degrees': 30.0
                }
            },
            'simulation_results': {
                'total_vessels': total_vessels,
                'vessels_with_maneuvers': vessels_with_maneuvers,
                'impact_rate': vessels_with_maneuvers / total_vessels if total_vessels > 0 else 0,
                'total_maneuvers': len(maneuver_events),
                'maneuver_types': maneuver_types,
                'time_range': time_range,
                'snapshots_generated': len(snapshots)
            }
        }
        
        # Export summary
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Exported simulation summary to {summary_path}")
        return summary_path