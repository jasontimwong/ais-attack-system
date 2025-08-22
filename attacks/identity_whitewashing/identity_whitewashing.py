#!/usr/bin/env python3
"""
S9: Identity Whitewashing Attack Plugin
船只身份漂白与变更 - Ship changes AIS identity periodically to evade tracking

Attack Description:
- Vessel periodically changes its identity (MMSI, name, call sign)
- Maintains operational continuity while appearing as different vessels
- Used to evade sanctions, hide illegal activities, or reset reputation
"""

import logging
import random
import string
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import pandas as pd
import numpy as np

from core.attack_plugin_v2 import StreamingAttackPlugin, DataPatch, PatchType

logger = logging.getLogger(__name__)


class IdentityWhitewashingAttack(StreamingAttackPlugin):
    """
    S9: Identity Whitewashing Attack
    
    Attack Pattern:
    1. Monitor vessel trajectory and timing
    2. At specified intervals or locations, change identity
    3. Generate new MMSI, vessel name, and call sign
    4. Maintain trajectory continuity but with new identity
    5. Optionally change vessel type and dimensions
    """
    
    def __init__(self, params: Dict[str, Any]):
        """Initialize identity whitewashing attack"""
        super().__init__(params)
        
        # Attack parameters with TEST_MODE support
        self.whitewash_interval = self._get_param_with_test_mode(
            params, 'whitewash_interval', 7200, 300  # 2 hours default, 5 min in test
        )
        self.location_based = params.get('location_based', False)
        self.change_vessel_type = params.get('change_vessel_type', True)
        self.identity_pool_size = params.get('identity_pool_size', 10)
        
        # Identity generation parameters
        self.mmsi_prefix = params.get('mmsi_prefix', '412')  # China prefix
        self.vessel_name_patterns = params.get('vessel_name_patterns', [
            'CARGO {num}', 'TRADER {num}', 'PACIFIC {num}', 
            'FORTUNE {num}', 'DRAGON {num}', 'PHOENIX {num}'
        ])
        
        # Tracking state
        self.vessel_states: Dict[int, Dict[str, Any]] = {}
        self.identity_changes: List[Dict[str, Any]] = []
        self.used_identities: set = set()
        
    def _generate_new_identity(self, original_mmsi: int, vessel_type: int) -> Dict[str, Any]:
        """Generate a new vessel identity"""
        # Generate unique MMSI
        new_mmsi = None
        attempts = 0
        while new_mmsi is None or new_mmsi in self.used_identities:
            suffix = ''.join(random.choices(string.digits, k=6))
            new_mmsi = int(self.mmsi_prefix + suffix)
            attempts += 1
            if attempts > 100:
                # Fallback to completely random MMSI
                new_mmsi = random.randint(200000000, 799999999)
                break
        
        self.used_identities.add(new_mmsi)
        
        # Generate vessel name
        pattern = random.choice(self.vessel_name_patterns)
        vessel_name = pattern.format(num=random.randint(1, 999))
        
        # Generate call sign (ITU format)
        call_sign = ''.join(random.choices(string.ascii_uppercase, k=2)) + \
                   ''.join(random.choices(string.digits, k=4))
        
        # Generate IMO number (optional, some vessels don't have)
        imo = f"IMO{random.randint(1000000, 9999999)}" if random.random() > 0.3 else ""
        
        # Vessel type modification
        if self.change_vessel_type:
            # Similar vessel types to avoid suspicion
            type_variations = {
                70: [70, 71, 72, 73, 74, 79],  # Cargo variations
                80: [80, 81, 82, 83, 84, 89],  # Tanker variations
                30: [30, 31, 32, 33, 34, 35],  # Fishing variations
            }
            base_type = (vessel_type // 10) * 10
            if base_type in type_variations:
                vessel_type = random.choice(type_variations[base_type])
        
        return {
            'mmsi': new_mmsi,
            'vessel_name': vessel_name,
            'call_sign': call_sign,
            'imo': imo,
            'vessel_type': vessel_type,
            'original_mmsi': original_mmsi,
            'change_time': datetime.now()
        }
    
    def _should_whitewash(self, vessel_state: Dict[str, Any], 
                         current_time: datetime) -> bool:
        """Determine if vessel should change identity"""
        if 'last_whitewash' not in vessel_state:
            vessel_state['last_whitewash'] = vessel_state['first_seen']
        
        time_since_last = (current_time - vessel_state['last_whitewash']).total_seconds()
        
        if self.location_based:
            # TODO: Implement location-based triggers (ports, boundaries)
            pass
        
        return time_since_last >= self.whitewash_interval
    
    def process_chunk(self, chunk_data: DataPatch) -> Iterator[DataPatch]:
        """Process a chunk of AIS data"""
        if chunk_data.patch_type == PatchType.DELETE:
            yield chunk_data
            return
            
        df = chunk_data.new_records.copy() if chunk_data.new_records is not None else chunk_data.modifications.copy()
        
        # Track vessels and check for whitewashing
        for idx, row in df.iterrows():
            mmsi = row['MMSI']
            timestamp = pd.to_datetime(row['BaseDateTime'])
            
            # Initialize vessel state if new
            if mmsi not in self.vessel_states:
                self.vessel_states[mmsi] = {
                    'original_mmsi': mmsi,
                    'current_mmsi': mmsi,
                    'first_seen': timestamp,
                    'last_whitewash': timestamp,
                    'whitewash_count': 0,
                    'identity_history': [],
                    'original_properties': {
                        'vessel_name': row.get('VesselName', ''),
                        'call_sign': row.get('CallSign', ''),
                        'imo': row.get('IMO', ''),
                        'vessel_type': row.get('VesselType', 0)
                    }
                }
            
            vessel_state = self.vessel_states[mmsi]
            
            # Check if whitewashing should occur
            if self._should_whitewash(vessel_state, timestamp):
                # Generate new identity
                new_identity = self._generate_new_identity(
                    vessel_state['original_mmsi'],
                    vessel_state['original_properties']['vessel_type']
                )
                
                # Update vessel state
                vessel_state['current_mmsi'] = new_identity['mmsi']
                vessel_state['last_whitewash'] = timestamp
                vessel_state['whitewash_count'] += 1
                vessel_state['identity_history'].append(new_identity)
                
                # Record identity change
                self.identity_changes.append({
                    'original_mmsi': vessel_state['original_mmsi'],
                    'old_mmsi': mmsi,
                    'new_mmsi': new_identity['mmsi'],
                    'timestamp': timestamp,
                    'position': (row['LAT'], row['LON']),
                    'new_identity': new_identity
                })
                
                logger.info(f"[S9] Vessel {mmsi} whitewashed to {new_identity['mmsi']} "
                           f"({new_identity['vessel_name']})")
            
            # Apply current identity to the row
            if vessel_state['whitewash_count'] > 0:
                current_identity = vessel_state['identity_history'][-1]
                df.at[idx, 'MMSI'] = current_identity['mmsi']
                df.at[idx, 'VesselName'] = current_identity['vessel_name']
                df.at[idx, 'CallSign'] = current_identity['call_sign']
                df.at[idx, 'IMO'] = current_identity['imo']
                df.at[idx, 'VesselType'] = current_identity['vessel_type']
                
                # Add attack metadata
                df.at[idx, 'attack_type'] = 'identity_whitewashing'
                df.at[idx, 'original_mmsi'] = vessel_state['original_mmsi']
                df.at[idx, 'whitewash_count'] = vessel_state['whitewash_count']
        
        yield DataPatch(
            patch_type=PatchType.MODIFY,
            modifications=df
        )
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """Get attack execution summary"""
        return {
            'attack_type': 'identity_whitewashing',
            'total_vessels_tracked': len(self.vessel_states),
            'vessels_whitewashed': sum(1 for v in self.vessel_states.values() 
                                     if v['whitewash_count'] > 0),
            'total_identity_changes': len(self.identity_changes),
            'max_whitewash_count': max((v['whitewash_count'] 
                                       for v in self.vessel_states.values()), 
                                      default=0),
            'parameters': {
                'whitewash_interval': self.whitewash_interval,
                'location_based': self.location_based,
                'change_vessel_type': self.change_vessel_type,
                'identity_pool_size': self.identity_pool_size
            }
        }
    
    def generate_patches(self, 
                        chunk_iter: Iterator[pd.DataFrame],
                        config: Dict[str, Any]) -> Iterator[DataPatch]:
        """Generate data patches from chunk iterator"""
        for chunk in chunk_iter:
            patch = DataPatch(
                patch_type=PatchType.INSERT,
                new_records=chunk
            )
            yield from self.process_chunk(patch)
    
    def validate_params(self) -> bool:
        """Validate plugin parameters"""
        if self.whitewash_interval < 60:
            logger.warning("Whitewash interval < 60s may be too frequent")
        if self.identity_pool_size < 5:
            logger.warning("Identity pool size < 5 may lead to MMSI collisions")
        return True
    
    def get_impact_type(self) -> str:
        """Get the impact type for this attack"""
        return "identity_manipulation"


def test_identity_whitewashing():
    """Test the identity whitewashing attack"""
    import os
    os.environ['AIS_TEST_MODE'] = '1'
    
    # Create test data
    now = datetime.now()
    test_data = []
    
    # Generate 30 minutes of data for a vessel
    for i in range(30):  # 30 minutes, 1 minute intervals
        test_data.append({
            'MMSI': 123456789,
            'BaseDateTime': now + timedelta(minutes=i),
            'LAT': 37.7749 + i * 0.001,
            'LON': -122.4194 + i * 0.001,
            'VesselName': 'CARGO STAR',
            'CallSign': 'AB1234',
            'VesselType': 70,
            'SOG': 12.5,
            'COG': 45.0
        })
    
    df = pd.DataFrame(test_data)
    
    # Create attack
    attack = IdentityWhitewashingAttack({
        'whitewash_interval': 7200,  # Will use 300s (5 min) in TEST_MODE
        'change_vessel_type': True
    })
    
    # Process data
    patch = DataPatch(
        patch_type=PatchType.INSERT,
        new_records=df
    )
    
    results = list(attack.process_chunk(patch))
    
    if results:
        result_df = results[0].modifications
        
        # Check for identity changes
        changed_vessels = result_df[result_df['MMSI'] != 123456789]
        if not changed_vessels.empty:
            print("Identity changes detected:")
            for idx, row in changed_vessels.iterrows():
                print(f"  Time: {row['BaseDateTime']}, New MMSI: {row['MMSI']}, "
                      f"New Name: {row['VesselName']}")
    
    # Print summary
    summary = attack.get_attack_summary()
    print(f"\nAttack Summary:")
    print(f"  Total identity changes: {summary['total_identity_changes']}")
    print(f"  Vessels whitewashed: {summary['vessels_whitewashed']}")
    print(f"  Whitewash interval: {attack.whitewash_interval}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_identity_whitewashing()