"""
Attack effectiveness evaluation module.

Analyzes the causal relationship between ghost vessels and real vessel responses
to determine if attacks successfully influenced vessel behavior.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from .cpa_utils import calc_min_cpa_track, calc_simple_cpa

logger = logging.getLogger(__name__)


class AttackEffectivenessEvaluator:
    """
    Evaluates attack effectiveness by establishing causal links between
    ghost vessels and real vessel responses.
    """
    
    def __init__(self, cpa_threshold_nm: float = 1.0, tcpa_threshold_min: float = 15.0):
        """
        Initialize evaluator.
        
        Args:
            cpa_threshold_nm: CPA threshold for considering vessels threatened (nm)
            tcpa_threshold_min: TCPA threshold for considering vessels threatened (minutes)
        """
        self.cpa_threshold_nm = cpa_threshold_nm
        self.tcpa_threshold_min = tcpa_threshold_min
        
    def evaluate_attack(self, 
                       diff_data: Dict[str, Any],
                       impact_events: Dict[str, Any],
                       attack_data: pd.DataFrame,
                       baseline_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate attack effectiveness by analyzing ghost-vessel interactions.
        
        Args:
            diff_data: Dictionary from diff.json containing attack metadata
            impact_events: Dictionary from impact_events.json containing vessel responses
            attack_data: Attack dataset including ghost vessels
            baseline_data: Baseline dataset without attacks
            
        Returns:
            Dictionary containing effectiveness evaluation results
        """
        results = []
        
        # Process each attack event from diff.json
        for event in diff_data.get('events', []):
            if event['attack_type'] != 'S1':  # Only process false target attacks
                continue
                
            event_result = self._evaluate_single_attack(
                event, impact_events, attack_data, baseline_data
            )
            results.append(event_result)
        
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'cpa_threshold_nm': self.cpa_threshold_nm,
                'tcpa_threshold_min': self.tcpa_threshold_min
            },
            'attacks': results
        }
    
    def _evaluate_single_attack(self,
                               attack_event: Dict[str, Any],
                               impact_events: Dict[str, Any],
                               attack_data: pd.DataFrame,
                               baseline_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a single attack event."""
        
        attack_id = attack_event['event_id']
        
        # Extract ghost vessel MMSI from attack metrics
        ghost_mmsi = self._find_ghost_mmsi(attack_event, attack_data)
        if ghost_mmsi is None:
            logger.warning(f"Could not find ghost MMSI for attack {attack_id}")
            return self._create_failed_result(attack_id, "Ghost vessel not found")
        
        # Get ghost vessel trajectory
        ghost_track = attack_data[attack_data['MMSI'] == ghost_mmsi].copy()
        if ghost_track.empty:
            return self._create_failed_result(attack_id, "Ghost trajectory not found")
        
        ghost_start_time = ghost_track['BaseDateTime'].min()
        ghost_end_time = ghost_track['BaseDateTime'].max()
        
        # Find threatened vessels (vessels that come close to ghost)
        threatened_vessels = self._find_threatened_vessels(
            ghost_track, baseline_data, attack_event
        )
        
        # Analyze vessel responses
        vessel_responses = []
        for vessel_mmsi in threatened_vessels:
            response = self._analyze_vessel_response(
                vessel_mmsi, ghost_track, impact_events, 
                baseline_data, attack_data, ghost_start_time
            )
            if response:
                vessel_responses.append(response)
        
        # Calculate overall effectiveness
        overall_effective = len(vessel_responses) > 0
        
        # Calculate reaction delay (min time from ghost appearance to first response)
        reaction_delay_sec = None
        if vessel_responses:
            response_times = [
                (datetime.fromisoformat(r['start_ts'].replace('Z', '+00:00')) - ghost_start_time).total_seconds()
                for r in vessel_responses if r.get('start_ts')
            ]
            if response_times:
                reaction_delay_sec = min(response_times)
        
        # Determine severity based on CPA and response magnitude
        severity = self._determine_severity(attack_event, vessel_responses)
        high_risk = severity == 'high'
        
        return {
            'attack_id': attack_id,
            'ghost_mmsi': ghost_mmsi,
            'ghost_start_ts': ghost_start_time.isoformat() + 'Z',
            'ghost_end_ts': ghost_end_time.isoformat() + 'Z',
            'threatened_vessels': threatened_vessels,
            'vessel_responses': vessel_responses,
            'overall_effective': overall_effective,
            'severity': severity,
            'high_risk': high_risk,
            'reaction_delay_sec': reaction_delay_sec,
            'consistency_check': self._check_consistency(attack_event, vessel_responses)
        }
    
    def _find_ghost_mmsi(self, attack_event: Dict[str, Any], attack_data: pd.DataFrame) -> Optional[int]:
        """Find ghost vessel MMSI from attack event."""
        
        # Check if MMSI is directly provided in metrics
        metrics = attack_event.get('metrics', {})
        if 'ghost_mmsi' in metrics:
            return metrics['ghost_mmsi']
        
        # Find new MMSI that appears only in attack data during event window
        start_ts = pd.to_datetime(attack_event['start_ts'])
        end_ts = pd.to_datetime(attack_event['end_ts'])
        
        # Get MMSIs in attack window
        attack_window = attack_data[
            (attack_data['BaseDateTime'] >= start_ts) & 
            (attack_data['BaseDateTime'] <= end_ts)
        ]
        
        # Find MMSI that starts with 999 (typical ghost vessel pattern)
        ghost_mmsis = attack_window[attack_window['MMSI'] >= 999000000]['MMSI'].unique()
        
        if len(ghost_mmsis) == 1:
            return int(ghost_mmsis[0])
        elif len(ghost_mmsis) > 1:
            # Return the one with most records
            mmsi_counts = attack_window[attack_window['MMSI'].isin(ghost_mmsis)]['MMSI'].value_counts()
            return int(mmsi_counts.index[0])
        
        return None
    
    def _find_threatened_vessels(self, 
                                ghost_track: pd.DataFrame,
                                baseline_data: pd.DataFrame,
                                attack_event: Dict[str, Any]) -> List[int]:
        """Find vessels threatened by ghost vessel."""
        
        threatened = set()
        
        # Time window for analysis
        start_time = ghost_track['BaseDateTime'].min()
        end_time = ghost_track['BaseDateTime'].max() + timedelta(minutes=10)
        
        # Get vessels in time window
        vessels_in_window = baseline_data[
            (baseline_data['BaseDateTime'] >= start_time) &
            (baseline_data['BaseDateTime'] <= end_time)
        ]['MMSI'].unique()
        
        # Check each vessel for CPA with ghost
        for vessel_mmsi in vessels_in_window:
            vessel_track = baseline_data[
                (baseline_data['MMSI'] == vessel_mmsi) &
                (baseline_data['BaseDateTime'] >= start_time) &
                (baseline_data['BaseDateTime'] <= end_time)
            ].copy()
            
            if len(vessel_track) < 5:  # Need enough points for analysis
                continue
            
            # Calculate minimum CPA with ghost
            cpa_result = calc_min_cpa_track(ghost_track, vessel_track)
            
            # Check if vessel is threatened
            if (cpa_result['min_cpa_nm'] < self.cpa_threshold_nm and
                0 < cpa_result['tcpa_min'] < self.tcpa_threshold_min):
                threatened.add(int(vessel_mmsi))
                logger.info(f"Vessel {vessel_mmsi} threatened by ghost: CPA={cpa_result['min_cpa_nm']:.3f}nm")
        
        return list(threatened)
    
    def _analyze_vessel_response(self,
                                vessel_mmsi: int,
                                ghost_track: pd.DataFrame,
                                impact_events: Dict[str, Any],
                                baseline_data: pd.DataFrame,
                                attack_data: pd.DataFrame,
                                ghost_start_time: datetime) -> Optional[Dict[str, Any]]:
        """Analyze if and how a vessel responded to ghost threat."""
        
        # Find vessel maneuvers from impact events
        vessel_maneuvers = []
        for event in impact_events.get('events', []):
            if event.get('vessel_mmsi') == vessel_mmsi:
                vessel_maneuvers.append(event)
        
        if not vessel_maneuvers:
            return None
        
        # Get the earliest maneuver after ghost appearance
        valid_maneuvers = []
        for maneuver in vessel_maneuvers:
            maneuver_start = pd.to_datetime(maneuver['start_time'].replace('Z', '+00:00'))
            if maneuver_start >= ghost_start_time:
                valid_maneuvers.append(maneuver)
        
        if not valid_maneuvers:
            return None
        
        # Sort by start time and take first
        maneuver = sorted(valid_maneuvers, key=lambda m: m['start_time'])[0]
        
        # Calculate CPA before and after maneuver
        maneuver_start = pd.to_datetime(maneuver['start_time'].replace('Z', '+00:00'))
        
        # Get vessel track before maneuver
        vessel_track_before = baseline_data[
            (baseline_data['MMSI'] == vessel_mmsi) &
            (baseline_data['BaseDateTime'] >= ghost_start_time) &
            (baseline_data['BaseDateTime'] < maneuver_start)
        ].copy()
        
        # Get vessel track after maneuver from attack data (includes maneuver)
        vessel_track_after = attack_data[
            (attack_data['MMSI'] == vessel_mmsi) &
            (attack_data['BaseDateTime'] >= maneuver_start)
        ].copy()
        
        # Calculate CPA before maneuver (projected)
        cpa_before = self._calculate_projected_cpa(vessel_track_before, ghost_track)
        
        # Calculate CPA after maneuver (actual)
        cpa_after = calc_min_cpa_track(vessel_track_after, ghost_track)
        
        # Determine response status
        status = 'unknown'
        if cpa_before and cpa_after['min_cpa_nm'] < float('inf'):
            if cpa_after['min_cpa_nm'] >= 1.5 * self.cpa_threshold_nm:
                status = 'success'
            elif cpa_after['min_cpa_nm'] > cpa_before:
                status = 'partial'
            else:
                status = 'failed'
        
        return {
            'mmsi': vessel_mmsi,
            'response': maneuver.get('maneuver_type', 'unknown'),
            'start_ts': maneuver['start_time'],
            'end_ts': maneuver.get('end_time'),
            'cpa_before_nm': round(cpa_before, 3) if cpa_before else None,
            'cpa_after_nm': round(cpa_after['min_cpa_nm'], 3) if cpa_after['min_cpa_nm'] < float('inf') else None,
            'status': status,
            'trigger': 'ghost_threat'
        }
    
    def _calculate_projected_cpa(self, 
                                vessel_track: pd.DataFrame, 
                                ghost_track: pd.DataFrame) -> Optional[float]:
        """Calculate projected CPA if vessel continues on current course."""
        
        if vessel_track.empty or ghost_track.empty:
            return None
        
        # Get last known state of vessel
        last_vessel_state = vessel_track.iloc[-1]
        
        # Find corresponding ghost position
        ghost_at_time = ghost_track[
            ghost_track['BaseDateTime'] >= last_vessel_state['BaseDateTime']
        ]
        
        if ghost_at_time.empty:
            return None
        
        ghost_state = ghost_at_time.iloc[0]
        
        # Calculate simple CPA
        cpa_nm, _ = calc_simple_cpa(
            last_vessel_state['LAT'], last_vessel_state['LON'],
            last_vessel_state['SOG'], last_vessel_state['COG'],
            ghost_state['LAT'], ghost_state['LON'],
            ghost_state['SOG'], ghost_state['COG']
        )
        
        return cpa_nm
    
    def _check_consistency(self, 
                          attack_event: Dict[str, Any],
                          vessel_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check consistency between diff.json and calculated values."""
        
        checks = {
            'cpa_match': False,
            'time_window_valid': False,
            'anomalies': []
        }
        
        # Check CPA consistency
        reported_cpa = attack_event.get('metrics', {}).get('min_cpa_nm')
        if reported_cpa and vessel_responses:
            calculated_cpas = [r['cpa_before_nm'] for r in vessel_responses if r.get('cpa_before_nm')]
            if calculated_cpas:
                min_calculated = min(calculated_cpas)
                if abs(reported_cpa - min_calculated) < 0.01:
                    checks['cpa_match'] = True
                else:
                    checks['anomalies'].append(
                        f"CPA mismatch: reported={reported_cpa:.3f}, calculated={min_calculated:.3f}"
                    )
        
        # Check time window
        attack_end = pd.to_datetime(attack_event['end_ts'])
        response_times = [
            pd.to_datetime(r['start_ts'].replace('Z', '+00:00')) 
            for r in vessel_responses if r.get('start_ts')
        ]
        
        if response_times:
            max_delay = timedelta(minutes=10)
            valid_times = all(t <= attack_end + max_delay for t in response_times)
            checks['time_window_valid'] = valid_times
            if not valid_times:
                checks['anomalies'].append("Response outside expected time window")
        
        return checks
    
    def _create_failed_result(self, attack_id: int, reason: str) -> Dict[str, Any]:
        """Create a result for failed attack evaluation."""
        return {
            'attack_id': attack_id,
            'ghost_mmsi': None,
            'threatened_vessels': [],
            'vessel_responses': [],
            'overall_effective': False,
            'reaction_delay_sec': None,
            'error': reason
        }
    
    def _determine_severity(self, attack_event: Dict[str, Any], 
                           vessel_responses: List[Dict[str, Any]]) -> str:
        """Determine attack severity based on CPA and response magnitude."""
        
        # Get minimum CPA from responses
        min_cpa = float('inf')
        max_heading_change = 0
        max_speed_drop_pct = 0
        
        for response in vessel_responses:
            # CPA before maneuver
            if response.get('cpa_before_nm'):
                min_cpa = min(min_cpa, response['cpa_before_nm'])
            
            # Check response magnitude
            if response.get('response') in ['turn_starboard', 'turn_port']:
                # Extract heading change from response or impact event
                # For now, assume standard 30 degree turn
                max_heading_change = max(max_heading_change, 30)
            
            # Check for speed reduction
            # This would need to be extracted from impact events
            # For now, we'll use a placeholder
        
        # Also check reported CPA from attack event
        reported_cpa = attack_event.get('metrics', {}).get('min_cpa_nm', float('inf'))
        min_cpa = min(min_cpa, reported_cpa)
        
        # Apply severity rules
        if min_cpa <= 0.1 and (max_heading_change >= 20 or max_speed_drop_pct >= 30):
            return 'high'
        elif min_cpa <= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Attack effectiveness results saved to {output_path}")