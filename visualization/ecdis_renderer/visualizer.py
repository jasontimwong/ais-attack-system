#!/usr/bin/env python3
"""
Visualizer - Module for creating ECDIS-style visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os


class Visualizer:
    """Create ECDIS-style visualizations of AIS attack scenarios"""
    
    # ECDIS-style color scheme
    COLORS = {
        'ghost': '#FF0000',        # Red for ghost vessel
        'target': '#0066CC',       # Deep blue for target
        'impacted': '#FF8C00',     # Orange for impacted vessels
        'nearby': '#808080',       # Gray for other vessels
        'background': '#E8F4F8',   # Light blue-gray background
        'grid': '#B0C4DE',         # Light steel blue for grid
        'text': '#1C1C1C'          # Almost black for text
    }
    
    # Additional colors for multiple impacted vessels
    IMPACT_COLORS = ['#FF8C00', '#FF69B4', '#32CD32', '#9370DB', '#FFD700']
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_overview(self, baseline_tracks: Dict, attack_tracks: Dict, 
                       maneuver_tracks: Dict, vessel_categories: Dict,
                       impact_events: List[Dict], attack_window: Dict,
                       cpa_info: Optional[Dict] = None) -> str:
        """Create dual-panel overview visualization"""
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        
        # Set ECDIS-style background
        for ax in [ax1, ax2]:
            ax.set_facecolor(self.COLORS['background'])
            ax.grid(True, color=self.COLORS['grid'], alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Get coordinate bounds
        all_lats, all_lons = self._get_coordinate_bounds(
            baseline_tracks, attack_tracks, maneuver_tracks
        )
        
        # Add some padding
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        padding = 0.1
        
        for ax in [ax1, ax2]:
            ax.set_xlim(min(all_lons) - lon_range * padding, 
                       max(all_lons) + lon_range * padding)
            ax.set_ylim(min(all_lats) - lat_range * padding, 
                       max(all_lats) + lat_range * padding)
        
        # Panel 1: Baseline (no ghost)
        self._plot_baseline_panel(ax1, baseline_tracks, vessel_categories)
        
        # Panel 2: Attack + Maneuvers
        self._plot_attack_panel(ax2, attack_tracks, maneuver_tracks, 
                               vessel_categories, impact_events, cpa_info)
        
        # Set titles and labels
        attack_start = attack_window['start'].strftime('%H:%M:%S')
        attack_end = attack_window['end'].strftime('%H:%M:%S')
        
        ax1.set_title('Baseline Traffic (No Attack)', fontsize=14, fontweight='bold')
        ax2.set_title(f'Attack Scenario ({attack_start} - {attack_end})', 
                     fontsize=14, fontweight='bold')
        
        for ax in [ax1, ax2]:
            ax.set_xlabel('Longitude (°E)', fontsize=10)
            ax.set_ylabel('Latitude (°N)', fontsize=10)
            ax.tick_params(labelsize=8)
        
        # Add legend
        self._add_legend(fig)
        
        # Add CPA information if available
        if cpa_info and impact_events:
            cpa_text = f"CPA: {impact_events[0]['cpa_nm']:.3f} nm"
            fig.text(0.5, 0.02, cpa_text, ha='center', fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'overview.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def _plot_baseline_panel(self, ax, baseline_tracks: Dict, vessel_categories: Dict):
        """Plot baseline traffic without ghost vessel"""
        
        # Plot each vessel category
        for category, vessels in vessel_categories.items():
            if category == 'ghost':
                continue  # No ghost in baseline
                
            color = self.COLORS[category]
            
            for i, mmsi in enumerate(vessels):
                if mmsi not in baseline_tracks or baseline_tracks[mmsi].empty:
                    continue
                    
                track = baseline_tracks[mmsi]
                
                # Use different colors for multiple impacted vessels
                if category == 'impacted' and len(vessels) > 1:
                    color = self.IMPACT_COLORS[i % len(self.IMPACT_COLORS)]
                
                # Plot track
                self._plot_vessel_track(ax, track, color, 
                                       linewidth=2 if category == 'target' else 1,
                                       label=f"{category.title()} {mmsi}" if category != 'nearby' else None)
                
                # Add position markers every 30 seconds
                self._add_position_markers(ax, track, color, interval_seconds=30)
    
    def _plot_attack_panel(self, ax, attack_tracks: Dict, maneuver_tracks: Dict,
                          vessel_categories: Dict, impact_events: List[Dict],
                          cpa_info: Optional[Dict]):
        """Plot attack scenario with ghost vessel and maneuvers"""
        
        # Plot ghost vessel first (so it's in background)
        ghost_mmsi = vessel_categories['ghost'][0]
        if ghost_mmsi in attack_tracks:
            ghost_track = attack_tracks[ghost_mmsi]
            self._plot_vessel_track(ax, ghost_track, self.COLORS['ghost'], 
                                   linewidth=3, linestyle='--', 
                                   label=f"Ghost {ghost_mmsi}")
            self._add_position_markers(ax, ghost_track, self.COLORS['ghost'], 
                                     interval_seconds=30, marker='D')
        
        # Plot other vessels
        for category, vessels in vessel_categories.items():
            if category == 'ghost':
                continue
                
            color = self.COLORS[category]
            
            for i, mmsi in enumerate(vessels):
                # Use different colors for multiple impacted vessels
                if category == 'impacted' and len(vessels) > 1:
                    color = self.IMPACT_COLORS[i % len(self.IMPACT_COLORS)]
                
                # Plot attack phase track
                if mmsi in attack_tracks and not attack_tracks[mmsi].empty:
                    track = attack_tracks[mmsi]
                    self._plot_vessel_track(ax, track, color,
                                           linewidth=2 if category == 'target' else 1)
                
                # Plot maneuver track (may overlap with attack track)
                if mmsi in maneuver_tracks and not maneuver_tracks[mmsi].empty:
                    track = maneuver_tracks[mmsi]
                    if category == 'impacted' or category == 'target':
                        # Show maneuver as thicker dashed line
                        self._plot_vessel_track(ax, track, color,
                                               linewidth=3, linestyle=':')
                
                # Add position markers
                if mmsi in attack_tracks:
                    self._add_position_markers(ax, attack_tracks[mmsi], color, 
                                             interval_seconds=30)
        
        # Mark CPA points
        if cpa_info:
            for event in impact_events:
                if 'cpa_position' in event and event['cpa_position']['pos2']:
                    pos = event['cpa_position']['pos2']
                    ax.scatter(pos['lon'], pos['lat'], color='red', s=200, 
                             marker='x', linewidth=3, zorder=1000)
                    ax.annotate('CPA', (pos['lon'], pos['lat']), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, fontweight='bold', color='red')
        
        # Mark maneuver points
        for event in impact_events:
            if 'cpa_position' in event and event['cpa_position']['pos2']:
                # Mark where vessel started turning
                mmsi = event['mmsi']
                if mmsi in maneuver_tracks and not maneuver_tracks[mmsi].empty:
                    track = maneuver_tracks[mmsi]
                    # Find maneuver point (first significant heading change)
                    if len(track) > 1:
                        track['heading_change'] = track['COG'].diff().abs()
                        significant_changes = track[track['heading_change'] > 10]
                        if not significant_changes.empty:
                            maneuver_point = significant_changes.iloc[0]
                            ax.scatter(maneuver_point['LON'], maneuver_point['LAT'], 
                                     color='blue', s=150, marker='*', 
                                     linewidth=2, zorder=999)
                            ax.annotate('Turn', 
                                      (maneuver_point['LON'], maneuver_point['LAT']),
                                      xytext=(5, -15), textcoords='offset points',
                                      fontsize=8, color='blue')
    
    def _plot_vessel_track(self, ax, track: pd.DataFrame, color: str, 
                          linewidth: float = 1, linestyle: str = '-',
                          label: Optional[str] = None):
        """Plot a vessel track as a continuous line"""
        if len(track) < 2:
            return
            
        ax.plot(track['LON'], track['LAT'], color=color, linewidth=linewidth,
               linestyle=linestyle, label=label, alpha=0.8)
    
    def _add_position_markers(self, ax, track: pd.DataFrame, color: str,
                            interval_seconds: int = 30, marker: str = 'o'):
        """Add position markers at regular intervals"""
        if track.empty:
            return
            
        # Sort by time
        track = track.sort_values('BaseDateTime')
        
        # Sample at intervals
        start_time = track['BaseDateTime'].min()
        markers = []
        
        for _, row in track.iterrows():
            time_diff = (row['BaseDateTime'] - start_time).total_seconds()
            if time_diff % interval_seconds == 0:
                markers.append(row)
        
        if markers:
            marker_df = pd.DataFrame(markers)
            ax.scatter(marker_df['LON'], marker_df['LAT'], 
                      color=color, s=20, marker=marker, alpha=0.6, zorder=10)
    
    def _get_coordinate_bounds(self, *track_dicts) -> Tuple[List[float], List[float]]:
        """Get coordinate bounds from multiple track dictionaries"""
        all_lats = []
        all_lons = []
        
        for track_dict in track_dicts:
            for tracks in track_dict.values():
                if isinstance(tracks, pd.DataFrame) and not tracks.empty:
                    all_lats.extend(tracks['LAT'].values)
                    all_lons.extend(tracks['LON'].values)
        
        return all_lats, all_lons
    
    def _add_legend(self, fig):
        """Add comprehensive legend to figure"""
        legend_elements = [
            Line2D([0], [0], color=self.COLORS['ghost'], lw=3, 
                          linestyle='--', label='Ghost Vessel'),
            Line2D([0], [0], color=self.COLORS['target'], lw=2, 
                          label='Target Vessel'),
            Line2D([0], [0], color=self.COLORS['impacted'], lw=2, 
                          label='Impacted Vessels'),
            Line2D([0], [0], color=self.COLORS['nearby'], lw=1, 
                          label='Other Vessels'),
            Line2D([0], [0], color='black', lw=3, linestyle=':', 
                          label='Evasive Maneuver'),
            Line2D([0], [0], marker='x', color='red', lw=0, 
                          markersize=10, label='CPA Point'),
            Line2D([0], [0], marker='*', color='blue', lw=0, 
                          markersize=10, label='Turn Start')
        ]
        
        fig.legend(handles=legend_elements, loc='center', 
                  bbox_to_anchor=(0.5, 0.95), ncol=4, fontsize=10)
    
    def create_impact_metrics(self, impact_events: List[Dict], 
                            vessel_categories: Dict) -> str:
        """Create bar chart of impact metrics"""
        
        if not impact_events:
            print("No impact events to visualize")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        # Prepare data
        vessels = []
        heading_changes = []
        speed_drops = []
        
        for event in impact_events:
            vessel_name = event.get('vessel_name', f"MMSI {event['mmsi']}")
            vessels.append(vessel_name)
            heading_changes.append(event.get('delta_heading', 0))
            speed_drops.append(event.get('speed_drop', 0))
        
        # Bar positions
        x = np.arange(len(vessels))
        
        # Heading changes
        bars1 = ax1.bar(x, heading_changes, color=self.COLORS['impacted'], alpha=0.7)
        ax1.set_xlabel('Vessel')
        ax1.set_ylabel('Heading Change (degrees)')
        ax1.set_title('Evasive Maneuver - Heading Changes')
        ax1.set_xticks(x)
        ax1.set_xticklabels(vessels, rotation=45, ha='right')
        ax1.axhline(y=20, color='red', linestyle='--', alpha=0.5, 
                   label='Significant threshold')
        ax1.legend()
        
        # Add value labels on bars
        for bar, val in zip(bars1, heading_changes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}°', ha='center', va='bottom')
        
        # Speed drops
        bars2 = ax2.bar(x, speed_drops, color='#FF4500', alpha=0.7)
        ax2.set_xlabel('Vessel')
        ax2.set_ylabel('Speed Reduction (knots)')
        ax2.set_title('Evasive Maneuver - Speed Reductions')
        ax2.set_xticks(x)
        ax2.set_xticklabels(vessels, rotation=45, ha='right')
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5,
                   label='Significant threshold')
        ax2.legend()
        
        # Add value labels on bars
        for bar, val in zip(bars2, speed_drops):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val:.1f} kt', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'impact_metrics.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def save_impact_metrics_csv(self, impact_events: List[Dict]) -> str:
        """Save impact metrics to CSV for further analysis"""
        
        if not impact_events:
            return None
        
        # Create DataFrame
        metrics_data = []
        for event in impact_events:
            metrics_data.append({
                'mmsi': event['mmsi'],
                'vessel_name': event.get('vessel_name', ''),
                'cpa_nm': event.get('cpa_nm', 0),
                'cpa_time': event.get('cpa_time', ''),
                'maneuver_time': event.get('maneuver_time', ''),
                'delta_heading': event.get('delta_heading', 0),
                'speed_drop': event.get('speed_drop', 0),
                'initial_speed': event.get('initial_speed', 0),
                'action': event.get('action', '')
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'impact_metrics.csv')
        df.to_csv(output_path, index=False)
        
        return output_path