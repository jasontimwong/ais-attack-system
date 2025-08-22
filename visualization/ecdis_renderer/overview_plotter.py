"""
Overview visualization module for AIS attack scenarios.
Generates side-by-side baseline vs attack+impact plots.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def plot_overview(baseline_df: pd.DataFrame, attack_df: pd.DataFrame, 
                 impact_df: pd.DataFrame, diff_data: Dict, 
                 impact_events: Dict, save_path: str):
    """
    Generate overview visualization showing baseline vs attack+impact.
    
    Args:
        baseline_df: Original vessel data
        attack_df: Data with ghost vessel injected
        impact_df: Data with vessel responses
        diff_data: Diff statistics
        impact_events: Impact event details
        save_path: Path to save the plot
    """
    logger.info("Generating overview visualization...")
    
    # Get key MMSIs
    target_mmsi = diff_data['target_mmsi']
    ghost_mmsi = diff_data['ghost_mmsi']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Determine plot bounds
    all_lats = pd.concat([baseline_df['LAT'], impact_df['LAT']])
    all_lons = pd.concat([baseline_df['LON'], impact_df['LON']])
    
    lat_center = all_lats.mean()
    lon_center = all_lons.mean()
    lat_range = 0.15  # degrees
    lon_range = 0.15
    
    # Plot 1: Baseline (20 min)
    plot_baseline(ax1, baseline_df, target_mmsi, lat_center, lon_center, 
                 lat_range, lon_range)
    
    # Plot 2: Attack + Impact
    plot_attack_impact(ax2, impact_df, target_mmsi, ghost_mmsi, 
                      impact_events, lat_center, lon_center, 
                      lat_range, lon_range)
    
    # Add title with CPA info
    scenario_name = diff_data.get('scenario', 'S1 Attack')
    cpa_info = ""
    if impact_events.get('events'):
        cpa_nm = impact_events['events'][0].get('cpa_nm', 0)
        cpa_info = f" (CPA: {cpa_nm:.3f} nm)"
    plt.suptitle(f'{scenario_name} - Overview{cpa_info}', fontsize=16, fontweight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {save_path}")


def plot_baseline(ax, df, target_mmsi, lat_center, lon_center, 
                 lat_range, lon_range):
    """Plot baseline vessel tracks"""
    ax.set_title("Baseline (20 min)", fontsize=14, fontweight='bold')
    
    # Plot all vessels
    vessel_count = 0
    for mmsi in df['MMSI'].unique():
        vessel_data = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')
        
        if mmsi == target_mmsi:
            # Highlight target vessel
            ax.plot(vessel_data['LON'], vessel_data['LAT'], 'b-', 
                   linewidth=2, label='Target', alpha=0.8, zorder=10)
            # Mark start position
            ax.plot(vessel_data['LON'].iloc[0], vessel_data['LAT'].iloc[0], 
                   'bo', markersize=8, zorder=11)
            # Mark end position
            ax.plot(vessel_data['LON'].iloc[-1], vessel_data['LAT'].iloc[-1], 
                   'bs', markersize=8, zorder=11)
        else:
            # Other vessels in gray
            ax.plot(vessel_data['LON'], vessel_data['LAT'], 'gray', 
                   linewidth=0.5, alpha=0.3)
            vessel_count += 1
    
    # Set limits and labels
    ax.set_xlim(lon_center - lon_range, lon_center + lon_range)
    ax.set_ylim(lat_center - lat_range, lat_center + lat_range)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add vessel count
    ax.text(0.02, 0.98, f'{vessel_count + 1} vessels', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_attack_impact(ax, df, target_mmsi, ghost_mmsi, impact_events,
                      lat_center, lon_center, lat_range, lon_range):
    """Plot attack and impact scenario"""
    ax.set_title("Attack + Impact", fontsize=14, fontweight='bold')
    
    # Get impact event details
    events = impact_events.get('events', [])
    primary_event = next((e for e in events if e['mmsi'] == target_mmsi), None)
    cascade_events = [e for e in events if e['mmsi'] != target_mmsi]
    
    # Plot background vessels
    vessel_count = 0
    for mmsi in df['MMSI'].unique():
        if mmsi in [target_mmsi, ghost_mmsi]:
            continue
        
        vessel_data = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')
        
        # Check if this vessel had cascade reaction
        cascade_event = next((e for e in cascade_events if e['mmsi'] == mmsi), None)
        
        if cascade_event:
            # Plot cascade vessel with reaction
            reaction_time = pd.to_datetime(cascade_event['reaction_time'])
            before = vessel_data[vessel_data['BaseDateTime'] < reaction_time]
            after = vessel_data[vessel_data['BaseDateTime'] >= reaction_time]
            
            ax.plot(before['LON'], before['LAT'], 'orange', 
                   linewidth=1, alpha=0.6)
            ax.plot(after['LON'], after['LAT'], 'darkorange', 
                   linewidth=1.5, alpha=0.8, 
                   label=f'Cascade L{cascade_event.get("cascade_level", 1)}')
            
            # Mark reaction point
            if len(after) > 0:
                ax.plot(after['LON'].iloc[0], after['LAT'].iloc[0], 
                       'o', color='darkorange', markersize=6)
        else:
            # Normal vessel
            ax.plot(vessel_data['LON'], vessel_data['LAT'], 'gray', 
                   linewidth=0.5, alpha=0.3)
        
        vessel_count += 1
    
    # Plot target vessel with maneuver
    target_data = df[df['MMSI'] == target_mmsi].sort_values('BaseDateTime')
    if primary_event and len(target_data) > 0:
        maneuver_time = pd.to_datetime(primary_event['maneuver_time'])
        
        before = target_data[target_data['BaseDateTime'] < maneuver_time]
        after = target_data[target_data['BaseDateTime'] >= maneuver_time]
        
        ax.plot(before['LON'], before['LAT'], 'b-', 
               linewidth=2, alpha=0.8, zorder=10)
        ax.plot(after['LON'], after['LAT'], 'r-', 
               linewidth=2, label='Target (maneuver)', alpha=0.8, zorder=10)
        
        # Mark maneuver point
        if len(after) > 0:
            ax.plot(after['LON'].iloc[0], after['LAT'].iloc[0], 
                   'r*', markersize=15, label='Maneuver', zorder=12)
    
    # Plot ghost vessel
    ghost_data = df[df['MMSI'] == ghost_mmsi].sort_values('BaseDateTime')
    if len(ghost_data) > 0:
        ax.plot(ghost_data['LON'], ghost_data['LAT'], 'r--', 
               linewidth=2, label='Ghost vessel', alpha=0.8, zorder=9)
        
        # Mark start
        ax.plot(ghost_data['LON'].iloc[0], ghost_data['LAT'].iloc[0], 
               'ro', markersize=8, zorder=11)
        
        # Mark silence zone (approximate)
        if len(ghost_data) > 20:
            silence_start_idx = max(0, len(ghost_data) - 30)
            silence_data = ghost_data.iloc[silence_start_idx:]
            
            # Create polygon for silence zone
            lons = silence_data['LON'].values
            lats = silence_data['LAT'].values
            
            # Expand slightly for visibility
            lon_center = lons.mean()
            lat_center = lats.mean()
            scale = 1.5
            
            expanded_lons = lon_center + (lons - lon_center) * scale
            expanded_lats = lat_center + (lats - lat_center) * scale
            
            # Create convex hull
            points = np.column_stack((expanded_lons, expanded_lats))
            if len(points) > 2:
                from matplotlib.patches import Polygon
                poly = Polygon(points, alpha=0.2, color='red', 
                             label='Silence zone')
                ax.add_patch(poly)
    
    # Mark CPA if available
    if primary_event and 'cpa_position' in primary_event:
        cpa_pos = primary_event.get('cpa_position', {})
        if 'ghost' in cpa_pos and cpa_pos['ghost']:
            ax.plot(cpa_pos['ghost']['lon'], cpa_pos['ghost']['lat'], 
                   'kx', markersize=12, markeredgewidth=3, 
                   label=f"CPA ({primary_event['cpa_nm']:.3f} nm)", zorder=13)
    
    # Set limits and labels
    ax.set_xlim(lon_center - lon_range, lon_center + lon_range)
    ax.set_ylim(lat_center - lat_range, lat_center + lat_range)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Add statistics
    stats_text = f'{vessel_count + 2} vessels\n'
    stats_text += f'{impact_events["impacted_vessels"]} impacted\n'
    stats_text += f'{impact_events.get("cascade_levels", 0)} cascade levels'
    
    ax.text(0.02, 0.98, stats_text, 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# Standalone test
if __name__ == "__main__":
    # Test with dummy data
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        
        # Load data
        baseline_df = pd.read_csv(f"{output_dir}/baseline.csv", parse_dates=['BaseDateTime'])
        attack_df = pd.read_csv(f"{output_dir}/attack.csv", parse_dates=['BaseDateTime'])
        impact_df = pd.read_csv(f"{output_dir}/impact.csv", parse_dates=['BaseDateTime'])
        
        import json
        with open(f"{output_dir}/diff.json", 'r') as f:
            diff_data = json.load(f)
        with open(f"{output_dir}/impact_events.json", 'r') as f:
            impact_events = json.load(f)
        
        # Generate plot
        plot_overview(baseline_df, attack_df, impact_df, diff_data, 
                     impact_events, f"{output_dir}/viz/overview_test.png")
        
        print(f"Test plot saved to {output_dir}/viz/overview_test.png")