#!/usr/bin/env python3
"""
Navigation Telemetry Analysis Tool
Collects and analyzes telemetry data from navigation components.
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TelemetryReport:
    """Comprehensive telemetry analysis report"""
    timestamp: datetime
    navigation_stats: Dict[str, Any]
    autopilot_stats: Dict[str, Any]
    precision_analysis: Dict[str, Any]
    improvement_recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'navigation_stats': self.navigation_stats,
            'autopilot_stats': self.autopilot_stats,
            'precision_analysis': self.precision_analysis,
            'improvement_recommendations': self.improvement_recommendations
        }


class TelemetryAnalyzer:
    """
    Comprehensive telemetry data analyzer for navigation systems.
    """
    
    def __init__(self, output_dir: str = "output/telemetry"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis thresholds
        self.precision_thresholds = {
            'excellent': 0.1,
            'good': 0.5,
            'fair': 1.0,
            'poor': 2.0,
            'critical': float('inf')
        }
        
        self.target_precision = 0.031  # degrees
        
    def load_navigation_telemetry(self, file_path: str) -> Dict[str, Any]:
        """
        Load navigation telemetry data from JSON file.
        
        Args:
            file_path: Path to navigation telemetry JSON file
            
        Returns:
            Navigation telemetry data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded navigation telemetry from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load navigation telemetry: {e}")
            return {}
    
    def load_autopilot_telemetry(self, file_path: str) -> Dict[str, Any]:
        """
        Load autopilot telemetry data from JSON file.
        
        Args:
            file_path: Path to autopilot telemetry JSON file
            
        Returns:
            Autopilot telemetry data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded autopilot telemetry from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load autopilot telemetry: {e}")
            return {}
    
    def analyze_route_accuracy(self, nav_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze route accuracy from navigation telemetry.
        
        Args:
            nav_data: Navigation telemetry data
            
        Returns:
            Route accuracy analysis
        """
        if not nav_data or 'routes' not in nav_data:
            return {'error': 'No route data available'}
        
        routes = nav_data['routes']
        
        # Extract metrics
        distance_errors = [r['distance_error'] for r in routes]
        waypoint_densities = [r['waypoint_density'] for r in routes]
        planned_distances = [r['planned_distance'] for r in routes]
        
        # Calculate statistics
        analysis = {
            'total_routes_analyzed': len(routes),
            'distance_accuracy': {
                'avg_error_nm': np.mean(distance_errors),
                'max_error_nm': np.max(distance_errors),
                'std_error_nm': np.std(distance_errors),
                'median_error_nm': np.median(distance_errors),
                'error_percentiles': {
                    '95th': np.percentile(distance_errors, 95),
                    '90th': np.percentile(distance_errors, 90),
                    '75th': np.percentile(distance_errors, 75)
                }
            },
            'waypoint_efficiency': {
                'avg_density': np.mean(waypoint_densities),
                'max_density': np.max(waypoint_densities),
                'min_density': np.min(waypoint_densities),
                'std_density': np.std(waypoint_densities)
            },
            'route_distribution': {
                'avg_distance_nm': np.mean(planned_distances),
                'max_distance_nm': np.max(planned_distances),
                'min_distance_nm': np.min(planned_distances),
                'total_distance_nm': np.sum(planned_distances)
            }
        }
        
        # Grade accuracy
        avg_error = analysis['distance_accuracy']['avg_error_nm']
        analysis['accuracy_grade'] = self._grade_accuracy(avg_error)
        
        # Identify improvement opportunities
        analysis['improvement_opportunities'] = []
        
        if avg_error > 0.1:
            analysis['improvement_opportunities'].append(
                f"High distance error ({avg_error:.3f}nm) - consider denser waypoints"
            )
        
        if np.mean(waypoint_densities) < 0.5:
            analysis['improvement_opportunities'].append(
                "Low waypoint density - increase interpolation frequency"
            )
        
        return analysis
    
    def analyze_heading_precision(self, autopilot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze heading precision from autopilot telemetry.
        
        Args:
            autopilot_data: Autopilot telemetry data
            
        Returns:
            Heading precision analysis
        """
        if not autopilot_data or 'telemetry_points' not in autopilot_data:
            return {'error': 'No autopilot telemetry data available'}
        
        telemetry_points = autopilot_data['telemetry_points']
        
        # Extract metrics
        heading_errors = [t['heading_error'] for t in telemetry_points]
        corrections = [abs(t['correction_applied']) for t in telemetry_points]
        time_deltas = [t['dt'] for t in telemetry_points]
        
        # Calculate statistics
        analysis = {
            'total_corrections_analyzed': len(telemetry_points),
            'heading_precision': {
                'avg_error_degrees': np.mean(heading_errors),
                'max_error_degrees': np.max(heading_errors),
                'std_error_degrees': np.std(heading_errors),
                'median_error_degrees': np.median(heading_errors),
                'error_percentiles': {
                    '95th': np.percentile(heading_errors, 95),
                    '90th': np.percentile(heading_errors, 90),
                    '75th': np.percentile(heading_errors, 75)
                }
            },
            'correction_analysis': {
                'avg_correction_degrees': np.mean(corrections),
                'max_correction_degrees': np.max(corrections),
                'std_correction_degrees': np.std(corrections),
                'correction_rate': len([c for c in corrections if c > 0.1]) / len(corrections)
            },
            'temporal_analysis': {
                'avg_time_delta': np.mean(time_deltas),
                'max_time_delta': np.max(time_deltas),
                'min_time_delta': np.min(time_deltas),
                'total_time_analyzed': np.sum(time_deltas)
            }
        }
        
        # Grade precision
        avg_error = analysis['heading_precision']['avg_error_degrees']
        analysis['precision_grade'] = self._grade_precision(avg_error)
        
        # Calculate improvement factor needed
        analysis['improvement_factor_needed'] = avg_error / self.target_precision
        
        # Identify improvement opportunities
        analysis['improvement_opportunities'] = []
        
        if avg_error > 2.0:
            analysis['improvement_opportunities'].append(
                f"Critical heading error ({avg_error:.3f}°) - requires immediate PID tuning"
            )
        elif avg_error > 1.0:
            analysis['improvement_opportunities'].append(
                f"Poor heading precision ({avg_error:.3f}°) - optimize PID parameters"
            )
        elif avg_error > 0.5:
            analysis['improvement_opportunities'].append(
                f"Fair heading precision ({avg_error:.3f}°) - consider adaptive PID"
            )
        
        if np.mean(corrections) > 5.0:
            analysis['improvement_opportunities'].append(
                "High correction values - system may be oscillating"
            )
        
        return analysis
    
    def generate_precision_trends(self, autopilot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate precision trend analysis from autopilot telemetry.
        
        Args:
            autopilot_data: Autopilot telemetry data
            
        Returns:
            Precision trend analysis
        """
        if not autopilot_data or 'telemetry_points' not in autopilot_data:
            return {'error': 'No autopilot telemetry data available'}
        
        telemetry_points = autopilot_data['telemetry_points']
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(telemetry_points)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate rolling statistics
        window_size = min(50, len(df) // 10)  # Adaptive window size
        if window_size > 0:
            df['rolling_avg_error'] = df['heading_error'].rolling(window=window_size).mean()
            df['rolling_std_error'] = df['heading_error'].rolling(window=window_size).std()
        
        # Identify trends
        trends = {
            'data_points': len(df),
            'time_span_minutes': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60,
            'trend_analysis': {}
        }
        
        if window_size > 0:
            # Calculate trend slope
            valid_data = df.dropna()
            if len(valid_data) > 10:
                x = np.arange(len(valid_data))
                y = valid_data['rolling_avg_error'].values
                
                # Linear regression
                slope, intercept = np.polyfit(x, y, 1)
                trends['trend_analysis'] = {
                    'slope_degrees_per_point': slope,
                    'trend_direction': 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable',
                    'r_squared': np.corrcoef(x, y)[0, 1] ** 2
                }
        
        # Identify periods of instability
        if len(df) > 0:
            high_error_periods = df[df['heading_error'] > 2.0]
            trends['instability_analysis'] = {
                'high_error_periods': len(high_error_periods),
                'high_error_percentage': len(high_error_periods) / len(df) * 100,
                'max_consecutive_errors': self._find_max_consecutive_errors(df['heading_error'].values)
            }
        
        return trends
    
    def _find_max_consecutive_errors(self, errors: np.ndarray, threshold: float = 1.0) -> int:
        """Find maximum consecutive errors above threshold."""
        consecutive = 0
        max_consecutive = 0
        
        for error in errors:
            if error > threshold:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _grade_accuracy(self, error: float) -> str:
        """Grade accuracy based on error value."""
        if error <= 0.01:
            return 'Excellent'
        elif error <= 0.05:
            return 'Good'
        elif error <= 0.1:
            return 'Fair'
        elif error <= 0.5:
            return 'Poor'
        else:
            return 'Critical'
    
    def _grade_precision(self, error: float) -> str:
        """Grade precision based on error value."""
        for grade, threshold in self.precision_thresholds.items():
            if error <= threshold:
                return grade.capitalize()
        return 'Critical'
    
    def generate_improvement_recommendations(self, nav_analysis: Dict[str, Any], 
                                          autopilot_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate improvement recommendations based on analysis.
        
        Args:
            nav_analysis: Navigation accuracy analysis
            autopilot_analysis: Autopilot precision analysis
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # Navigation improvements
        if 'distance_accuracy' in nav_analysis:
            avg_error = nav_analysis['distance_accuracy']['avg_error_nm']
            if avg_error > 0.1:
                recommendations.append(
                    f"Reduce waypoint interval to improve route accuracy "
                    f"(current error: {avg_error:.3f}nm)"
                )
        
        # Autopilot improvements
        if 'heading_precision' in autopilot_analysis:
            avg_error = autopilot_analysis['heading_precision']['avg_error_degrees']
            improvement_factor = autopilot_analysis.get('improvement_factor_needed', 1)
            
            if improvement_factor > 50:
                recommendations.append(
                    f"Implement adaptive PID controller - need {improvement_factor:.1f}x improvement"
                )
                recommendations.append(
                    "Consider waypoint density optimization for better heading guidance"
                )
                recommendations.append(
                    "Implement predictive control for smoother trajectory following"
                )
            elif improvement_factor > 10:
                recommendations.append(
                    f"Optimize PID parameters - need {improvement_factor:.1f}x improvement"
                )
                recommendations.append(
                    "Reduce turn rate constraints for more responsive control"
                )
            elif improvement_factor > 2:
                recommendations.append(
                    f"Fine-tune PID gains - need {improvement_factor:.1f}x improvement"
                )
        
        # System-level improvements
        if len(recommendations) > 0:
            recommendations.append("Implement comprehensive telemetry monitoring")
            recommendations.append("Add real-time precision feedback system")
        
        return recommendations
    
    def generate_comprehensive_report(self, nav_file: Optional[str] = None,
                                    autopilot_file: Optional[str] = None) -> TelemetryReport:
        """
        Generate comprehensive telemetry analysis report.
        
        Args:
            nav_file: Optional path to navigation telemetry file
            autopilot_file: Optional path to autopilot telemetry file
            
        Returns:
            Comprehensive telemetry report
        """
        logger.info("Generating comprehensive telemetry report...")
        
        # Load data
        nav_data = self.load_navigation_telemetry(nav_file) if nav_file else {}
        autopilot_data = self.load_autopilot_telemetry(autopilot_file) if autopilot_file else {}
        
        # Analyze data
        nav_analysis = self.analyze_route_accuracy(nav_data)
        autopilot_analysis = self.analyze_heading_precision(autopilot_data)
        
        # Generate precision analysis
        precision_analysis = {}
        if autopilot_data:
            precision_analysis = self.generate_precision_trends(autopilot_data)
        
        # Generate recommendations
        recommendations = self.generate_improvement_recommendations(nav_analysis, autopilot_analysis)
        
        # Create comprehensive report
        report = TelemetryReport(
            timestamp=datetime.now(),
            navigation_stats=nav_analysis,
            autopilot_stats=autopilot_analysis,
            precision_analysis=precision_analysis,
            improvement_recommendations=recommendations
        )
        
        # Save report
        report_file = self.output_dir / f"telemetry_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Comprehensive telemetry report saved to {report_file}")
        
        return report
    
    def plot_precision_trends(self, autopilot_file: str) -> str:
        """
        Generate precision trend plots.
        
        Args:
            autopilot_file: Path to autopilot telemetry file
            
        Returns:
            Path to generated plot file
        """
        try:
            # Load data
            autopilot_data = self.load_autopilot_telemetry(autopilot_file)
            
            if not autopilot_data or 'telemetry_points' not in autopilot_data:
                logger.error("No autopilot telemetry data available for plotting")
                return ""
            
            telemetry_points = autopilot_data['telemetry_points']
            df = pd.DataFrame(telemetry_points)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Navigation System Precision Analysis', fontsize=16)
            
            # Plot 1: Heading Error Over Time
            axes[0, 0].plot(df['timestamp'], df['heading_error'], alpha=0.7, linewidth=1)
            axes[0, 0].set_title('Heading Error Over Time')
            axes[0, 0].set_ylabel('Error (degrees)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=self.target_precision, color='red', linestyle='--', 
                              label=f'Target: {self.target_precision:.3f}°')
            axes[0, 0].legend()
            
            # Plot 2: Correction Values Over Time
            axes[0, 1].plot(df['timestamp'], df['correction_applied'], alpha=0.7, linewidth=1, color='orange')
            axes[0, 1].set_title('Correction Applied Over Time')
            axes[0, 1].set_ylabel('Correction (degrees)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Error Distribution
            axes[1, 0].hist(df['heading_error'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title('Heading Error Distribution')
            axes[1, 0].set_xlabel('Error (degrees)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(x=np.mean(df['heading_error']), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(df["heading_error"]):.3f}°')
            axes[1, 0].legend()
            
            # Plot 4: Rolling Average
            window_size = min(50, len(df) // 10)
            if window_size > 0:
                df['rolling_avg'] = df['heading_error'].rolling(window=window_size).mean()
                axes[1, 1].plot(df['timestamp'], df['rolling_avg'], linewidth=2, color='purple')
                axes[1, 1].set_title(f'Rolling Average Error (window={window_size})')
                axes[1, 1].set_ylabel('Error (degrees)')
                axes[1, 1].grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in axes.flat:
                if ax.get_xlabel() == '' and len(ax.get_xticks()) > 0:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"precision_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Precision trends plot saved to {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Failed to generate precision trends plot: {e}")
            return ""


def main():
    """CLI entry point for telemetry analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Navigation Telemetry Analyzer")
    parser.add_argument("--nav-file", help="Path to navigation telemetry JSON file")
    parser.add_argument("--autopilot-file", help="Path to autopilot telemetry JSON file")
    parser.add_argument("--output-dir", default="output/telemetry",
                       help="Output directory for analysis results")
    parser.add_argument("--plot", action="store_true",
                       help="Generate precision trend plots")
    
    args = parser.parse_args()
    
    # Create analyzer instance
    analyzer = TelemetryAnalyzer(output_dir=args.output_dir)
    
    try:
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report(
            nav_file=args.nav_file,
            autopilot_file=args.autopilot_file
        )
        
        # Generate plots if requested
        if args.plot and args.autopilot_file:
            plot_file = analyzer.plot_precision_trends(args.autopilot_file)
            if plot_file:
                print(f"Precision trends plot generated: {plot_file}")
        
        # Print summary
        print("\n=== TELEMETRY ANALYSIS SUMMARY ===")
        
        if report.navigation_stats and 'distance_accuracy' in report.navigation_stats:
            nav_stats = report.navigation_stats['distance_accuracy']
            print(f"Navigation accuracy: {nav_stats['avg_error_nm']:.3f}nm average error")
        
        if report.autopilot_stats and 'heading_precision' in report.autopilot_stats:
            autopilot_stats = report.autopilot_stats['heading_precision']
            print(f"Autopilot precision: {autopilot_stats['avg_error_degrees']:.3f}° average error")
            
            if 'improvement_factor_needed' in report.autopilot_stats:
                factor = report.autopilot_stats['improvement_factor_needed']
                print(f"Improvement factor needed: {factor:.1f}x")
        
        print(f"\nRecommendations ({len(report.improvement_recommendations)}):")
        for i, rec in enumerate(report.improvement_recommendations, 1):
            print(f"  {i}. {rec}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())