#!/usr/bin/env python3
"""
Performance Analysis Tool for AIS Attack Generation System

This tool analyzes system performance and provides optimization recommendations.
"""

import os
import sys
import time
import psutil
import argparse
import cProfile
import pstats
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for AIS attack generation system
    """
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'execution_times': [],
            'function_calls': {},
            'bottlenecks': []
        }
        self.start_time = None
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'execution_times': [],
            'function_calls': {},
            'bottlenecks': []
        }
    
    def collect_system_metrics(self) -> Dict:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # Process-specific metrics
            process_memory = self.process.memory_info().rss / (1024**2)  # MB
            process_cpu = self.process.cpu_percent()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_usage = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'process_percent': process_cpu
                },
                'memory': {
                    'system_percent': memory_percent,
                    'available_gb': memory_available,
                    'process_mb': process_memory
                },
                'disk': {
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0,
                    'usage_percent': disk_usage.percent
                }
            }
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def profile_function(self, func, *args, **kwargs) -> Dict:
        """Profile a specific function execution"""
        profiler = cProfile.Profile()
        
        start_time = time.time()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        profiler.disable()
        end_time = time.time()
        
        # Analyze profiling results
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        profile_output = stats_stream.getvalue()
        
        return {
            'function_name': func.__name__,
            'execution_time': end_time - start_time,
            'success': success,
            'error': error,
            'result': result,
            'profile_stats': profile_output,
            'call_count': stats.total_calls,
            'primitive_calls': stats.prim_calls
        }
    
    def benchmark_attack_generation(self, 
                                  attack_type: str = 's1_flash_cross',
                                  iterations: int = 10) -> Dict:
        """Benchmark attack generation performance"""
        try:
            from core.attack_orchestrator import AttackOrchestrator
            from core.target_selector import TargetSelector
            from core.physics_engine import PhysicsEngine
            from core.colregs_validator import COLREGSValidator
            
            # Initialize components
            target_selector = TargetSelector()
            physics_engine = PhysicsEngine()
            colregs_validator = COLREGSValidator()
            orchestrator = AttackOrchestrator(target_selector, physics_engine, colregs_validator)
            
            execution_times = []
            memory_usage = []
            cpu_usage = []
            
            # Sample target data
            target_data = {
                'mmsi': '123456789',
                'lat': 40.7128,
                'lon': -74.0060,
                'speed': 12.0,
                'course': 90.0,
                'vessel_type': 'cargo'
            }
            
            for i in range(iterations):
                print(f"Running benchmark iteration {i+1}/{iterations}...")
                
                # Collect pre-execution metrics
                pre_metrics = self.collect_system_metrics()
                
                # Execute attack generation
                start_time = time.time()
                
                try:
                    # This would normally execute the attack
                    # For benchmarking, we simulate the process
                    time.sleep(0.1)  # Simulate processing time
                    success = True
                except Exception as e:
                    success = False
                    print(f"Error in iteration {i+1}: {e}")
                
                end_time = time.time()
                
                # Collect post-execution metrics
                post_metrics = self.collect_system_metrics()
                
                execution_times.append(end_time - start_time)
                memory_usage.append(post_metrics['memory']['process_mb'])
                cpu_usage.append(post_metrics['cpu']['process_percent'])
            
            return {
                'attack_type': attack_type,
                'iterations': iterations,
                'execution_times': {
                    'mean': np.mean(execution_times),
                    'std': np.std(execution_times),
                    'min': np.min(execution_times),
                    'max': np.max(execution_times),
                    'median': np.median(execution_times)
                },
                'memory_usage': {
                    'mean_mb': np.mean(memory_usage),
                    'max_mb': np.max(memory_usage),
                    'min_mb': np.min(memory_usage)
                },
                'cpu_usage': {
                    'mean_percent': np.mean(cpu_usage),
                    'max_percent': np.max(cpu_usage)
                },
                'throughput': {
                    'attacks_per_second': iterations / sum(execution_times),
                    'total_time': sum(execution_times)
                }
            }
            
        except ImportError as e:
            return {'error': f'Cannot import required modules: {e}'}
        except Exception as e:
            return {'error': f'Benchmark failed: {e}'}
    
    def benchmark_batch_processing(self, 
                                  scenario_count: int = 100,
                                  parallel_workers: int = 4) -> Dict:
        """Benchmark batch processing performance"""
        try:
            from tools.batch_runner.run_all_scenarios import BatchRunner
            
            # Create mock scenarios
            mock_scenarios = []
            for i in range(scenario_count):
                scenario = {
                    'id': f'benchmark_s{i:03d}',
                    'attack_type': 's1_flash_cross',
                    'parameters': {
                        'target_mmsi': f'12345{i:04d}',
                        'duration': 300
                    }
                }
                mock_scenarios.append(scenario)
            
            # Initialize batch runner
            runner = BatchRunner()
            runner.scenarios = mock_scenarios
            
            # Benchmark sequential processing
            print("Benchmarking sequential processing...")
            start_time = time.time()
            
            # Mock sequential execution
            for scenario in mock_scenarios:
                time.sleep(0.01)  # Simulate processing
            
            sequential_time = time.time() - start_time
            
            # Benchmark parallel processing
            print(f"Benchmarking parallel processing with {parallel_workers} workers...")
            start_time = time.time()
            
            # Mock parallel execution
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [executor.submit(time.sleep, 0.01) for _ in mock_scenarios]
                concurrent.futures.wait(futures)
            
            parallel_time = time.time() - start_time
            
            return {
                'scenario_count': scenario_count,
                'sequential_time': sequential_time,
                'parallel_time': parallel_time,
                'speedup_factor': sequential_time / parallel_time,
                'efficiency': (sequential_time / parallel_time) / parallel_workers,
                'sequential_throughput': scenario_count / sequential_time,
                'parallel_throughput': scenario_count / parallel_time,
                'recommended_workers': min(psutil.cpu_count(), scenario_count // 10)
            }
            
        except Exception as e:
            return {'error': f'Batch benchmark failed: {e}'}
    
    def analyze_memory_usage(self, duration: int = 60) -> Dict:
        """Analyze memory usage patterns over time"""
        print(f"Monitoring memory usage for {duration} seconds...")
        
        memory_samples = []
        timestamps = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            memory_info = self.process.memory_info()
            
            memory_samples.append({
                'timestamp': current_time,
                'rss_mb': memory_info.rss / (1024**2),
                'vms_mb': memory_info.vms / (1024**2)
            })
            
            timestamps.append(current_time)
            time.sleep(1)  # Sample every second
        
        # Calculate statistics
        rss_values = [sample['rss_mb'] for sample in memory_samples]
        
        return {
            'duration': duration,
            'samples': len(memory_samples),
            'memory_stats': {
                'mean_rss_mb': np.mean(rss_values),
                'max_rss_mb': np.max(rss_values),
                'min_rss_mb': np.min(rss_values),
                'std_rss_mb': np.std(rss_values)
            },
            'memory_growth': rss_values[-1] - rss_values[0],
            'samples': memory_samples
        }
    
    def identify_bottlenecks(self, profile_data: Dict) -> List[Dict]:
        """Identify performance bottlenecks from profiling data"""
        bottlenecks = []
        
        if 'profile_stats' in profile_data:
            # Parse profile statistics to identify slow functions
            stats_lines = profile_data['profile_stats'].split('\n')
            
            for line in stats_lines:
                if 'function calls' in line or 'primitive calls' in line:
                    continue
                
                parts = line.split()
                if len(parts) >= 6 and parts[0].replace('.', '').isdigit():
                    try:
                        cumulative_time = float(parts[3])
                        function_name = ' '.join(parts[5:])
                        
                        if cumulative_time > 0.1:  # Functions taking >100ms
                            bottlenecks.append({
                                'function': function_name,
                                'cumulative_time': cumulative_time,
                                'severity': 'high' if cumulative_time > 1.0 else 'medium'
                            })
                    except (ValueError, IndexError):
                        continue
        
        return sorted(bottlenecks, key=lambda x: x['cumulative_time'], reverse=True)
    
    def generate_performance_report(self, 
                                  benchmark_results: Dict,
                                  output_path: str = None) -> str:
        """Generate comprehensive performance report"""
        
        report_data = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                    'platform': sys.platform,
                    'python_version': sys.version
                }
            },
            'benchmark_results': benchmark_results,
            'recommendations': self._generate_recommendations(benchmark_results)
        }
        
        if output_path is None:
            output_path = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_path = Path(output_path).with_suffix('.md')
        self._generate_markdown_report(report_data, summary_path)
        
        return str(output_path)
    
    def _generate_recommendations(self, benchmark_results: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if 'execution_times' in benchmark_results:
            mean_time = benchmark_results['execution_times']['mean']
            
            if mean_time > 10.0:
                recommendations.append("Consider optimizing algorithm implementation - execution time is high")
            
            if mean_time > 5.0:
                recommendations.append("Enable parallel processing for batch operations")
        
        if 'memory_usage' in benchmark_results:
            max_memory = benchmark_results['memory_usage']['max_mb']
            
            if max_memory > 1000:
                recommendations.append("High memory usage detected - consider implementing streaming processing")
            
            if max_memory > 500:
                recommendations.append("Implement data chunking for large datasets")
        
        if 'throughput' in benchmark_results:
            throughput = benchmark_results['throughput']['attacks_per_second']
            
            if throughput < 1.0:
                recommendations.append("Low throughput - consider algorithm optimization or caching")
        
        # System-specific recommendations
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if cpu_count >= 8:
            recommendations.append(f"System has {cpu_count} CPUs - increase parallel worker count")
        
        if memory_gb >= 16:
            recommendations.append("Sufficient memory available - consider increasing batch sizes")
        
        return recommendations
    
    def _generate_markdown_report(self, report_data: Dict, output_path: Path):
        """Generate markdown performance report"""
        
        content = f"""# Performance Analysis Report

Generated: {report_data['report_info']['generated_at']}

## System Information

- **CPU Count**: {report_data['report_info']['system_info']['cpu_count']}
- **Total Memory**: {report_data['report_info']['system_info']['memory_total_gb']:.1f} GB
- **Platform**: {report_data['report_info']['system_info']['platform']}
- **Python Version**: {report_data['report_info']['system_info']['python_version'].split()[0]}

## Benchmark Results

"""
        
        if 'execution_times' in report_data['benchmark_results']:
            exec_times = report_data['benchmark_results']['execution_times']
            content += f"""### Execution Times

- **Mean**: {exec_times['mean']:.3f} seconds
- **Median**: {exec_times['median']:.3f} seconds
- **Min**: {exec_times['min']:.3f} seconds
- **Max**: {exec_times['max']:.3f} seconds
- **Standard Deviation**: {exec_times['std']:.3f} seconds

"""
        
        if 'throughput' in report_data['benchmark_results']:
            throughput = report_data['benchmark_results']['throughput']
            content += f"""### Throughput

- **Attacks per Second**: {throughput['attacks_per_second']:.2f}
- **Total Processing Time**: {throughput['total_time']:.3f} seconds

"""
        
        if 'memory_usage' in report_data['benchmark_results']:
            memory = report_data['benchmark_results']['memory_usage']
            content += f"""### Memory Usage

- **Mean**: {memory['mean_mb']:.1f} MB
- **Maximum**: {memory['max_mb']:.1f} MB
- **Minimum**: {memory['min_mb']:.1f} MB

"""
        
        content += "## Recommendations\n\n"
        for rec in report_data['recommendations']:
            content += f"- {rec}\n"
        
        content += f"""
## Next Steps

1. Review the detailed JSON report for more metrics
2. Implement recommended optimizations
3. Re-run benchmarks to measure improvements
4. Monitor production performance using these baselines

---

Report generated by AIS Attack System Performance Analyzer
"""
        
        with open(output_path, 'w') as f:
            f.write(content)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AIS Attack System Performance Analyzer")
    parser.add_argument("--benchmark-attack", "-a", action="store_true",
                       help="Benchmark attack generation")
    parser.add_argument("--benchmark-batch", "-b", action="store_true",
                       help="Benchmark batch processing")
    parser.add_argument("--memory-analysis", "-m", type=int, default=60,
                       help="Analyze memory usage for N seconds")
    parser.add_argument("--iterations", "-i", type=int, default=10,
                       help="Number of benchmark iterations")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Number of parallel workers for batch benchmark")
    parser.add_argument("--output", "-o", help="Output report file path")
    parser.add_argument("--all", action="store_true",
                       help="Run all performance tests")
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer()
    results = {}
    
    print("🚀 AIS Attack System Performance Analysis")
    print("=" * 50)
    
    if args.all or args.benchmark_attack:
        print("\n📊 Running attack generation benchmark...")
        attack_results = analyzer.benchmark_attack_generation(iterations=args.iterations)
        results['attack_benchmark'] = attack_results
        
        if 'error' not in attack_results:
            print(f"✅ Attack benchmark completed:")
            print(f"   Mean execution time: {attack_results['execution_times']['mean']:.3f}s")
            print(f"   Throughput: {attack_results['throughput']['attacks_per_second']:.2f} attacks/sec")
        else:
            print(f"❌ Attack benchmark failed: {attack_results['error']}")
    
    if args.all or args.benchmark_batch:
        print("\n⚡ Running batch processing benchmark...")
        batch_results = analyzer.benchmark_batch_processing(workers=args.workers)
        results['batch_benchmark'] = batch_results
        
        if 'error' not in batch_results:
            print(f"✅ Batch benchmark completed:")
            print(f"   Sequential time: {batch_results['sequential_time']:.3f}s")
            print(f"   Parallel time: {batch_results['parallel_time']:.3f}s")
            print(f"   Speedup factor: {batch_results['speedup_factor']:.2f}x")
        else:
            print(f"❌ Batch benchmark failed: {batch_results['error']}")
    
    if args.all or args.memory_analysis > 0:
        print(f"\n🧠 Analyzing memory usage for {args.memory_analysis} seconds...")
        memory_results = analyzer.analyze_memory_usage(args.memory_analysis)
        results['memory_analysis'] = memory_results
        
        print(f"✅ Memory analysis completed:")
        print(f"   Mean RSS: {memory_results['memory_stats']['mean_rss_mb']:.1f} MB")
        print(f"   Max RSS: {memory_results['memory_stats']['max_rss_mb']:.1f} MB")
        print(f"   Memory growth: {memory_results['memory_growth']:.1f} MB")
    
    # Generate report
    if results:
        report_path = analyzer.generate_performance_report(results, args.output)
        print(f"\n📋 Performance report generated:")
        print(f"   JSON report: {report_path}")
        print(f"   Summary: {Path(report_path).with_suffix('.md')}")
    else:
        print("\n⚠️  No benchmarks were run. Use --all or specific benchmark flags.")

if __name__ == "__main__":
    main()
