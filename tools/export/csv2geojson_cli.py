#!/usr/bin/env python3
"""Command-line tool for converting AIS CSV data to GeoJSON format."""

import argparse
import logging
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import multiprocessing as mp
from multiprocessing import Pool, get_context
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_converter import convert_csv_to_geojson, batch_convert_directory
from core.zone_extractor import ZoneExtractor, extract_zones_from_scenario
from core.trajectory_sampler import SamplingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Convert AIS CSV data to GeoJSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  %(prog)s -i baseline.csv -o baseline.geojson -s S2

  # Convert baseline and attack files
  %(prog)s --baseline baseline.csv --attack attack.csv --out-dir output/

  # Batch convert directory
  %(prog)s --batch --input-dir data/ --out-dir geojson/ --workers 4
  
  # Extract zones from config
  %(prog)s --zones-only --zone-config scenario.yaml --out-dir output/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-i', '--input', 
                            help='Input CSV file')
    input_group.add_argument('--baseline',
                            help='Baseline CSV file')
    input_group.add_argument('--batch', action='store_true',
                            help='Batch process directory')
    
    parser.add_argument('--attack',
                       help='Attack CSV file (use with --baseline)')
    parser.add_argument('--input-dir',
                       help='Input directory for batch mode')
    
    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('-o', '--output',
                             help='Output GeoJSON file')
    output_group.add_argument('--out-dir',
                             help='Output directory')
    
    # Processing options
    parser.add_argument('-s', '--scenario', default='S1',
                       choices=['S1', 'S2'],
                       help='Scenario type for sampling (default: S1)')
    parser.add_argument('-w', '--workers', type=int,
                       default=mp.cpu_count(),
                       help=f'Number of worker processes (default: {mp.cpu_count()})')
    parser.add_argument('--sample-rate', type=float,
                       help='Override default sampling rate (0.0-1.0)')
    parser.add_argument('--vessel-type', default='baseline',
                       choices=['baseline', 'attack'],
                       help='Vessel type for single file conversion')
    parser.add_argument('--attack-type',
                       choices=['ghost', 'spoof', 'zone_violation'],
                       help='Attack type if vessel-type is attack')
    
    # Zone extraction
    parser.add_argument('--zone-config',
                       help='Configuration file containing guard zones')
    parser.add_argument('--zones-only', action='store_true',
                       help='Only extract zones, skip trajectory conversion')
    
    # Other options
    parser.add_argument('-p', '--progress', action='store_true',
                       help='Show progress bar')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    
    return parser


def convert_single_file(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert a single CSV file to GeoJSON."""
    input_file = args.input or args.baseline
    
    if args.output:
        output_file = args.output
    else:
        # Generate output filename
        input_path = Path(input_file)
        output_dir = Path(args.out_dir) if args.out_dir else input_path.parent
        output_file = output_dir / (input_path.stem + '.geojson')
    
    # Create custom sampling config if needed
    sample_config = None
    if args.sample_rate is not None:
        sample_config = SamplingConfig(sample_rate=args.sample_rate)
    
    # Convert file
    stats = convert_csv_to_geojson(
        input_path=input_file,
        output_path=str(output_file),
        vessel_type=args.vessel_type,
        attack_type=args.attack_type,
        scenario=args.scenario,
        sample_config=sample_config
    )
    
    return stats


def convert_baseline_attack(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert baseline and attack files."""
    output_dir = Path(args.out_dir) if args.out_dir else Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Convert baseline
    baseline_output = output_dir / 'baseline.geojson'
    logger.info(f"Converting baseline: {args.baseline}")
    results['baseline'] = convert_csv_to_geojson(
        input_path=args.baseline,
        output_path=str(baseline_output),
        vessel_type='baseline',
        scenario=args.scenario
    )
    
    # Convert attack if provided
    if args.attack:
        attack_output = output_dir / 'attack.geojson'
        logger.info(f"Converting attack: {args.attack}")
        
        # Try to determine attack type from filename
        attack_type = args.attack_type
        if not attack_type:
            if 'ghost' in args.attack.lower():
                attack_type = 'ghost'
            elif 'spoof' in args.attack.lower():
                attack_type = 'spoof'
            elif 'zone' in args.attack.lower():
                attack_type = 'zone_violation'
        
        results['attack'] = convert_csv_to_geojson(
            input_path=args.attack,
            output_path=str(attack_output),
            vessel_type='attack',
            attack_type=attack_type,
            scenario=args.scenario
        )
    
    return results


def process_vessel_group(params: Dict[str, Any]) -> Dict[str, Any]:
    """Process a group of vessels (for multiprocessing)."""
    from core.csv_reader import AISDataReader
    from core.trajectory_sampler import sample_vessel_trajectories
    from core.data_converter import trajectory_to_geojson
    
    input_path = params['input_path']
    vessel_ids = params['vessel_ids']
    vessel_type = params['vessel_type']
    attack_type = params.get('attack_type')
    sample_config = params['sample_config']
    
    # Read data for these vessels
    reader = AISDataReader(input_path)
    features = []
    
    for vessel_id in vessel_ids:
        vessel_df = reader.get_vessel_data(vessel_id)
        
        if not vessel_df.empty:
            # Apply sampling
            sampled = sample_vessel_trajectories(
                vessel_df, 
                sample_config,
                vessel_column='mmsi'
            )
            
            # Convert to GeoJSON
            feature = trajectory_to_geojson(sampled, vessel_type, attack_type)
            if feature:
                features.append(feature)
    
    return {
        'features': features,
        'vessel_count': len(vessel_ids),
        'feature_count': len(features)
    }


def convert_with_multiprocessing(
    input_path: str,
    output_path: str,
    vessel_type: str = 'baseline',
    attack_type: Optional[str] = None,
    scenario: str = 'S1',
    workers: int = 4,
    show_progress: bool = False
) -> Dict[str, Any]:
    """Convert CSV to GeoJSON using multiprocessing."""
    from core.csv_reader import AISDataReader
    from core.trajectory_sampler import create_scenario_config
    from core.geojson_types import create_feature_collection, save_geojson
    
    logger.info(f"Converting {input_path} using {workers} workers...")
    
    # Get vessel list
    reader = AISDataReader(input_path)
    vessel_list = reader.get_vessel_list()
    total_vessels = len(vessel_list)
    
    logger.info(f"Found {total_vessels} vessels to process")
    
    # Split vessels among workers
    chunk_size = max(1, total_vessels // workers)
    vessel_chunks = [vessel_list[i:i+chunk_size] 
                    for i in range(0, total_vessels, chunk_size)]
    
    # Prepare parameters for each worker
    sample_config = create_scenario_config(scenario)
    params_list = []
    
    for chunk in vessel_chunks:
        params_list.append({
            'input_path': input_path,
            'vessel_ids': chunk,
            'vessel_type': vessel_type,
            'attack_type': attack_type,
            'sample_config': sample_config
        })
    
    # Process in parallel
    all_features = []
    ctx = get_context('spawn')  # Use spawn for Windows compatibility
    
    with ctx.Pool(processes=workers) as pool:
        if show_progress:
            results = list(tqdm(
                pool.imap(process_vessel_group, params_list),
                total=len(params_list),
                desc="Processing chunks"
            ))
        else:
            results = pool.map(process_vessel_group, params_list)
    
    # Collect all features
    for result in results:
        all_features.extend(result['features'])
    
    # Get data summary
    data_summary = reader.get_data_summary()
    
    # Create metadata
    metadata = {
        'source_file': Path(input_path).name,
        'vessel_type': vessel_type,
        'scenario': scenario,
        'conversion_stats': {
            'initial_points': data_summary['total_rows'],
            'initial_vessels': data_summary['vessel_count'],
            'output_vessels': len(all_features),
            'workers_used': workers
        }
    }
    
    if attack_type:
        metadata['attack_type'] = attack_type
    
    # Create and save FeatureCollection
    feature_collection = create_feature_collection(all_features, metadata)
    save_geojson(feature_collection, output_path)
    
    # Calculate output size
    output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    
    stats = {
        'input_file': input_path,
        'output_file': output_path,
        'initial_points': data_summary['total_rows'],
        'vessels': len(all_features),
        'output_size_mb': round(output_size_mb, 2),
        'workers_used': workers
    }
    
    logger.info(f"Created {output_path} ({output_size_mb:.1f} MB) "
                f"with {len(all_features)} vessel trajectories")
    
    return stats


def extract_zones(args: argparse.Namespace) -> None:
    """Extract and save guard zones."""
    output_dir = Path(args.out_dir) if args.out_dir else Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = ZoneExtractor()
    
    # Load zones from config
    logger.info(f"Loading zones from {args.zone_config}")
    extractor.load_from_config(args.zone_config)
    
    # Merge overlapping zones
    extractor.merge_overlapping_zones()
    
    # Simplify if needed
    extractor.simplify_zones()
    
    # Save to GeoJSON
    output_file = output_dir / 'zones.geojson'
    extractor.save_geojson(str(output_file))
    
    # Print summary
    summary = extractor.get_zone_summary()
    logger.info(f"Zone summary: {summary}")


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Track execution time
    start_time = time.time()
    
    try:
        # Handle zone extraction
        if args.zones_only:
            if not args.zone_config:
                parser.error("--zone-config required when using --zones-only")
            extract_zones(args)
            return
        
        # Extract zones if config provided
        if args.zone_config:
            extract_zones(args)
        
        # Handle different conversion modes
        if args.batch:
            if not args.input_dir:
                parser.error("--input-dir required for batch mode")
            if not args.out_dir:
                parser.error("--out-dir required for batch mode")
            
            logger.info(f"Batch converting files in {args.input_dir}")
            results = batch_convert_directory(
                input_dir=args.input_dir,
                output_dir=args.out_dir,
                scenario=args.scenario
            )
            
            # Print summary
            success_count = sum(1 for r in results if 'error' not in r)
            logger.info(f"Converted {success_count}/{len(results)} files successfully")
            
        elif args.baseline:
            if not args.out_dir and not args.output:
                parser.error("Either --out-dir or --output required")
            
            # Use multiprocessing for large files
            if args.workers > 1:
                output_dir = Path(args.out_dir) if args.out_dir else Path('.')
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Convert baseline with multiprocessing
                baseline_output = output_dir / 'baseline.geojson'
                stats = convert_with_multiprocessing(
                    args.baseline,
                    str(baseline_output),
                    vessel_type='baseline',
                    scenario=args.scenario,
                    workers=args.workers,
                    show_progress=args.progress
                )
                
                # Convert attack if provided
                if args.attack:
                    attack_output = output_dir / 'attack.geojson'
                    attack_type = args.attack_type
                    if not attack_type and 'zone' in args.attack.lower():
                        attack_type = 'zone_violation'
                    
                    stats_attack = convert_with_multiprocessing(
                        args.attack,
                        str(attack_output),
                        vessel_type='attack',
                        attack_type=attack_type,
                        scenario=args.scenario,
                        workers=args.workers,
                        show_progress=args.progress
                    )
            else:
                results = convert_baseline_attack(args)
            
        elif args.input:
            if not args.output and not args.out_dir:
                parser.error("Either --output or --out-dir required")
            
            if args.workers > 1:
                # Determine output path
                if args.output:
                    output_file = args.output
                else:
                    input_path = Path(args.input)
                    output_dir = Path(args.out_dir)
                    output_file = output_dir / (input_path.stem + '.geojson')
                
                stats = convert_with_multiprocessing(
                    args.input,
                    str(output_file),
                    vessel_type=args.vessel_type,
                    attack_type=args.attack_type,
                    scenario=args.scenario,
                    workers=args.workers,
                    show_progress=args.progress
                )
            else:
                stats = convert_single_file(args)
            
        else:
            parser.error("No input specified. Use -i, --baseline, or --batch")
        
        # Print execution time
        elapsed_time = time.time() - start_time
        logger.info(f"Total execution time: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()