#!/usr/bin/env python3
"""Fix NaN values in existing GeoJSON files."""

import json
import sys
from pathlib import Path

def fix_nan_in_json(input_path, output_path=None):
    """Fix NaN values in a JSON file by replacing them with null."""
    if output_path is None:
        output_path = input_path
    
    # Read the file as text
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace NaN and Infinity values
    content = content.replace(': NaN', ': null')
    content = content.replace(':NaN', ':null')
    content = content.replace(': Infinity', ': null')
    content = content.replace(':Infinity', ':null')
    content = content.replace(': -Infinity', ': null')
    content = content.replace(':-Infinity', ':null')
    
    # Parse to validate JSON
    try:
        data = json.loads(content)
        print(f"✓ Successfully parsed JSON from {input_path}")
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse JSON: {e}")
        return False
    
    # Write back with proper formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Fixed NaN values and saved to {output_path}")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_geojson_nan.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    success = fix_nan_in_json(input_file, output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()