#!/usr/bin/env python3
"""
System Check Tool for AIS Attack Generation System

This tool verifies that all system components are properly installed and configured.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message: str, status: bool, quiet: bool = False):
    """Print status message with color coding"""
    if quiet and status:
        return
    
    if status:
        icon = f"{Colors.GREEN}✅{Colors.ENDC}"
        status_text = f"{Colors.GREEN}PASS{Colors.ENDC}"
    else:
        icon = f"{Colors.RED}❌{Colors.ENDC}"
        status_text = f"{Colors.RED}FAIL{Colors.ENDC}"
    
    print(f"{icon} {message}: {status_text}")

def check_python_version() -> bool:
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    return version.major == 3 and version.minor >= 8

def check_required_packages() -> Tuple[bool, List[str]]:
    """Check if all required Python packages are installed"""
    required_packages = [
        'numpy',
        'pandas', 
        'scipy',
        'matplotlib',
        'geopandas',
        'shapely',
        'pyproj',
        'pyyaml',
        'plotly',
        'scikit-learn',
        'pytest'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_optional_packages() -> Dict[str, bool]:
    """Check optional packages for enhanced functionality"""
    optional_packages = {
        'numba': 'Performance optimization',
        'pyais': 'AIS message parsing',
        'fastapi': 'Web API functionality',
        'uvicorn': 'Web server',
        'dash': 'Interactive dashboards'
    }
    
    results = {}
    for package, description in optional_packages.items():
        try:
            importlib.import_module(package)
            results[package] = True
        except ImportError:
            results[package] = False
    
    return results

def check_nodejs() -> bool:
    """Check if Node.js is installed (for web interface)"""
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            # Extract major version number
            major_version = int(version.lstrip('v').split('.')[0])
            return major_version >= 16
        return False
    except FileNotFoundError:
        return False

def check_directories() -> Tuple[bool, List[str]]:
    """Check if required directories exist"""
    required_dirs = [
        'core',
        'attacks', 
        'visualization',
        'datasets',
        'tools',
        'configs',
        'docs'
    ]
    
    missing_dirs = []
    project_root = Path(__file__).parent.parent
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0, missing_dirs

def check_config_files() -> Tuple[bool, List[str]]:
    """Check if required configuration files exist"""
    required_configs = [
        'configs/default_attack_config.yaml',
        'requirements.txt',
        'setup.py'
    ]
    
    missing_configs = []
    project_root = Path(__file__).parent.parent
    
    for config_file in required_configs:
        config_path = project_root / config_file
        if not config_path.exists():
            missing_configs.append(config_file)
    
    return len(missing_configs) == 0, missing_configs

def check_core_modules() -> Tuple[bool, List[str]]:
    """Check if core modules can be imported"""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    core_modules = [
        'core.attack_orchestrator',
        'core.target_selector', 
        'core.physics_engine',
        'core.colregs_validator',
        'core.auto_labeler'
    ]
    
    failed_imports = []
    
    for module in core_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            failed_imports.append(f"{module}: {str(e)}")
    
    return len(failed_imports) == 0, failed_imports

def check_attack_modules() -> Tuple[bool, List[str]]:
    """Check if attack modules can be imported"""
    attack_modules = [
        'attacks.flash_cross',
        'attacks.zone_violation',
        'attacks.ghost_swarm'
    ]
    
    failed_imports = []
    
    for module in attack_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            failed_imports.append(f"{module}: {str(e)}")
    
    return len(failed_imports) == 0, failed_imports

def check_memory() -> Tuple[bool, str]:
    """Check available memory"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb >= 4.0:
            return True, f"{available_gb:.1f} GB available"
        else:
            return False, f"Only {available_gb:.1f} GB available (recommended: 4GB+)"
    except ImportError:
        return True, "Unable to check (psutil not installed)"

def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space"""
    try:
        import shutil
        project_root = Path(__file__).parent.parent
        total, used, free = shutil.disk_usage(project_root)
        free_gb = free / (1024**3)
        
        if free_gb >= 2.0:
            return True, f"{free_gb:.1f} GB available"
        else:
            return False, f"Only {free_gb:.1f} GB available (recommended: 2GB+)"
    except:
        return True, "Unable to check disk space"

def run_system_check(quiet: bool = False, verbose: bool = False) -> bool:
    """Run comprehensive system check"""
    if not quiet:
        print(f"{Colors.BLUE}{Colors.BOLD}🚢 AIS Attack System - System Check{Colors.ENDC}")
        print(f"{Colors.BLUE}{'='*50}{Colors.ENDC}")
        print()
    
    all_checks_passed = True
    
    # Python version check
    python_ok = check_python_version()
    print_status(f"Python version {sys.version.split()[0]}", python_ok, quiet)
    all_checks_passed &= python_ok
    
    # Required packages check
    packages_ok, missing = check_required_packages()
    print_status("Required Python packages", packages_ok, quiet)
    if not packages_ok and verbose:
        print(f"  {Colors.RED}Missing packages: {', '.join(missing)}{Colors.ENDC}")
        print(f"  {Colors.YELLOW}Install with: pip install {' '.join(missing)}{Colors.ENDC}")
    all_checks_passed &= packages_ok
    
    # Optional packages check
    if verbose:
        optional_results = check_optional_packages()
        for package, installed in optional_results.items():
            print_status(f"Optional: {package}", installed, quiet)
    
    # Node.js check (optional)
    nodejs_ok = check_nodejs()
    print_status("Node.js 16+ (for web interface)", nodejs_ok, quiet)
    if not nodejs_ok and verbose:
        print(f"  {Colors.YELLOW}Node.js not found or version < 16{Colors.ENDC}")
        print(f"  {Colors.YELLOW}Install from: https://nodejs.org/{Colors.ENDC}")
    
    # Directory structure check
    dirs_ok, missing_dirs = check_directories()
    print_status("Project directory structure", dirs_ok, quiet)
    if not dirs_ok and verbose:
        print(f"  {Colors.RED}Missing directories: {', '.join(missing_dirs)}{Colors.ENDC}")
    all_checks_passed &= dirs_ok
    
    # Configuration files check
    configs_ok, missing_configs = check_config_files()
    print_status("Configuration files", configs_ok, quiet)
    if not configs_ok and verbose:
        print(f"  {Colors.RED}Missing files: {', '.join(missing_configs)}{Colors.ENDC}")
    all_checks_passed &= configs_ok
    
    # Core modules check
    core_ok, failed_core = check_core_modules()
    print_status("Core modules", core_ok, quiet)
    if not core_ok and verbose:
        print(f"  {Colors.RED}Failed imports:{Colors.ENDC}")
        for failure in failed_core:
            print(f"    {failure}")
    all_checks_passed &= core_ok
    
    # Attack modules check
    attacks_ok, failed_attacks = check_attack_modules()
    print_status("Attack modules", attacks_ok, quiet)
    if not attacks_ok and verbose:
        print(f"  {Colors.RED}Failed imports:{Colors.ENDC}")
        for failure in failed_attacks:
            print(f"    {failure}")
    all_checks_passed &= attacks_ok
    
    # System resources check
    memory_ok, memory_info = check_memory()
    print_status(f"Memory ({memory_info})", memory_ok, quiet)
    
    disk_ok, disk_info = check_disk_space()
    print_status(f"Disk space ({disk_info})", disk_ok, quiet)
    
    if not quiet:
        print()
        if all_checks_passed:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 All critical checks passed! System is ready.{Colors.ENDC}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ Some critical checks failed. Please fix the issues above.{Colors.ENDC}")
        
        print()
        print("Next steps:")
        if all_checks_passed:
            print("• Run your first attack: python -m core.attack_orchestrator --scenario s1_flash_cross")
            print("• Start web interface: cd visualization/web_interface && npm run dev")
            print("• View documentation: docs/QUICK_START.md")
        else:
            print("• Install missing dependencies: pip install -r requirements.txt")
            print("• Check project structure and configuration files")
            print("• Re-run system check: python tools/system_check.py")
    
    return all_checks_passed

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AIS Attack System - System Check")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only show failed checks")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed information about failures")
    
    args = parser.parse_args()
    
    success = run_system_check(quiet=args.quiet, verbose=args.verbose)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
