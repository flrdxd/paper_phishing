"""
Central Path Configuration for Location-Independent Script Execution

This module handles all path definitions and ensures scripts can run from any directory
on both local machines and remote servers.
"""

import os
import sys

def setup_project_paths():
    """
    Setup project paths and ensure imports work from any directory.

    Returns:
        dict: Dictionary containing all project directory paths
    """
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Add to sys.path if not already present
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Define all project directories
    return {
        'PROJECT_ROOT': project_root,
        'SCRIPT_DIR': script_dir,
        'MODELS_DIR': os.path.join(project_root, 'models'),
        'PLOTS_DIR': os.path.join(project_root, 'plots'),
        'RESULTS_DIR': os.path.join(project_root, 'results'),
        'DATA_DIR': os.path.join(project_root, 'data'),
        'LOGS_DIR': os.path.join(project_root, 'logs'),
        'TEMP_DIR': os.path.join(project_root, 'temp')
    }

# Initialize paths
PATHS = setup_project_paths()

# Ensure directories exist
for dir_name, dir_path in PATHS.items():
    if isinstance(dir_path, str) and dir_name.endswith('_DIR'):
        os.makedirs(dir_path, exist_ok=True)

def get_path(path_name):
    """
    Get a specific path by name.

    Args:
        path_name: Name of the path (e.g., 'MODELS_DIR')

    Returns:
        str: Path string
    """
    return PATHS.get(path_name, '')

def ensure_dir_exists(path):
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Directory path to check/create
    """
    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    # Test path configuration
    print("Project Path Configuration:")
    print(f"Project Root: {PATHS['PROJECT_ROOT']}")
    print(f"Script Dir: {PATHS['SCRIPT_DIR']}")
    print(f"Models Dir: {PATHS['MODELS_DIR']}")
    print(f"Plots Dir: {PATHS['PLOTS_DIR']}")
    print(f"Results Dir: {PATHS['RESULTS_DIR']}")
    print(f"Data Dir: {PATHS['DATA_DIR']}")
    print(f"Logs Dir: {PATHS['LOGS_DIR']}")
    print(f"Temp Dir: {PATHS['TEMP_DIR']}")
    print("\nAll directories verified and created if needed.")