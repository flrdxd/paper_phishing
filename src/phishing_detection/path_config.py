"""
Central Path Configuration for Location-Independent Script Execution

This module handles all path definitions and ensures scripts can run from any directory
on both local machines and remote servers.
"""

from pathlib import Path

def setup_project_paths():
    """
    Setup project paths and ensure imports work from any directory.

    Returns:
        dict: Dictionary containing all project directory paths
    """
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]
    artifacts_dir = project_root / '.artifacts'

    # Define all project directories
    return {
        'PROJECT_ROOT': str(project_root),
        'PACKAGE_DIR': str(package_dir),
        'ARTIFACTS_DIR': str(artifacts_dir),
        'MODELS_DIR': str(artifacts_dir / 'models'),
        'PLOTS_DIR': str(artifacts_dir / 'plots'),
        'RESULTS_DIR': str(artifacts_dir / 'results'),
        'DATA_DIR': str(artifacts_dir / 'data'),
        'LOGS_DIR': str(artifacts_dir / 'logs'),
        'TEMP_DIR': str(artifacts_dir / 'temp')
    }

# Initialize paths
PATHS = setup_project_paths()

# Ensure directories exist
for dir_name, dir_path in PATHS.items():
    if isinstance(dir_path, str) and dir_name.endswith('_DIR'):
        ensure_dir = Path(dir_path)
        ensure_dir.mkdir(parents=True, exist_ok=True)

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
    Path(path).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Test path configuration
    print("Project Path Configuration:")
    print(f"Project Root: {PATHS['PROJECT_ROOT']}")
    print(f"Package Dir: {PATHS['PACKAGE_DIR']}")
    print(f"Artifacts Dir: {PATHS['ARTIFACTS_DIR']}")
    print(f"Models Dir: {PATHS['MODELS_DIR']}")
    print(f"Plots Dir: {PATHS['PLOTS_DIR']}")
    print(f"Results Dir: {PATHS['RESULTS_DIR']}")
    print(f"Data Dir: {PATHS['DATA_DIR']}")
    print(f"Logs Dir: {PATHS['LOGS_DIR']}")
    print(f"Temp Dir: {PATHS['TEMP_DIR']}")
    print("\nAll directories verified and created if needed.")
