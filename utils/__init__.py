# -*- coding: utf-8 -*-
# utils package
# Import functions from root utils.py to make them accessible from utils package
import sys
import os
import importlib.util

# Get project root directory
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import functions from root utils.py using importlib to avoid name conflict
_utils_path = os.path.join(_project_root, "utils.py")
if os.path.exists(_utils_path):
    try:
        # Use a unique module name to avoid conflicts
        spec = importlib.util.spec_from_file_location("_root_utils_module", _utils_path)
        _root_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_root_utils)
        # Re-export get_info_score and load_ds functions
        if hasattr(_root_utils, 'get_info_score'):
            get_info_score = _root_utils.get_info_score
        if hasattr(_root_utils, 'load_ds'):
            load_ds = _root_utils.load_ds
    except Exception as e:
        # If import fails, raise a more informative error
        raise ImportError(
            f"Failed to import functions from utils.py at {_utils_path}: {e}. "
            f"Please ensure all dependencies (hydra, torch, etc.) are installed."
        ) from e
else:
    raise ImportError("Cannot find utils.py at {}".format(_utils_path))
