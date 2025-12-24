"""Pytest configuration and shared fixtures."""

import os
import sys

# Add the parent directory to the path so we can import server
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
