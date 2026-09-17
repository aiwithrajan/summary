import os
import sys

# Add parent directory to sys.path to resolve agent and server modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app
