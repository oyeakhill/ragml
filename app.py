"""
Redirect to enhanced_app.py - the main application file.

This file exists for backward compatibility.
"""

import subprocess
import sys

print("Redirecting to enhanced_app.py...")
subprocess.run([sys.executable, "-m", "streamlit", "run", "enhanced_app.py"])
