"""
BudgetWise AI - Streamlit Cloud Entry Point
This file serves as the main entry point for Streamlit Cloud deployment.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
current_dir = Path(__file__).parent.absolute()
app_dir = current_dir / "app"

# Try multiple possible paths for the app directory
possible_app_dirs = [
    app_dir,  # Standard location
    current_dir,  # Current directory
    Path("app"),  # Relative app directory
]

# Find the correct app directory
actual_app_dir = None
for path in possible_app_dirs:
    if path.exists() and (path / "budgetwise_app.py").exists():
        actual_app_dir = path
        break

if actual_app_dir is not None:
    sys.path.insert(0, str(actual_app_dir))
else:
    # If we can't find the app directory, try to find budgetwise_app.py directly
    possible_app_files = [
        current_dir / "app" / "budgetwise_app.py",
        current_dir / "budgetwise_app.py",
        Path("app") / "budgetwise_app.py",
        Path("budgetwise_app.py"),
    ]
    
    for app_file in possible_app_files:
        if app_file.exists():
            sys.path.insert(0, str(app_file.parent))
            actual_app_dir = app_file.parent
            break

# Import and run the main application
try:
    from budgetwise_app import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    import streamlit as st
    st.error(f"Error importing application: {e}")
    st.info("Please ensure all dependencies are installed and the app files are accessible.")
    st.info(f"Current directory: {current_dir}")
    st.info(f"App directory: {actual_app_dir}")