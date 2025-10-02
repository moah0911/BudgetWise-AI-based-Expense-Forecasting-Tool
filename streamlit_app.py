"""
BudgetWise AI - Streamlit Cloud Entry Point
This file serves as the main entry point for Streamlit Cloud deployment.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
current_dir = Path(__file__).parent
app_dir = current_dir / "app"
sys.path.insert(0, str(app_dir))

# Import and run the main application
try:
    from budgetwise_app import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    import streamlit as st
    st.error(f"Error importing application: {e}")
    st.info("Please ensure all dependencies are installed and the app directory is accessible.")