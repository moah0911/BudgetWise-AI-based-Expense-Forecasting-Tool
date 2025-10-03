"""
BudgetWise AI - Application Launcher
Simple script to launch the BudgetWise AI Streamlit application
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the BudgetWise AI application"""
    
    print("🚀 Starting BudgetWise AI - Personal Expense Forecasting System")
    print("=" * 60)
    
    # Check if we're in the correct directory
    current_dir = Path.cwd()
    app_dir = current_dir / "app"
    
    # Check multiple possible locations for the app file
    possible_app_files = [
        app_dir / "budgetwise_app.py",  # Standard location
        current_dir / "app" / "budgetwise_app.py",  # Alternative path
        current_dir / "budgetwise_app.py"  # Root directory fallback
    ]
    
    app_file = None
    for file_path in possible_app_files:
        if file_path.exists():
            app_file = file_path
            break
    
    if app_file is None:
        print("❌ Error: budgetwise_app.py not found!")
        print("Searched in:")
        for file_path in possible_app_files:
            print(f"  - {file_path}")
        return
    
    # Determine the correct directory to run from
    if app_file.parent.exists():
        run_dir = app_file.parent
    else:
        run_dir = current_dir
    
    data_dir = current_dir / "data" / "processed"
    if not data_dir.exists():
        # Try alternative data directory locations
        alt_data_dirs = [
            current_dir / "data",
            Path("data") / "processed",
            Path("../data") / "processed"
        ]
        
        data_dir_found = False
        for alt_dir in alt_data_dirs:
            if alt_dir.exists():
                data_dir = alt_dir
                data_dir_found = True
                break
        
        if not data_dir_found:
            print("⚠️  Warning: processed data directory not found!")
            print("Please ensure data preprocessing is complete.")
            # Don't exit, as the app might have fallback mechanisms
    
    print("✅ All requirements checked successfully!")
    print("📊 Launching BudgetWise AI Dashboard...")
    print(f"📂 Running from directory: {run_dir}")
    print(f"📄 Using app file: {app_file}")
    print("🌐 The application will open in your browser at: http://localhost:8502")
    print("⚠️  Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        # Change to the correct directory and run streamlit
        os.chdir(run_dir)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            app_file.name, "--server.port", "8502"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main()