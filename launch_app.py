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
    
    if not app_dir.exists():
        print("❌ Error: app directory not found!")
        print("Please run this script from the BudgetWise project root directory.")
        return
    
    # Check if required files exist
    app_file = app_dir / "budgetwise_app.py"
    if not app_file.exists():
        print("❌ Error: budgetwise_app.py not found!")
        return
    
    data_dir = current_dir / "data" / "processed"
    if not data_dir.exists():
        print("❌ Error: processed data directory not found!")
        print("Please ensure data preprocessing is complete.")
        return
    
    print("✅ All requirements checked successfully!")
    print("📊 Launching BudgetWise AI Dashboard...")
    print("🌐 The application will open in your browser at: http://localhost:8502")
    print("⚠️  Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        # Change to app directory and run streamlit
        os.chdir(app_dir)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "budgetwise_app.py", "--server.port", "8502"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main()