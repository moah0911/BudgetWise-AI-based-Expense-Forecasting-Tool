#!/usr/bin/env python3
"""Test script to verify the data loading fix"""

from pathlib import Path
import pandas as pd

def test_data_loading():
    """Test the improved data loading logic"""
    # Simulate the path resolution from the app
    current_dir = Path(__file__).parent.absolute()
    app_dir = current_dir / "app"
    root_dir = current_dir  # We're already in root
    data_path = root_dir / "data" / "processed"
    
    print(f"🔍 Path Resolution Test:")
    print(f"Current dir: {current_dir}")
    print(f"Root dir: {root_dir}")
    print(f"Data path: {data_path}")
    print(f"Train data exists: {(data_path / 'train_data.csv').exists()}")
    
    if (data_path / 'train_data.csv').exists():
        print(f"\n📊 Loading Real Data:")
        # Load the actual processed data
        train_df = pd.read_csv(data_path / 'train_data.csv')
        val_df = pd.read_csv(data_path / 'val_data.csv')
        test_df = pd.read_csv(data_path / 'test_data.csv')
        
        # Combine all data
        all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        print(f"✅ Real data loaded successfully!")
        print(f"   Total records: {len(all_data):,}")
        print(f"   Max daily expense: ₹{all_data['total_daily_expense'].max():,.2f}")
        print(f"   Mean daily expense: ₹{all_data['total_daily_expense'].mean():,.2f}")
        print(f"   95th percentile: ₹{all_data['total_daily_expense'].quantile(0.95):,.2f}")
        print(f"   Date range: {all_data['date'].min()} to {all_data['date'].max()}")
        
        # Compare with what we expect vs synthetic data
        if all_data['total_daily_expense'].max() > 10000:
            print(f"⚠️  WARNING: Max expense > ₹10,000 suggests synthetic data might still be used")
        else:
            print(f"✅ Data looks realistic (max expense ≤ ₹10,000)")
            
    else:
        print(f"❌ Real data not found at {data_path}")

if __name__ == "__main__":
    test_data_loading()