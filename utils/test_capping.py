#!/usr/bin/env python3
"""
Quick test script to verify transaction capping fix
"""

import pandas as pd
from src.data_preprocessing import AdvancedDataPreprocessor

def test_transaction_capping():
    print("🧪 Testing Transaction Capping Fix")
    print("=" * 50)
    
    # Load raw data
    print("📁 Loading raw dataset...")
    df = pd.read_csv('data/raw/budgetwise_finance_dataset.csv')
    print(f"   Original dataset shape: {df.shape}")
    
    # Show original amount statistics
    print(f"\n📊 Original amount statistics:")
    original_amounts = pd.to_numeric(df['amount'].astype(str).str.replace(',', '').str.replace('₹', '').str.replace('$', ''), errors='coerce').dropna()
    print(f"   • Min: ₹{original_amounts.min():,.2f}")
    print(f"   • Max: ₹{original_amounts.max():,.2f}")
    print(f"   • Mean: ₹{original_amounts.mean():,.2f}")
    print(f"   • Median: ₹{original_amounts.median():,.2f}")
    
    # Process with capping
    print(f"\n🔧 Processing with transaction capping...")
    preprocessor = AdvancedDataPreprocessor()
    processed_df = preprocessor.advanced_amount_processing(df)
    
    # Show final results
    print(f"\n✅ Final Results:")
    print(f"   • Processed dataset shape: {processed_df.shape}")
    print(f"   • Amount range: ₹{processed_df['amount'].min():.2f} to ₹{processed_df['amount'].max():.2f}")
    print(f"   • Mean amount: ₹{processed_df['amount'].mean():.2f}")
    print(f"   • Median amount: ₹{processed_df['amount'].median():.2f}")
    
    # Check capping statistics
    if 'amount_capped' in processed_df.columns:
        total_capped = processed_df['amount_capped'].sum()
        capped_high = processed_df['amount_capped_high'].sum()
        capped_low = processed_df['amount_capped_low'].sum()
        print(f"\n🎯 Capping Statistics:")
        print(f"   • Total transactions capped: {total_capped:,}")
        print(f"   • High amounts capped: {capped_high:,}")
        print(f"   • Low amounts capped: {capped_low:,}")
    
    print(f"\n🎉 Transaction capping test completed successfully!")
    return processed_df

if __name__ == "__main__":
    result_df = test_transaction_capping()