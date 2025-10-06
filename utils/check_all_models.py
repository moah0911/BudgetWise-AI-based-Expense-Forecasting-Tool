#!/usr/bin/env python3
"""
Check all trained models in result files
"""

import pandas as pd
from pathlib import Path

def check_all_models():
    print("🔍 CHECKING ALL TRAINED MODELS")
    print("=" * 50)
    
    model_files = {
        'Baseline': 'models/baseline/baseline_results.csv',
        'Machine Learning': 'models/ml/ml_results.csv', 
        'Deep Learning': 'models/deep_learning/dl_results.csv',
        'Transformer': 'models/transformer/transformer_results.csv'
    }
    
    total_models = 0
    all_models = []
    
    for category, file_path in model_files.items():
        try:
            df = pd.read_csv(file_path)
            if 'model_name' in df.columns:
                models = df['model_name'].tolist()
            else:
                # Use index if no model_name column
                df = pd.read_csv(file_path, index_col=0)
                models = df.index.tolist()
            
            print(f"\n📊 {category} ({len(models)} models):")
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model}")
                all_models.append(f"{category}: {model}")
            
            total_models += len(models)
            
        except Exception as e:
            print(f"\n❌ Error loading {category}: {e}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   • Total models trained: {total_models}")
    print(f"   • Categories: {len(model_files)}")
    
    print(f"\n📋 ALL MODELS LIST:")
    for i, model in enumerate(all_models, 1):
        print(f"   {i}. {model}")

if __name__ == "__main__":
    check_all_models()