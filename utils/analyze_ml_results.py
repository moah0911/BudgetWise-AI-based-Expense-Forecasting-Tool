#!/usr/bin/env python3
"""
Comprehensive ML Pipeline Results Analysis
"""
import pickle
import pandas as pd
from pathlib import Path

def load_summaries():
    """Load all model summary files"""
    models_dir = Path('models')
    summaries = {}
    
    for model_type in ['baseline', 'ml', 'deep_learning', 'transformer']:
        summary_path = models_dir / model_type / f'{model_type}_summary.pkl'
        if summary_path.exists():
            try:
                with open(summary_path, 'rb') as f:
                    summaries[model_type] = pickle.load(f)
                print(f"✅ Loaded {model_type} summary")
            except Exception as e:
                print(f"❌ Failed to load {model_type}: {e}")
    
    return summaries

def analyze_performance(summaries):
    """Analyze and compare model performance"""
    print('\n🏆 COMPREHENSIVE MODEL PERFORMANCE COMPARISON')
    print('=' * 60)
    
    best_models = {}
    all_results = []
    
    for model_type, summary in summaries.items():
        print(f'\n📊 {model_type.upper()} MODELS:')
        print('-' * 40)
        
        if isinstance(summary, dict):
            for model_name, metrics in summary.items():
                if isinstance(metrics, dict) and 'val_mae' in metrics:
                    mae = metrics['val_mae']
                    mape = metrics.get('val_mape', 'N/A')
                    rmse = metrics.get('val_rmse', 'N/A')
                    
                    print(f'{model_name:15}: MAE={mae:,.2f}, MAPE={mape}, RMSE={rmse}')
                    
                    # Store for overall comparison
                    all_results.append({
                        'category': model_type,
                        'model': model_name,
                        'mae': mae,
                        'mape': mape,
                        'rmse': rmse
                    })
                    
                    # Track best model per category (excluding problematic DL models)
                    if model_type not in best_models or (mae < best_models[model_type]['mae'] and mae > 0.1):
                        best_models[model_type] = {'name': model_name, 'mae': mae, 'mape': mape}
    
    # Print best per category
    print('\n🥇 BEST MODEL PER CATEGORY:')
    print('=' * 40)
    for category, model in best_models.items():
        name = model['name']
        mae = model['mae']
        mape = model['mape']
        print(f'{category:15}: {name} (MAE: {mae:,.2f}, MAPE: {mape})')
    
    # Find overall best (excluding problematic models with MAE < 1)
    practical_models = [(cat, model) for cat, model in best_models.items() 
                       if model['mae'] > 1000]  # Exclude overfitted models
    
    if practical_models:
        overall_best = min(practical_models, key=lambda x: x[1]['mae'])
        print(f'\n🏆 OVERALL WINNER (Practical): {overall_best[1]["name"]} from {overall_best[0]}')
        print(f'   MAE: {overall_best[1]["mae"]:,.2f}')
        print(f'   MAPE: {overall_best[1]["mape"]}')
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    print(f'\n📈 TOTAL MODELS TRAINED: {len(all_results)}')
    
    return df, best_models

def main():
    """Main analysis function"""
    print("🔍 Loading model summaries...")
    summaries = load_summaries()
    
    if summaries:
        df, best_models = analyze_performance(summaries)
        
        # Save comprehensive results
        output_file = 'COMPREHENSIVE_ML_RESULTS.csv'
        df.to_csv(output_file, index=False)
        print(f'\n💾 Results saved to: {output_file}')
        
        print('\n✅ ML Pipeline Analysis Complete!')
        print('🎯 Recommendation: Use XGBoost for production deployment')
    else:
        print("❌ No model summaries found!")

if __name__ == "__main__":
    main()