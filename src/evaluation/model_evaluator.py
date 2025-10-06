"""
Model Evaluation Framework for BudgetWise Forecasting System
Provides comprehensive evaluation and comparison of all models.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison framework.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.models_path = Path("models")
        self.reports_path = Path("reports")
        self.reports_path.mkdir(exist_ok=True)
        
        # Load performance data
        self.baseline_performance = self._load_performance("baseline_performance.pkl")
        self.ml_performance = self._load_performance("ml_performance.pkl")
        self.dl_performance = self._load_performance("dl_performance.pkl")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {}
    
    def _load_performance(self, filename: str) -> Dict:
        """Load performance data from pickle file."""
        try:
            return joblib.load(self.models_path / filename)
        except FileNotFoundError:
            logger.warning(f"Performance file {filename} not found.")
            return {}
    
    def combine_all_performance(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Combine all model performance data.
        
        Returns:
            Dictionary with structure: {category: {model_type: {metric: value}}}
        """
        combined_performance = {}
        
        # Get all categories
        all_categories = set()
        for perf_data in [self.baseline_performance, self.ml_performance, self.dl_performance]:
            all_categories.update(perf_data.keys())
        
        for category in all_categories:
            combined_performance[category] = {}
            
            # Add baseline models
            if category in self.baseline_performance:
                for model_name, metrics in self.baseline_performance[category].items():
                    combined_performance[category][f"baseline_{model_name}"] = metrics
            
            # Add ML models
            if category in self.ml_performance:
                for model_name, metrics in self.ml_performance[category].items():
                    combined_performance[category][f"ml_{model_name}"] = metrics
            
            # Add DL models
            if category in self.dl_performance:
                for model_name, metrics in self.dl_performance[category].items():
                    combined_performance[category][f"dl_{model_name}"] = metrics
        
        return combined_performance
    
    def calculate_model_rankings(self, performance_data: Dict) -> Dict[str, Dict[str, int]]:
        """
        Calculate model rankings for each category based on MAE.
        
        Args:
            performance_data: Combined performance data
            
        Returns:
            Dictionary with model rankings for each category
        """
        rankings = {}
        
        for category, models in performance_data.items():
            if not models:
                continue
            
            # Sort models by MAE (lower is better)
            sorted_models = sorted(
                models.items(),
                key=lambda x: x[1].get('MAE', float('inf'))
            )
            
            rankings[category] = {}
            for rank, (model_name, _) in enumerate(sorted_models, 1):
                rankings[category][model_name] = rank
        
        return rankings
    
    def generate_performance_summary_table(self, performance_data: Dict) -> pd.DataFrame:
        """
        Generate a comprehensive performance summary table.
        
        Args:
            performance_data: Combined performance data
            
        Returns:
            DataFrame with performance summary
        """
        summary_data = []
        
        for category, models in performance_data.items():
            for model_name, metrics in models.items():
                summary_data.append({
                    'Category': category,
                    'Model': model_name,
                    'MAE': metrics.get('MAE', np.nan),
                    'RMSE': metrics.get('RMSE', np.nan),
                    'MAPE': metrics.get('MAPE', np.nan),
                    'R2': metrics.get('R2', np.nan),
                    'Directional_Accuracy': metrics.get('Directional_Accuracy', np.nan)
                })
        
        return pd.DataFrame(summary_data)
    
    def find_best_models(self, performance_data: Dict) -> Dict[str, Tuple[str, Dict[str, float]]]:
        """
        Find the best performing model for each category.
        
        Args:
            performance_data: Combined performance data
            
        Returns:
            Dictionary with best model for each category
        """
        best_models = {}
        
        for category, models in performance_data.items():
            if not models:
                continue
            
            best_model_name = min(
                models.keys(),
                key=lambda x: models[x].get('MAE', float('inf'))
            )
            
            best_models[category] = (best_model_name, models[best_model_name])
        
        return best_models
    
    def create_performance_visualizations(self, performance_data: Dict) -> None:
        """
        Create various performance visualization plots.
        
        Args:
            performance_data: Combined performance data
        """
        plt.style.use('seaborn-v0_8')
        
        # 1. MAE Comparison Heatmap
        self._create_mae_heatmap(performance_data)
        
        # 2. Model Type Performance Comparison
        self._create_model_type_comparison(performance_data)
        
        # 3. Metric Distribution Boxplots
        self._create_metric_distributions(performance_data)
        
        # 4. Best Model Summary
        self._create_best_model_summary(performance_data)
    
    def _create_mae_heatmap(self, performance_data: Dict) -> None:
        """Create MAE comparison heatmap."""
        # Prepare data for heatmap
        categories = list(performance_data.keys())
        all_models = set()
        
        for models in performance_data.values():
            all_models.update(models.keys())
        
        all_models = sorted(list(all_models))
        
        # Create MAE matrix
        mae_matrix = np.full((len(categories), len(all_models)), np.nan)
        
        for i, category in enumerate(categories):
            for j, model in enumerate(all_models):
                if model in performance_data[category]:
                    mae_matrix[i, j] = performance_data[category][model].get('MAE', np.nan)
        
        # Create heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            mae_matrix,
            xticklabels=all_models,
            yticklabels=categories,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Mean Absolute Error (MAE)'}
        )
        plt.title('Model Performance Comparison - MAE Heatmap')
        plt.xlabel('Models')
        plt.ylabel('Expense Categories')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.reports_path / 'mae_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_type_comparison(self, performance_data: Dict) -> None:
        """Create model type performance comparison."""
        model_types = {'baseline': [], 'ml': [], 'dl': []}
        
        for category, models in performance_data.items():
            for model_name, metrics in models.items():
                mae = metrics.get('MAE', np.nan)
                if not np.isnan(mae):
                    if model_name.startswith('baseline_'):
                        model_types['baseline'].append(mae)
                    elif model_name.startswith('ml_'):
                        model_types['ml'].append(mae)
                    elif model_name.startswith('dl_'):
                        model_types['dl'].append(mae)
        
        # Create box plot
        plt.figure(figsize=(10, 6))
        
        data_to_plot = []
        labels = []
        
        for model_type, values in model_types.items():
            if values:
                data_to_plot.append(values)
                labels.append(f"{model_type.upper()} Models")
        
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.title('Model Type Performance Comparison')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.reports_path / 'model_type_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metric_distributions(self, performance_data: Dict) -> None:
        """Create metric distribution plots."""
        summary_df = self.generate_performance_summary_table(performance_data)
        
        if summary_df.empty:
            return
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['MAE', 'RMSE', 'MAPE', 'Directional_Accuracy']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            # Filter out NaN and infinite values
            data = summary_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            
            if not data.empty:
                ax.hist(data, bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel(metric)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {metric}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.reports_path / 'metric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_best_model_summary(self, performance_data: Dict) -> None:
        """Create best model summary visualization."""
        best_models = self.find_best_models(performance_data)
        
        if not best_models:
            return
        
        # Count model types
        model_type_counts = {'baseline': 0, 'ml': 0, 'dl': 0}
        
        for category, (model_name, _) in best_models.items():
            if model_name.startswith('baseline_'):
                model_type_counts['baseline'] += 1
            elif model_name.startswith('ml_'):
                model_type_counts['ml'] += 1
            elif model_name.startswith('dl_'):
                model_type_counts['dl'] += 1
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        
        labels = []
        sizes = []
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        for i, (model_type, count) in enumerate(model_type_counts.items()):
            if count > 0:
                labels.append(f"{model_type.upper()} Models ({count})")
                sizes.append(count)
        
        if sizes:
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors[:len(sizes)])
            plt.title('Best Performing Model Types by Category')
        
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(self.reports_path / 'best_model_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_report(self) -> str:
        """
        Generate a detailed evaluation report.
        
        Returns:
            Path to the generated report
        """
        performance_data = self.combine_all_performance()
        summary_df = self.generate_performance_summary_table(performance_data)
        best_models = self.find_best_models(performance_data)
        rankings = self.calculate_model_rankings(performance_data)
        
        # Generate visualizations
        self.create_performance_visualizations(performance_data)
        
        # Create HTML report
        report_html = self._generate_html_report(summary_df, best_models, rankings)
        
        # Save report
        report_path = self.reports_path / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        # Also save summary as JSON
        self._save_summary_json(performance_data, best_models, rankings)
        
        logger.info(f"Detailed evaluation report generated: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self, summary_df: pd.DataFrame, 
                             best_models: Dict, rankings: Dict) -> str:
        """Generate HTML evaluation report."""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>BudgetWise Forecasting - Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2E86AB; text-align: center; }
                h2 { color: #A23B72; border-bottom: 2px solid #A23B72; }
                h3 { color: #F18F01; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                th { background-color: #f2f2f2; font-weight: bold; }
                .best-model { background-color: #e8f5e8; font-weight: bold; }
                .metric-good { color: #28a745; }
                .metric-warning { color: #ffc107; }
                .metric-danger { color: #dc3545; }
                .summary-box { background-color: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; }
                .visualization { text-align: center; margin: 20px 0; }
                .footer { text-align: center; margin-top: 40px; color: #666; }
            </style>
        </head>
        <body>
            <h1>🏦 BudgetWise Forecasting - Model Evaluation Report</h1>
            <div class="summary-box">
                <p><strong>Report Generated:</strong> {timestamp}</p>
                <p><strong>Total Models Evaluated:</strong> {total_models}</p>
                <p><strong>Expense Categories:</strong> {total_categories}</p>
            </div>
            
            <h2>📊 Executive Summary</h2>
            {executive_summary}
            
            <h2>🏆 Best Performing Models by Category</h2>
            {best_models_table}
            
            <h2>📈 Complete Performance Summary</h2>
            {complete_performance_table}
            
            <h2>🎯 Model Rankings</h2>
            {rankings_table}
            
            <h2>📸 Performance Visualizations</h2>
            <div class="visualization">
                <h3>MAE Heatmap</h3>
                <img src="mae_heatmap.png" alt="MAE Heatmap" style="max-width: 100%;">
                
                <h3>Model Type Comparison</h3>
                <img src="model_type_comparison.png" alt="Model Type Comparison" style="max-width: 100%;">
                
                <h3>Best Model Distribution</h3>
                <img src="best_model_summary.png" alt="Best Model Summary" style="max-width: 100%;">
                
                <h3>Metric Distributions</h3>
                <img src="metric_distributions.png" alt="Metric Distributions" style="max-width: 100%;">
            </div>
            
            <div class="footer">
                <p>Generated by BudgetWise Forecasting System | {timestamp}</p>
            </div>
        </body>
        </html>
        """
        
        # Generate content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_models = len(summary_df) if not summary_df.empty else 0
        total_categories = len(best_models)
        
        executive_summary = self._generate_executive_summary(best_models, summary_df)
        best_models_table = self._generate_best_models_table(best_models)
        complete_performance_table = summary_df.to_html(classes='performance-table', escape=False) if not summary_df.empty else "<p>No performance data available.</p>"
        rankings_table = self._generate_rankings_table(rankings)
        
        return html_template.format(
            timestamp=timestamp,
            total_models=total_models,
            total_categories=total_categories,
            executive_summary=executive_summary,
            best_models_table=best_models_table,
            complete_performance_table=complete_performance_table,
            rankings_table=rankings_table
        )
    
    def _generate_executive_summary(self, best_models: Dict, summary_df: pd.DataFrame) -> str:
        """Generate executive summary."""
        if summary_df.empty:
            return "<p>No models have been trained yet.</p>"
        
        # Calculate overall statistics
        avg_mae = summary_df['MAE'].mean()
        best_overall_mae = summary_df['MAE'].min()
        
        # Count model types in best models
        model_type_counts = {'baseline': 0, 'ml': 0, 'dl': 0}
        for category, (model_name, _) in best_models.items():
            if model_name.startswith('baseline_'):
                model_type_counts['baseline'] += 1
            elif model_name.startswith('ml_'):
                model_type_counts['ml'] += 1
            elif model_name.startswith('dl_'):
                model_type_counts['dl'] += 1
        
        dominant_type = max(model_type_counts.items(), key=lambda x: x[1])
        
        summary = f"""
        <div class="summary-box">
            <h3>🎯 Key Insights:</h3>
            <ul>
                <li><strong>Average MAE across all models:</strong> {avg_mae:.2f}</li>
                <li><strong>Best overall MAE achieved:</strong> {best_overall_mae:.2f}</li>
                <li><strong>Dominant model type:</strong> {dominant_type[0].upper()} models ({dominant_type[1]} categories)</li>
                <li><strong>Total categories with best models:</strong> {len(best_models)}</li>
            </ul>
        </div>
        """
        
        return summary
    
    def _generate_best_models_table(self, best_models: Dict) -> str:
        """Generate best models table."""
        if not best_models:
            return "<p>No best models identified.</p>"
        
        table_html = """
        <table>
            <tr>
                <th>Category</th>
                <th>Best Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>MAPE (%)</th>
                <th>R²</th>
                <th>Directional Accuracy (%)</th>
            </tr>
        """
        
        for category, (model_name, metrics) in best_models.items():
            table_html += f"""
            <tr class="best-model">
                <td>{category}</td>
                <td>{model_name}</td>
                <td>{metrics.get('MAE', 'N/A'):.2f}</td>
                <td>{metrics.get('RMSE', 'N/A'):.2f}</td>
                <td>{metrics.get('MAPE', 'N/A'):.1f}</td>
                <td>{metrics.get('R2', 'N/A'):.3f}</td>
                <td>{metrics.get('Directional_Accuracy', 'N/A'):.1f}</td>
            </tr>
            """
        
        table_html += "</table>"
        return table_html
    
    def _generate_rankings_table(self, rankings: Dict) -> str:
        """Generate rankings table."""
        if not rankings:
            return "<p>No rankings available.</p>"
        
        # Convert rankings to a more readable format
        all_models = set()
        for category_rankings in rankings.values():
            all_models.update(category_rankings.keys())
        
        table_html = """
        <table>
            <tr>
                <th>Model</th>
        """
        
        # Add category headers
        for category in rankings.keys():
            table_html += f"<th>{category}</th>"
        
        table_html += "</tr>"
        
        # Add model rows
        for model in sorted(all_models):
            table_html += f"<tr><td>{model}</td>"
            
            for category in rankings.keys():
                rank = rankings[category].get(model, '-')
                if rank == 1:
                    table_html += f'<td class="best-model">{rank}</td>'
                elif rank <= 3:
                    table_html += f'<td class="metric-good">{rank}</td>'
                else:
                    table_html += f'<td>{rank}</td>'
            
            table_html += "</tr>"
        
        table_html += "</table>"
        return table_html
    
    def _save_summary_json(self, performance_data: Dict, best_models: Dict, rankings: Dict) -> None:
        """Save summary data as JSON."""
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'performance_data': performance_data,
            'best_models': {k: {'model': v[0], 'metrics': v[1]} for k, v in best_models.items()},
            'rankings': rankings
        }
        
        json_path = self.reports_path / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        logger.info(f"Summary JSON saved: {json_path}")
    
    def print_quick_summary(self) -> None:
        """Print a quick summary to console."""
        performance_data = self.combine_all_performance()
        best_models = self.find_best_models(performance_data)
        
        print("\n" + "="*70)
        print("🏦 BUDGETWISE FORECASTING - QUICK EVALUATION SUMMARY")
        print("="*70)
        
        if not best_models:
            print("❌ No trained models found. Please train models first.")
            return
        
        print(f"\n📊 Best Models by Category:")
        print("-" * 50)
        
        for category, (model_name, metrics) in best_models.items():
            mae = metrics.get('MAE', 0)
            print(f"• {category:<20} | {model_name:<25} | MAE: {mae:.2f}")
        
        # Model type summary
        model_type_counts = {'Baseline': 0, 'ML': 0, 'DL': 0}
        for category, (model_name, _) in best_models.items():
            if model_name.startswith('baseline_'):
                model_type_counts['Baseline'] += 1
            elif model_name.startswith('ml_'):
                model_type_counts['ML'] += 1
            elif model_name.startswith('dl_'):
                model_type_counts['DL'] += 1
        
        print(f"\n🏆 Model Type Performance:")
        print("-" * 30)
        for model_type, count in model_type_counts.items():
            print(f"• {model_type:<10}: {count} best models")
        
        print("\n" + "="*70)


def main():
    """Main function to run model evaluation."""
    evaluator = ModelEvaluator()
    
    # Print quick summary
    evaluator.print_quick_summary()
    
    # Generate detailed report
    report_path = evaluator.generate_detailed_report()
    print(f"\n📋 Detailed report generated: {report_path}")


if __name__ == "__main__":
    main()