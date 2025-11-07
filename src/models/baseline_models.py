"""
Baseline Models for BudgetWise Forecasting System
Implements Linear Regression, ARIMA, and Prophet models for comparison.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("Prophet not available. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

class BaselineModels:
    """
    Baseline forecasting models for comparison with advanced models.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.features_path = Path(self.config['data']['features_data_path'])
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        self.models = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'data': {
                'features_data_path': 'data/features/'
            },
            'models': {
                'baseline': {
                    'linear_regression': {'fit_intercept': True},
                    'arima': {'max_p': 5, 'max_d': 2, 'max_q': 5, 'seasonal': True},
                    'prophet': {'daily_seasonality': True, 'weekly_seasonality': True, 'yearly_seasonality': True}
                }
            }
        }
    
    def load_engineered_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load engineered features from CSV files."""
        try:
            train_data = pd.read_csv(self.features_path / "train_features.csv")
            val_data = pd.read_csv(self.features_path / "val_features.csv")
            test_data = pd.read_csv(self.features_path / "test_features.csv")
            
            # Convert date columns
            for df in [train_data, val_data, test_data]:
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info("Engineered features loaded successfully")
            return train_data, val_data, test_data
            
        except FileNotFoundError:
            logger.error("Engineered features not found. Please run feature engineering first.")
            raise
    
    def prepare_linear_regression_data(self, train_df: pd.DataFrame, 
                                     val_df: pd.DataFrame, 
                                     test_df: pd.DataFrame, 
                                     target_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for linear regression model.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            target_column: Name of the target column
            
        Returns:
            Tuple of X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Select feature columns (exclude date and target)
        feature_cols = [col for col in train_df.columns if col not in ['date', target_column]]
        
        # Prepare training data
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df[target_column].fillna(0).values
        
        # Prepare validation data
        X_val = val_df[feature_cols].fillna(0).values
        y_val = val_df[target_column].fillna(0).values
        
        # Prepare test data
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df[target_column].fillna(0).values
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                              category: str) -> LinearRegression:
        """
        Train linear regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            category: Category name for the model
            
        Returns:
            Trained LinearRegression model
        """
        logger.info(f"Training Linear Regression for {category}...")
        
        config = self.config['models']['baseline']['linear_regression']
        model = LinearRegression(**config)
        model.fit(X_train, y_train)
        
        return model
    
    def train_arima_model(self, data: pd.Series, category: str) -> Any:
        """
        Train ARIMA model with automatic order selection.
        
        Args:
            data: Time series data
            category: Category name for the model
            
        Returns:
            Fitted ARIMA model
        """
        logger.info(f"Training ARIMA for {category}...")
        
        # Remove any NaN or infinite values
        data_clean = data.dropna().replace([np.inf, -np.inf], 0)
        
        if len(data_clean) < 10:
            logger.warning(f"Insufficient data for ARIMA model for {category}")
            return None
        
        try:
            # Simple ARIMA model - can be enhanced with auto_arima
            model = ARIMA(data_clean, order=(2, 1, 2))
            fitted_model = model.fit()
            return fitted_model
        
        except Exception as e:
            logger.warning(f"ARIMA training failed for {category}: {str(e)}")
            return None
    
    def train_prophet_model(self, data: pd.DataFrame, category: str) -> Any:
        """
        Train Prophet model.
        
        Args:
            data: Dataframe with 'date' and target columns
            category: Category name for the model
            
        Returns:
            Fitted Prophet model
        """
        if not PROPHET_AVAILABLE:
            logger.warning(f"Prophet not available for {category}")
            return None
            
        logger.info(f"Training Prophet for {category}...")
        
        # Prepare data for Prophet
        prophet_data = data.rename(columns={'date': 'ds', category: 'y'})
        prophet_data = prophet_data[['ds', 'y']].dropna()
        
        if len(prophet_data) < 10:
            logger.warning(f"Insufficient data for Prophet model for {category}")
            return None
        
        try:
            config = self.config['models']['baseline']['prophet']
            model = Prophet(**config)
            model.fit(prophet_data)
            return model
        
        except Exception as e:
            logger.warning(f"Prophet training failed for {category}: {str(e)}")
            return None
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.inf
        
        # Directional Accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
    
    def train_all_baseline_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train all baseline models for all expense categories.
        
        Returns:
            Dictionary containing all trained models and their performance
        """
        logger.info("Training all baseline models...")
        
        # Load data
        train_data, val_data, test_data = self.load_engineered_data()
        
        # Get category columns (expense categories)
        category_cols = [col for col in train_data.columns 
                        if col not in ['date'] and not col.startswith(('year', 'month', 'day', 'is_', 
                                                                      'dayofweek', 'dayofyear', 'week', 
                                                                      'quarter', '_lag_', '_rolling_', 
                                                                      '_seasonal_', '_ratio', '_cv_', 
                                                                      '_pct_change', '_volatility', 
                                                                      'total_daily_expense', '_sin', '_cos'))]
        
        models_performance = {}
        
        for category in category_cols[:5]:  # Limit to top 5 categories for demo
            logger.info(f"Training models for category: {category}")
            
            category_models = {}
            category_performance = {}
            
            # 1. Linear Regression
            try:
                X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_linear_regression_data(
                    train_data, val_data, test_data, category
                )
                
                lr_model = self.train_linear_regression(X_train, y_train, category)
                
                # Predict and evaluate
                y_pred_val = lr_model.predict(X_val)
                lr_performance = self.evaluate_model(y_val, y_pred_val)
                
                category_models['linear_regression'] = lr_model
                category_performance['linear_regression'] = lr_performance
                
            except Exception as e:
                logger.warning(f"Linear Regression failed for {category}: {str(e)}")
            
            # 2. ARIMA
            try:
                # Combine train and validation for ARIMA
                combined_data = pd.concat([train_data, val_data])
                arima_series = combined_data.set_index('date')[category]
                
                arima_model = self.train_arima_model(arima_series, category)
                
                if arima_model is not None:
                    # Forecast
                    forecast_steps = len(test_data)
                    forecast = arima_model.forecast(steps=forecast_steps)
                    
                    # Evaluate
                    y_test_arima = test_data[category].fillna(0).values
                    arima_performance = self.evaluate_model(y_test_arima, forecast)
                    
                    category_models['arima'] = arima_model
                    category_performance['arima'] = arima_performance
                
            except Exception as e:
                logger.warning(f"ARIMA failed for {category}: {str(e)}")
            
            # 3. Prophet
            if PROPHET_AVAILABLE:
                try:
                    # Combine train and validation for Prophet
                    combined_data = pd.concat([train_data, val_data])
                    prophet_data = combined_data[['date', category]]
                    
                    prophet_model = self.train_prophet_model(prophet_data, category)
                    
                    if prophet_model is not None:
                        # Create future dataframe
                        future = prophet_model.make_future_dataframe(periods=len(test_data))
                        forecast = prophet_model.predict(future)
                        
                        # Get predictions for test period
                        test_predictions = forecast['yhat'].tail(len(test_data)).values
                        
                        # Evaluate
                        y_test_prophet = test_data[category].fillna(0).values
                        prophet_performance = self.evaluate_model(y_test_prophet, test_predictions)
                        
                        category_models['prophet'] = prophet_model
                        category_performance['prophet'] = prophet_performance
                
                except Exception as e:
                    logger.warning(f"Prophet failed for {category}: {str(e)}")
            
            models_performance[category] = {
                'models': category_models,
                'performance': category_performance
            }
        
        # Save models
        self.save_baseline_models(models_performance)
        
        # Print performance summary
        self._print_performance_summary(models_performance)
        
        return models_performance
    
    def save_baseline_models(self, models_performance: Dict[str, Dict[str, Any]]) -> None:
        """
        Save trained baseline models.
        
        Args:
            models_performance: Dictionary containing models and performance metrics
        """
        logger.info("Saving baseline models...")
        
        # Save each model
        for category, category_data in models_performance.items():
            models = category_data['models']
            
            for model_name, model in models.items():
                model_filename = f"baseline_{model_name}_{category}.pkl"
                joblib.dump(model, self.models_path / model_filename)
        
        # Save performance metrics
        performance_data = {}
        for category, category_data in models_performance.items():
            performance_data[category] = category_data['performance']
        
        joblib.dump(performance_data, self.models_path / "baseline_performance.pkl")
        
        logger.info("Baseline models saved successfully!")
    
    def _print_performance_summary(self, models_performance: Dict[str, Dict[str, Any]]) -> None:
        """Print performance summary of all baseline models."""
        
        print("\n" + "="*70)
        print("BASELINE MODELS PERFORMANCE SUMMARY")
        print("="*70)
        
        for category, category_data in models_performance.items():
            print(f"\n📊 Category: {category}")
            print("-" * 50)
            
            performance = category_data['performance']
            
            if not performance:
                print("   ⚠️  No models trained successfully")
                continue
            
            # Create performance table
            print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'Dir.Acc':<10}")
            print("-" * 60)
            
            for model_name, metrics in performance.items():
                mae = metrics.get('MAE', 0)
                rmse = metrics.get('RMSE', 0)
                mape = metrics.get('MAPE', 0)
                dir_acc = metrics.get('Directional_Accuracy', 0)
                
                print(f"{model_name:<20} {mae:<10.2f} {rmse:<10.2f} {mape:<10.2f}% {dir_acc:<10.1f}%")
        
        print("="*70)


def main():
    """Main function to train baseline models."""
    baseline = BaselineModels()
    baseline.train_all_baseline_models()


if __name__ == "__main__":
    main()