"""
Machine Learning Models for BudgetWise Forecasting System
Implements Random Forest, XGBoost, and LightGBM models.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

class MLModels:
    """
    Machine Learning models for expense forecasting.
    Implements Random Forest, XGBoost, and LightGBM with hyperparameter tuning.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.features_path = Path(self.config['data']['features_data_path'])
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}
        
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
                'ml_models': {
                    'random_forest': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5],
                        'random_state': 42
                    },
                    'xgboost': {
                        'n_estimators': [100, 200],
                        'max_depth': [6, 10],
                        'learning_rate': [0.01, 0.1],
                        'random_state': 42
                    },
                    'lightgbm': {
                        'n_estimators': [100, 200],
                        'max_depth': [6, 10],
                        'learning_rate': [0.01, 0.1],
                        'random_state': 42
                    }
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
    
    def prepare_ml_data(self, train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, 
                       target_column: str,
                       scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
        """
        Prepare data for ML models with optional scaling.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            target_column: Name of the target column
            scale_features: Whether to scale features
            
        Returns:
            Tuple of X_train, X_val, X_test, y_train, y_val, y_test, scaler
        """
        # Select feature columns (exclude date and target)
        feature_cols = [col for col in train_df.columns if col not in ['date', target_column]]
        
        # Prepare data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_column].fillna(0).values
        
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df[target_column].fillna(0).values
        
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_column].fillna(0).values
        
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        else:
            X_train = X_train.values
            X_val = X_val.values
            X_test = X_test.values
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray, 
                          category: str, tune_hyperparameters: bool = True) -> RandomForestRegressor:
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            category: Category name for the model
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained RandomForestRegressor model
        """
        logger.info(f"Training Random Forest for {category}...")
        
        if tune_hyperparameters:
            # Define parameter grid
            config = self.config['models']['ml_models']['random_forest']
            param_grid = {
                'n_estimators': config.get('n_estimators', [100, 200]),
                'max_depth': config.get('max_depth', [10, 20]),
                'min_samples_split': config.get('min_samples_split', [2, 5])
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Combine train and validation for cross-validation
            X_combined = np.concatenate([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            
            rf = RandomForestRegressor(random_state=config.get('random_state', 42))
            
            # Grid search
            grid_search = GridSearchCV(
                rf, param_grid, cv=tscv, scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_combined, y_combined)
            best_model = grid_search.best_estimator_
            
            logger.info(f"Best Random Forest parameters for {category}: {grid_search.best_params_}")
            
        else:
            # Use default parameters
            config = self.config['models']['ml_models']['random_forest']
            best_model = RandomForestRegressor(
                n_estimators=config.get('n_estimators', [100])[0],
                max_depth=config.get('max_depth', [10])[0],
                min_samples_split=config.get('min_samples_split', [2])[0],
                random_state=config.get('random_state', 42)
            )
            best_model.fit(X_train, y_train)
        
        return best_model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray, 
                     category: str, tune_hyperparameters: bool = True) -> Any:
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            category: Category name for the model
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained XGBoost model
        """
        if not XGBOOST_AVAILABLE:
            logger.warning(f"XGBoost not available for {category}")
            return None
            
        logger.info(f"Training XGBoost for {category}...")
        
        if tune_hyperparameters:
            # Define parameter grid
            config = self.config['models']['ml_models']['xgboost']
            param_grid = {
                'n_estimators': config.get('n_estimators', [100, 200]),
                'max_depth': config.get('max_depth', [6, 10]),
                'learning_rate': config.get('learning_rate', [0.01, 0.1])
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Combine train and validation for cross-validation
            X_combined = np.concatenate([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            
            xgb_model = xgb.XGBRegressor(
                random_state=config.get('random_state', 42),
                objective='reg:squarederror'
            )
            
            # Grid search
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=tscv, scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_combined, y_combined)
            best_model = grid_search.best_estimator_
            
            logger.info(f"Best XGBoost parameters for {category}: {grid_search.best_params_}")
            
        else:
            # Use default parameters
            config = self.config['models']['ml_models']['xgboost']
            best_model = xgb.XGBRegressor(
                n_estimators=config.get('n_estimators', [100])[0],
                max_depth=config.get('max_depth', [6])[0],
                learning_rate=config.get('learning_rate', [0.01])[0],
                random_state=config.get('random_state', 42),
                objective='reg:squarederror'
            )
            best_model.fit(X_train, y_train)
        
        return best_model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray, 
                      category: str, tune_hyperparameters: bool = True) -> Any:
        """
        Train LightGBM model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            category: Category name for the model
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained LightGBM model
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning(f"LightGBM not available for {category}")
            return None
            
        logger.info(f"Training LightGBM for {category}...")
        
        if tune_hyperparameters:
            # Define parameter grid
            config = self.config['models']['ml_models']['lightgbm']
            param_grid = {
                'n_estimators': config.get('n_estimators', [100, 200]),
                'max_depth': config.get('max_depth', [6, 10]),
                'learning_rate': config.get('learning_rate', [0.01, 0.1])
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Combine train and validation for cross-validation
            X_combined = np.concatenate([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            
            lgb_model = lgb.LGBMRegressor(
                random_state=config.get('random_state', 42),
                objective='regression',
                verbose=-1
            )
            
            # Grid search
            grid_search = GridSearchCV(
                lgb_model, param_grid, cv=tscv, scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_combined, y_combined)
            best_model = grid_search.best_estimator_
            
            logger.info(f"Best LightGBM parameters for {category}: {grid_search.best_params_}")
            
        else:
            # Use default parameters
            config = self.config['models']['ml_models']['lightgbm']
            best_model = lgb.LGBMRegressor(
                n_estimators=config.get('n_estimators', [100])[0],
                max_depth=config.get('max_depth', [6])[0],
                learning_rate=config.get('learning_rate', [0.01])[0],
                random_state=config.get('random_state', 42),
                objective='regression',
                verbose=-1
            )
            best_model.fit(X_train, y_train)
        
        return best_model
    
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
        
        # R² Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
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
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
    
    def get_feature_importance(self, model: Any, feature_names: List[str], 
                             top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance and return top_n
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            return dict(sorted_features[:top_n])
        
        return {}
    
    def train_all_ml_models(self, tune_hyperparameters: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Train all ML models for all expense categories.
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing all trained models and their performance
        """
        logger.info("Training all ML models...")
        
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
        
        # Get feature column names for importance analysis
        feature_cols = [col for col in train_data.columns if col not in ['date'] + category_cols]
        
        models_performance = {}
        
        for category in category_cols[:3]:  # Limit to top 3 categories for demo
            logger.info(f"Training ML models for category: {category}")
            
            category_models = {}
            category_performance = {}
            category_scalers = {}
            category_feature_importance = {}
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test, scaler = self.prepare_ml_data(
                train_data, val_data, test_data, category, scale_features=False
            )
            
            # 1. Random Forest
            try:
                rf_model = self.train_random_forest(
                    X_train, y_train, X_val, y_val, category, tune_hyperparameters
                )
                
                # Predict and evaluate
                y_pred_test = rf_model.predict(X_test)
                rf_performance = self.evaluate_model(y_test, y_pred_test)
                
                # Feature importance
                rf_importance = self.get_feature_importance(rf_model, feature_cols)
                
                category_models['random_forest'] = rf_model
                category_performance['random_forest'] = rf_performance
                category_feature_importance['random_forest'] = rf_importance
                
            except Exception as e:
                logger.warning(f"Random Forest failed for {category}: {str(e)}")
            
            # 2. XGBoost
            if XGBOOST_AVAILABLE:
                try:
                    xgb_model = self.train_xgboost(
                        X_train, y_train, X_val, y_val, category, tune_hyperparameters
                    )
                    
                    if xgb_model is not None:
                        # Predict and evaluate
                        y_pred_test = xgb_model.predict(X_test)
                        xgb_performance = self.evaluate_model(y_test, y_pred_test)
                        
                        # Feature importance
                        xgb_importance = self.get_feature_importance(xgb_model, feature_cols)
                        
                        category_models['xgboost'] = xgb_model
                        category_performance['xgboost'] = xgb_performance
                        category_feature_importance['xgboost'] = xgb_importance
                
                except Exception as e:
                    logger.warning(f"XGBoost failed for {category}: {str(e)}")
            
            # 3. LightGBM
            if LIGHTGBM_AVAILABLE:
                try:
                    lgb_model = self.train_lightgbm(
                        X_train, y_train, X_val, y_val, category, tune_hyperparameters
                    )
                    
                    if lgb_model is not None:
                        # Predict and evaluate
                        y_pred_test = lgb_model.predict(X_test)
                        lgb_performance = self.evaluate_model(y_test, y_pred_test)
                        
                        # Feature importance
                        lgb_importance = self.get_feature_importance(lgb_model, feature_cols)
                        
                        category_models['lightgbm'] = lgb_model
                        category_performance['lightgbm'] = lgb_performance
                        category_feature_importance['lightgbm'] = lgb_importance
                
                except Exception as e:
                    logger.warning(f"LightGBM failed for {category}: {str(e)}")
            
            models_performance[category] = {
                'models': category_models,
                'performance': category_performance,
                'scalers': category_scalers,
                'feature_importance': category_feature_importance
            }
        
        # Save models
        self.save_ml_models(models_performance)
        
        # Print performance summary
        self._print_performance_summary(models_performance)
        
        return models_performance
    
    def save_ml_models(self, models_performance: Dict[str, Dict[str, Any]]) -> None:
        """
        Save trained ML models and associated artifacts.
        
        Args:
            models_performance: Dictionary containing models and performance metrics
        """
        logger.info("Saving ML models...")
        
        # Save each model and scaler
        for category, category_data in models_performance.items():
            models = category_data['models']
            scalers = category_data.get('scalers', {})
            
            for model_name, model in models.items():
                model_filename = f"ml_{model_name}_{category}.pkl"
                joblib.dump(model, self.models_path / model_filename)
                
                # Save scaler if exists
                if model_name in scalers:
                    scaler_filename = f"scaler_{model_name}_{category}.pkl"
                    joblib.dump(scalers[model_name], self.models_path / scaler_filename)
        
        # Save performance metrics and feature importance
        performance_data = {}
        feature_importance_data = {}
        
        for category, category_data in models_performance.items():
            performance_data[category] = category_data['performance']
            feature_importance_data[category] = category_data.get('feature_importance', {})
        
        joblib.dump(performance_data, self.models_path / "ml_performance.pkl")
        joblib.dump(feature_importance_data, self.models_path / "ml_feature_importance.pkl")
        
        logger.info("ML models saved successfully!")
    
    def _print_performance_summary(self, models_performance: Dict[str, Dict[str, Any]]) -> None:
        """Print performance summary of all ML models."""
        
        print("\n" + "="*80)
        print("MACHINE LEARNING MODELS PERFORMANCE SUMMARY")
        print("="*80)
        
        for category, category_data in models_performance.items():
            print(f"\n📊 Category: {category}")
            print("-" * 70)
            
            performance = category_data['performance']
            feature_importance = category_data.get('feature_importance', {})
            
            if not performance:
                print("   ⚠️  No models trained successfully")
                continue
            
            # Create performance table
            print(f"{'Model':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<8} {'Dir.Acc':<10}")
            print("-" * 70)
            
            for model_name, metrics in performance.items():
                mae = metrics.get('MAE', 0)
                rmse = metrics.get('RMSE', 0)
                mape = metrics.get('MAPE', 0)
                r2 = metrics.get('R2', 0)
                dir_acc = metrics.get('Directional_Accuracy', 0)
                
                print(f"{model_name:<15} {mae:<10.2f} {rmse:<10.2f} {mape:<10.1f}% {r2:<8.3f} {dir_acc:<10.1f}%")
            
            # Show top 3 important features for the best performing model
            if feature_importance:
                best_model = min(performance.keys(), key=lambda x: performance[x].get('MAE', float('inf')))
                if best_model in feature_importance:
                    print(f"\n🔝 Top 3 Features for {best_model}:")
                    for i, (feature, importance) in enumerate(list(feature_importance[best_model].items())[:3]):
                        print(f"   {i+1}. {feature}: {importance:.4f}")
        
        print("="*80)


def main():
    """Main function to train ML models."""
    ml_models = MLModels()
    ml_models.train_all_ml_models(tune_hyperparameters=True)


if __name__ == "__main__":
    main()