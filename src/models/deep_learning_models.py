"""
Deep Learning Models for BudgetWise Forecasting System
Implements LSTM, GRU, CNN, and Transformer models for time series forecasting.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, 
                                        Conv1D, MaxPooling1D, Flatten,
                                        MultiHeadAttention, LayerNormalization,
                                        Input, GlobalAveragePooling1D)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

class DeepLearningModels:
    """
    Deep Learning models for expense forecasting.
    Implements LSTM, GRU, CNN, and Transformer models using TensorFlow/Keras.
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
                'deep_learning': {
                    'sequence_length': 30,
                    'lstm': {
                        'units': [64, 32],
                        'dropout': 0.2,
                        'epochs': 100,
                        'batch_size': 32,
                        'learning_rate': 0.001
                    },
                    'gru': {
                        'units': [64, 32],
                        'dropout': 0.2,
                        'epochs': 100,
                        'batch_size': 32,
                        'learning_rate': 0.001
                    },
                    'cnn': {
                        'filters': [64, 32],
                        'kernel_size': 3,
                        'dropout': 0.2,
                        'epochs': 100,
                        'batch_size': 32,
                        'learning_rate': 0.001
                    },
                    'transformer': {
                        'embed_dim': 64,
                        'num_heads': 4,
                        'ff_dim': 128,
                        'dropout': 0.1,
                        'epochs': 100,
                        'batch_size': 32,
                        'learning_rate': 0.001
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
    
    def create_sequences(self, data: np.ndarray, sequence_length: int, 
                        target_col: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            data: Input data array
            sequence_length: Length of input sequences
            target_col: Index of target column
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(data)):
            # Input sequence (all features)
            seq = data[i-sequence_length:i]
            sequences.append(seq)
            
            # Target (only the target column)
            target = data[i, target_col]
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def prepare_dl_data(self, train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, 
                       target_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
        """
        Prepare data for deep learning models.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            target_column: Name of the target column
            
        Returns:
            Tuple of X_train, X_val, X_test, y_train, y_val, y_test, scaler, n_features
        """
        # Get sequence length from config
        sequence_length = self.config['models']['deep_learning']['sequence_length']
        
        # Select feature columns (exclude date)
        feature_cols = [col for col in train_df.columns if col not in ['date']]
        
        # Combine all data for proper scaling
        all_data = pd.concat([train_df[feature_cols], val_df[feature_cols], test_df[feature_cols]])
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(all_data.fillna(0))
        
        # Split back into train, val, test
        train_size = len(train_df)
        val_size = len(val_df)
        
        train_scaled = scaled_data[:train_size]
        val_scaled = scaled_data[train_size:train_size + val_size]
        test_scaled = scaled_data[train_size + val_size:]
        
        # Find target column index
        target_col_idx = feature_cols.index(target_column)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled, sequence_length, target_col_idx)
        X_val, y_val = self.create_sequences(val_scaled, sequence_length, target_col_idx)
        X_test, y_test = self.create_sequences(test_scaled, sequence_length, target_col_idx)
        
        n_features = len(feature_cols)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, n_features
    
    def build_lstm_model(self, input_shape: Tuple[int, int], config: Dict) -> Any:
        """
        Build LSTM model.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            config: Model configuration
            
        Returns:
            Compiled LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available for LSTM")
            return None
        
        model = Sequential()
        
        units = config.get('units', [64, 32])
        dropout = config.get('dropout', 0.2)
        
        # First LSTM layer
        model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for unit in units[1:-1]:
            model.add(LSTM(unit, return_sequences=True))
            model.add(Dropout(dropout))
        
        # Last LSTM layer
        if len(units) > 1:
            model.add(LSTM(units[-1]))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def build_gru_model(self, input_shape: Tuple[int, int], config: Dict) -> Any:
        """
        Build GRU model.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            config: Model configuration
            
        Returns:
            Compiled GRU model
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available for GRU")
            return None
        
        model = Sequential()
        
        units = config.get('units', [64, 32])
        dropout = config.get('dropout', 0.2)
        
        # First GRU layer
        model.add(GRU(units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # Additional GRU layers
        for unit in units[1:-1]:
            model.add(GRU(unit, return_sequences=True))
            model.add(Dropout(dropout))
        
        # Last GRU layer
        if len(units) > 1:
            model.add(GRU(units[-1]))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def build_cnn_model(self, input_shape: Tuple[int, int], config: Dict) -> Any:
        """
        Build CNN model for time series.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            config: Model configuration
            
        Returns:
            Compiled CNN model
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available for CNN")
            return None
        
        model = Sequential()
        
        filters = config.get('filters', [64, 32])
        kernel_size = config.get('kernel_size', 3)
        dropout = config.get('dropout', 0.2)
        
        # First Conv layer
        model.add(Conv1D(filters[0], kernel_size, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(2))
        model.add(Dropout(dropout))
        
        # Additional Conv layers
        for filter_size in filters[1:]:
            model.add(Conv1D(filter_size, kernel_size, activation='relu'))
            model.add(MaxPooling1D(2))
            model.add(Dropout(dropout))
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def build_transformer_model(self, input_shape: Tuple[int, int], config: Dict) -> Any:
        """
        Build simple Transformer model for time series.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            config: Model configuration
            
        Returns:
            Compiled Transformer model
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available for Transformer")
            return None
        
        embed_dim = config.get('embed_dim', 64)
        num_heads = config.get('num_heads', 4)
        ff_dim = config.get('ff_dim', 128)
        dropout = config.get('dropout', 0.1)
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
        attention = Dropout(dropout)(attention)
        
        # Add & Norm
        attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
        
        # Feed forward
        ffn = Dense(ff_dim, activation='relu')(attention)
        ffn = Dense(embed_dim)(ffn)
        ffn = Dropout(dropout)(ffn)
        
        # Add & Norm
        ffn = LayerNormalization(epsilon=1e-6)(attention + ffn)
        
        # Global average pooling
        pooling = GlobalAveragePooling1D()(ffn)
        
        # Output layer
        outputs = Dense(1)(pooling)
        
        # Create and compile model
        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_dl_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, config: Dict) -> Any:
        """
        Train deep learning model.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            config: Training configuration
            
        Returns:
            Trained model
        """
        if not TENSORFLOW_AVAILABLE or model is None:
            return None
        
        # Callbacks
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return model
    
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
    
    def train_all_dl_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train all deep learning models for all expense categories.
        
        Returns:
            Dictionary containing all trained models and their performance
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot train deep learning models.")
            return {}
        
        logger.info("Training all deep learning models...")
        
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
        
        for category in category_cols[:2]:  # Limit to top 2 categories for demo
            logger.info(f"Training deep learning models for category: {category}")
            
            category_models = {}
            category_performance = {}
            category_scalers = {}
            
            try:
                # Prepare data
                X_train, X_val, X_test, y_train, y_val, y_test, scaler, n_features = self.prepare_dl_data(
                    train_data, val_data, test_data, category
                )
                
                if len(X_train) == 0:
                    logger.warning(f"No training sequences created for {category}")
                    continue
                
                input_shape = (X_train.shape[1], X_train.shape[2])
                category_scalers['scaler'] = scaler
                
                # 1. LSTM
                try:
                    lstm_config = self.config['models']['deep_learning']['lstm']
                    lstm_model = self.build_lstm_model(input_shape, lstm_config)
                    
                    if lstm_model is not None:
                        lstm_model = self.train_dl_model(
                            lstm_model, X_train, y_train, X_val, y_val, lstm_config
                        )
                        
                        # Predict and evaluate
                        y_pred_test = lstm_model.predict(X_test, verbose=0).flatten()
                        lstm_performance = self.evaluate_model(y_test, y_pred_test)
                        
                        category_models['lstm'] = lstm_model
                        category_performance['lstm'] = lstm_performance
                
                except Exception as e:
                    logger.warning(f"LSTM training failed for {category}: {str(e)}")
                
                # 2. GRU
                try:
                    gru_config = self.config['models']['deep_learning']['gru']
                    gru_model = self.build_gru_model(input_shape, gru_config)
                    
                    if gru_model is not None:
                        gru_model = self.train_dl_model(
                            gru_model, X_train, y_train, X_val, y_val, gru_config
                        )
                        
                        # Predict and evaluate
                        y_pred_test = gru_model.predict(X_test, verbose=0).flatten()
                        gru_performance = self.evaluate_model(y_test, y_pred_test)
                        
                        category_models['gru'] = gru_model
                        category_performance['gru'] = gru_performance
                
                except Exception as e:
                    logger.warning(f"GRU training failed for {category}: {str(e)}")
                
                # 3. CNN
                try:
                    cnn_config = self.config['models']['deep_learning']['cnn']
                    cnn_model = self.build_cnn_model(input_shape, cnn_config)
                    
                    if cnn_model is not None:
                        cnn_model = self.train_dl_model(
                            cnn_model, X_train, y_train, X_val, y_val, cnn_config
                        )
                        
                        # Predict and evaluate
                        y_pred_test = cnn_model.predict(X_test, verbose=0).flatten()
                        cnn_performance = self.evaluate_model(y_test, y_pred_test)
                        
                        category_models['cnn'] = cnn_model
                        category_performance['cnn'] = cnn_performance
                
                except Exception as e:
                    logger.warning(f"CNN training failed for {category}: {str(e)}")
                
                # 4. Transformer (simplified)
                try:
                    transformer_config = self.config['models']['deep_learning']['transformer']
                    transformer_model = self.build_transformer_model(input_shape, transformer_config)
                    
                    if transformer_model is not None:
                        transformer_model = self.train_dl_model(
                            transformer_model, X_train, y_train, X_val, y_val, transformer_config
                        )
                        
                        # Predict and evaluate
                        y_pred_test = transformer_model.predict(X_test, verbose=0).flatten()
                        transformer_performance = self.evaluate_model(y_test, y_pred_test)
                        
                        category_models['transformer'] = transformer_model
                        category_performance['transformer'] = transformer_performance
                
                except Exception as e:
                    logger.warning(f"Transformer training failed for {category}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Data preparation failed for {category}: {str(e)}")
                continue
            
            models_performance[category] = {
                'models': category_models,
                'performance': category_performance,
                'scalers': category_scalers
            }
        
        # Save models
        self.save_dl_models(models_performance)
        
        # Print performance summary
        self._print_performance_summary(models_performance)
        
        return models_performance
    
    def save_dl_models(self, models_performance: Dict[str, Dict[str, Any]]) -> None:
        """
        Save trained deep learning models and associated artifacts.
        
        Args:
            models_performance: Dictionary containing models and performance metrics
        """
        if not TENSORFLOW_AVAILABLE:
            return
        
        logger.info("Saving deep learning models...")
        
        # Save each model and scaler
        for category, category_data in models_performance.items():
            models = category_data['models']
            scalers = category_data.get('scalers', {})
            
            for model_name, model in models.items():
                model_filename = f"dl_{model_name}_{category}.keras"
                model.save(self.models_path / model_filename)
            
            # Save scalers
            for scaler_name, scaler in scalers.items():
                scaler_filename = f"dl_scaler_{category}.pkl"
                joblib.dump(scaler, self.models_path / scaler_filename)
        
        # Save performance metrics
        performance_data = {}
        for category, category_data in models_performance.items():
            performance_data[category] = category_data['performance']
        
        joblib.dump(performance_data, self.models_path / "dl_performance.pkl")
        
        logger.info("Deep learning models saved successfully!")
    
    def _print_performance_summary(self, models_performance: Dict[str, Dict[str, Any]]) -> None:
        """Print performance summary of all deep learning models."""
        
        print("\n" + "="*80)
        print("DEEP LEARNING MODELS PERFORMANCE SUMMARY")
        print("="*80)
        
        for category, category_data in models_performance.items():
            print(f"\n🧠 Category: {category}")
            print("-" * 70)
            
            performance = category_data['performance']
            
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
        
        print("="*80)


def main():
    """Main function to train deep learning models."""
    dl_models = DeepLearningModels()
    dl_models.train_all_dl_models()


if __name__ == "__main__":
    main()