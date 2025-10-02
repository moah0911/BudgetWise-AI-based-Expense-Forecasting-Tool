"""
Feature Engineering Module for BudgetWise Forecasting System
Creates advanced time-series features for improved forecasting accuracy.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for time series forecasting.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.processed_path = Path(self.config['data']['processed_data_path'])
        self.features_path = Path(self.config['data']['features_data_path'])
        self.scalers = {}
        self.encoders = {}
        
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
                'processed_data_path': 'data/processed/',
                'features_data_path': 'data/features/'
            },
            'features': {
                'lag_periods': [1, 7, 14, 30, 90],
                'rolling_windows': [7, 14, 30, 60, 90],
                'seasonal_periods': [7, 30, 365]
            }
        }
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed data from CSV files."""
        try:
            train_data = pd.read_csv(self.processed_path / "train_data.csv")
            val_data = pd.read_csv(self.processed_path / "val_data.csv")
            test_data = pd.read_csv(self.processed_path / "test_data.csv")
            
            # Convert date columns
            for df in [train_data, val_data, test_data]:
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info("Processed data loaded successfully")
            return train_data, val_data, test_data
            
        except FileNotFoundError:
            logger.error("Processed data files not found. Please run preprocessing first.")
            raise
    
    def create_lag_features(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            df: Input dataframe
            target_cols: Columns to create lag features for
            
        Returns:
            pd.DataFrame: Dataframe with lag features
        """
        logger.info("Creating lag features...")
        
        df_with_lags = df.copy()
        lag_periods = self.config['features']['lag_periods']
        
        for col in target_cols:
            if col in df.columns:
                for lag in lag_periods:
                    df_with_lags[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df_with_lags
    
    def create_rolling_features(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """
        Create rolling window statistical features.
        
        Args:
            df: Input dataframe
            target_cols: Columns to create rolling features for
            
        Returns:
            pd.DataFrame: Dataframe with rolling features
        """
        logger.info("Creating rolling window features...")
        
        df_with_rolling = df.copy()
        rolling_windows = self.config['features']['rolling_windows']
        
        for col in target_cols:
            if col in df.columns:
                for window in rolling_windows:
                    # Rolling mean
                    df_with_rolling[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    
                    # Rolling standard deviation
                    df_with_rolling[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                    
                    # Rolling minimum and maximum
                    df_with_rolling[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                    df_with_rolling[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                    
                    # Rolling median
                    df_with_rolling[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window).median()
        
        return df_with_rolling
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from date column.
        
        Args:
            df: Input dataframe with date column
            
        Returns:
            pd.DataFrame: Dataframe with time features
        """
        logger.info("Creating time-based features...")
        
        df_with_time = df.copy()
        
        # Extract basic time components
        df_with_time['year'] = df_with_time['date'].dt.year
        df_with_time['month'] = df_with_time['date'].dt.month
        df_with_time['day'] = df_with_time['date'].dt.day
        df_with_time['dayofweek'] = df_with_time['date'].dt.dayofweek
        df_with_time['dayofyear'] = df_with_time['date'].dt.dayofyear
        df_with_time['week'] = df_with_time['date'].dt.isocalendar().week
        df_with_time['quarter'] = df_with_time['date'].dt.quarter
        
        # Create cyclical features
        df_with_time['month_sin'] = np.sin(2 * np.pi * df_with_time['month'] / 12)
        df_with_time['month_cos'] = np.cos(2 * np.pi * df_with_time['month'] / 12)
        df_with_time['day_sin'] = np.sin(2 * np.pi * df_with_time['day'] / 31)
        df_with_time['day_cos'] = np.cos(2 * np.pi * df_with_time['day'] / 31)
        df_with_time['dayofweek_sin'] = np.sin(2 * np.pi * df_with_time['dayofweek'] / 7)
        df_with_time['dayofweek_cos'] = np.cos(2 * np.pi * df_with_time['dayofweek'] / 7)
        
        # Binary indicators
        df_with_time['is_weekend'] = (df_with_time['dayofweek'] >= 5).astype(int)
        df_with_time['is_month_start'] = (df_with_time['day'] <= 5).astype(int)
        df_with_time['is_month_end'] = (df_with_time['day'] >= 25).astype(int)
        
        # Season indicators
        df_with_time['is_spring'] = df_with_time['month'].isin([3, 4, 5]).astype(int)
        df_with_time['is_summer'] = df_with_time['month'].isin([6, 7, 8]).astype(int)
        df_with_time['is_autumn'] = df_with_time['month'].isin([9, 10, 11]).astype(int)
        df_with_time['is_winter'] = df_with_time['month'].isin([12, 1, 2]).astype(int)
        
        return df_with_time
    
    def create_seasonal_features(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """
        Create seasonal decomposition features.
        
        Args:
            df: Input dataframe
            target_cols: Columns to create seasonal features for
            
        Returns:
            pd.DataFrame: Dataframe with seasonal features
        """
        logger.info("Creating seasonal features...")
        
        df_with_seasonal = df.copy()
        seasonal_periods = self.config['features']['seasonal_periods']
        
        for col in target_cols:
            if col in df.columns:
                for period in seasonal_periods:
                    # Simple seasonal decomposition using moving averages
                    df_with_seasonal[f'{col}_seasonal_{period}'] = (
                        df[col].rolling(window=period, center=True).mean()
                    )
        
        return df_with_seasonal
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        logger.info("Creating interaction features...")
        
        df_with_interactions = df.copy()
        
        # Get category columns (assuming they are the expense categories)
        category_cols = [col for col in df.columns if col not in ['date', 'year', 'month', 'day', 
                        'dayofweek', 'dayofyear', 'week', 'quarter', 'is_weekend', 'is_month_start', 
                        'is_month_end', 'is_spring', 'is_summer', 'is_autumn', 'is_winter']]
        
        # Total daily expense
        if len(category_cols) > 0:
            df_with_interactions['total_daily_expense'] = df_with_interactions[category_cols].sum(axis=1)
            
            # Expense ratios
            for col in category_cols[:5]:  # Limit to top 5 categories to avoid too many features
                df_with_interactions[f'{col}_ratio'] = (
                    df_with_interactions[col] / (df_with_interactions['total_daily_expense'] + 1e-8)
                )
            
            # Weekend vs weekday spending
            weekend_mask = df_with_interactions['is_weekend'] == 1
            for col in category_cols[:3]:  # Top 3 categories
                weekend_mean = df_with_interactions[weekend_mask][col].mean()
                weekday_mean = df_with_interactions[~weekend_mask][col].mean()
                df_with_interactions[f'{col}_weekend_ratio'] = weekend_mean / (weekday_mean + 1e-8)
        
        return df_with_interactions
    
    def create_volatility_features(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """
        Create volatility and stability features.
        
        Args:
            df: Input dataframe
            target_cols: Columns to create volatility features for
            
        Returns:
            pd.DataFrame: Dataframe with volatility features
        """
        logger.info("Creating volatility features...")
        
        df_with_volatility = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                # Coefficient of variation (rolling)
                rolling_mean = df[col].rolling(window=30).mean()
                rolling_std = df[col].rolling(window=30).std()
                df_with_volatility[f'{col}_cv_30'] = rolling_std / (rolling_mean + 1e-8)
                
                # Rate of change
                df_with_volatility[f'{col}_pct_change'] = df[col].pct_change()
                df_with_volatility[f'{col}_pct_change_7d'] = df[col].pct_change(periods=7)
                
                # Volatility (rolling standard deviation)
                df_with_volatility[f'{col}_volatility_7d'] = df[col].rolling(window=7).std()
                df_with_volatility[f'{col}_volatility_30d'] = df[col].rolling(window=30).std()
        
        return df_with_volatility
    
    def scale_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of scaled dataframes
        """
        logger.info("Scaling numerical features...")
        
        # Identify numerical columns (exclude date and binary features)
        numerical_cols = []
        for col in train_df.columns:
            if col not in ['date'] and train_df[col].dtype in ['int64', 'float64']:
                if not col.startswith('is_'):  # Exclude binary indicators
                    numerical_cols.append(col)
        
        # Fit scaler on training data
        scaler = StandardScaler()
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()
        
        # Scale numerical columns
        train_scaled[numerical_cols] = scaler.fit_transform(train_df[numerical_cols].fillna(0))
        val_scaled[numerical_cols] = scaler.transform(val_df[numerical_cols].fillna(0))
        test_scaled[numerical_cols] = scaler.transform(test_df[numerical_cols].fillna(0))
        
        # Store scaler for later use
        self.scalers['numerical'] = scaler
        
        return train_scaled, val_scaled, test_scaled
    
    def run_feature_engineering_pipeline(self) -> None:
        """
        Run the complete feature engineering pipeline.
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Load processed data
        train_data, val_data, test_data = self.load_processed_data()
        
        # Get category columns (expense categories)
        category_cols = [col for col in train_data.columns if col not in ['date']]
        
        # Apply feature engineering to each dataset
        datasets = {'train': train_data, 'val': val_data, 'test': test_data}
        engineered_datasets = {}
        
        for name, df in datasets.items():
            logger.info(f"Engineering features for {name} dataset...")
            
            # Create time features
            df_features = self.create_time_features(df)
            
            # Create lag features
            df_features = self.create_lag_features(df_features, category_cols)
            
            # Create rolling features
            df_features = self.create_rolling_features(df_features, category_cols)
            
            # Create seasonal features
            df_features = self.create_seasonal_features(df_features, category_cols)
            
            # Create interaction features
            df_features = self.create_interaction_features(df_features)
            
            # Create volatility features
            df_features = self.create_volatility_features(df_features, category_cols)
            
            engineered_datasets[name] = df_features
        
        # Scale features
        train_scaled, val_scaled, test_scaled = self.scale_features(
            engineered_datasets['train'], 
            engineered_datasets['val'], 
            engineered_datasets['test']
        )
        
        # Save engineered features
        self.save_engineered_features(train_scaled, val_scaled, test_scaled)
        
        logger.info("Feature engineering pipeline completed successfully!")
        
        # Print summary
        self._print_feature_summary(train_scaled)
    
    def save_engineered_features(self, train_data: pd.DataFrame, 
                               val_data: pd.DataFrame, 
                               test_data: pd.DataFrame) -> None:
        """
        Save engineered features to files.
        
        Args:
            train_data: Training dataset with features
            val_data: Validation dataset with features
            test_data: Test dataset with features
        """
        logger.info("Saving engineered features...")
        
        # Create features directory if it doesn't exist
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_data.to_csv(self.features_path / "train_features.csv", index=False)
        val_data.to_csv(self.features_path / "val_features.csv", index=False)
        test_data.to_csv(self.features_path / "test_features.csv", index=False)
        
        # Save scalers and encoders
        import joblib
        joblib.dump(self.scalers, self.features_path / "scalers.pkl")
        joblib.dump(self.encoders, self.features_path / "encoders.pkl")
        
        logger.info("Engineered features saved successfully!")
    
    def _print_feature_summary(self, df: pd.DataFrame) -> None:
        """Print feature engineering summary."""
        
        print("\n" + "="*50)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*50)
        
        print(f"📊 Total features: {len(df.columns)}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"📈 Training samples: {len(df):,}")
        
        # Count feature types
        lag_features = len([col for col in df.columns if '_lag_' in col])
        rolling_features = len([col for col in df.columns if '_rolling_' in col])
        time_features = len([col for col in df.columns if col in ['year', 'month', 'day', 'dayofweek', 
                            'is_weekend', 'is_month_start', 'is_month_end']])
        seasonal_features = len([col for col in df.columns if '_seasonal_' in col])
        interaction_features = len([col for col in df.columns if '_ratio' in col])
        volatility_features = len([col for col in df.columns if 'volatility' in col or 'pct_change' in col])
        
        print(f"\n🔧 Feature breakdown:")
        print(f"   📊 Lag features: {lag_features}")
        print(f"   📈 Rolling features: {rolling_features}")
        print(f"   📅 Time features: {time_features}")
        print(f"   🌊 Seasonal features: {seasonal_features}")
        print(f"   🔗 Interaction features: {interaction_features}")
        print(f"   📊 Volatility features: {volatility_features}")
        
        print("="*50)


def main():
    """Main function to run feature engineering pipeline."""
    engineer = FeatureEngineer()
    engineer.run_feature_engineering_pipeline()


if __name__ == "__main__":
    main()