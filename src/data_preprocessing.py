#!/usr/bin/env python3
"""
BudgetWise AI - Advanced Data Preprocessing Module
Copyright (c) 2025 moah0911
Repository: https://github.com/moah0911/BudgetWise-AI-based-Expense-Forecasting-Tool

This file is part of BudgetWise AI project - Personal Expense Forecasting Tool.
Licensed under MIT License with Attribution Requirement.

Advanced Data Preprocessing Module for BudgetWise Forecasting System
Implements comprehensive data loading, cleaning, merging, and validation based on notebook analysis.
Handles multiple datasets with intelligent duplicate resolution, data quality improvement, and standardization.

Author: moah0911
Created: October 2025
Project: BudgetWise AI - Personal Expense Forecasting Tool
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
from datetime import datetime, timedelta
import warnings
import re
from collections import Counter
from scipy import stats

# Try to import rapidfuzz for fuzzy matching, fallback to basic matching if not available
try:
    from rapidfuzz import process
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not FUZZY_MATCHING_AVAILABLE:
    logger.warning("rapidfuzz not available - using basic string matching for category standardization")

class AdvancedDataPreprocessor:
    """
    Advanced data preprocessing for BudgetWise financial data with multi-dataset support.
    Implements comprehensive cleaning, merging, and quality improvement strategies.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.raw_data_path = Path(self.config['data']['raw_data_path'])
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.datasets = {}
        self.combined_data = None
        self.processed_data = None
        
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
                'raw_data_path': 'data/raw/',
                'processed_data_path': 'data/processed/',
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15
            },
            'preprocessing': {
                'duplicate_threshold': 0.95,
                'outlier_method': 'iqr',
                'outlier_factor': 1.5,
                'date_formats': ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y'],
                'amount_patterns': [r'₹\s*([0-9,]+\.?[0-9]*)', r'\$\s*([0-9,]+\.?[0-9]*)', r'([0-9,]+\.?[0-9]*)']
            }
        }

    def load_multiple_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load multiple BudgetWise datasets from raw data directory.
        
        Returns:
            Dictionary containing loaded datasets
        """
        logger.info("🔄 Loading Multiple BudgetWise Finance Datasets...")
        logger.info("=" * 60)
        
        dataset_files = {
            'primary': 'budgetwise_finance_dataset.csv',
            'secondary': 'budgetwise_synthetic_dirty.csv',
            'combined_original': 'budgetwise_original_combined.csv'
        }
        
        datasets = {}
        
        for dataset_name, filename in dataset_files.items():
            file_path = self.raw_data_path / filename
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    df['dataset_source'] = dataset_name
                    datasets[dataset_name] = df
                    logger.info(f"✅ {dataset_name.title()} dataset loaded: {df.shape[0]:,} records, {df.shape[1]} columns")
                except Exception as e:
                    logger.error(f"❌ Error loading {dataset_name} dataset: {e}")
            else:
                logger.warning(f"⚠️  {dataset_name.title()} dataset not found: {file_path}")
        
        if not datasets:
            raise FileNotFoundError("No datasets found in the raw data directory")
        
        # Store datasets
        self.datasets = datasets
        
        # Log comparison
        logger.info(f"\n🔍 Dataset Comparison:")
        for name, df in datasets.items():
            if name != 'combined_original':  # Skip already processed dataset
                main_columns = [col for col in df.columns if col != 'dataset_source']
                logger.info(f"📋 {name.title()} dataset columns: {main_columns}")
        
        logger.info(f"📊 Total records to process: {sum(df.shape[0] for df in datasets.values()):,}")
        
        return datasets

    def merge_datasets(self) -> pd.DataFrame:
        """
        Intelligently merge multiple datasets with conflict resolution.
        
        Returns:
            Combined and cleaned DataFrame
        """
        logger.info("🔗 Merging datasets with intelligent conflict resolution...")
        
        # If we have the combined_original dataset, use it as base
        if 'combined_original' in self.datasets:
            logger.info("✅ Using existing combined dataset as base")
            combined_df = self.datasets['combined_original'].copy()
            
            # Remove the extra preprocessing columns to get clean base
            base_columns = ['transaction_id', 'user_id', 'date', 'transaction_type', 
                          'category', 'amount', 'payment_mode', 'location', 'notes']
            
            # Keep dataset source if available
            if 'dataset_source' in combined_df.columns:
                base_columns.append('dataset_source')
            
            # Filter to base columns that exist
            available_columns = [col for col in base_columns if col in combined_df.columns]
            combined_df = combined_df[available_columns].copy()
            
        else:
            # Merge primary and secondary datasets
            datasets_to_merge = []
            
            if 'primary' in self.datasets:
                datasets_to_merge.append(self.datasets['primary'])
            
            if 'secondary' in self.datasets:
                datasets_to_merge.append(self.datasets['secondary'])
            
            if not datasets_to_merge:
                raise ValueError("No datasets available for merging")
            
            # Combine datasets
            combined_df = pd.concat(datasets_to_merge, ignore_index=True)
            logger.info(f"✅ Combined datasets: {combined_df.shape[0]:,} total records")
        
        # Store combined data
        self.combined_data = combined_df.copy()
        
        # Display dataset source distribution
        if 'dataset_source' in combined_df.columns:
            source_dist = combined_df['dataset_source'].value_counts()
            logger.info(f"\n📋 Dataset source distribution:")
            for source, count in source_dist.items():
                percentage = (count / len(combined_df)) * 100
                logger.info(f"   • {str(source).title()}: {count:,} records ({percentage:.1f}%)")
        
        return combined_df

    def advanced_duplicate_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced duplicate detection and resolution with intelligent handling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates resolved
        """
        logger.info("🔍 Advanced Duplicate Detection & Resolution...")
        logger.info("=" * 50)
        
        initial_count = len(df)
        
        # Step 1: Remove exact duplicate rows
        logger.info("\n🎯 Step 1: Exact Duplicate Row Detection")
        df_before_dedup = df.copy()
        df = df.drop_duplicates()
        exact_duplicates_removed = initial_count - len(df)
        logger.info(f"   • Exact duplicate rows removed: {exact_duplicates_removed:,}")
        logger.info(f"   • Remaining records: {len(df):,}")
        
        # Step 2: Analyze transaction ID duplicates
        logger.info("\n🎯 Step 2: Transaction ID Duplicate Analysis")
        if 'transaction_id' in df.columns:
            duplicate_txn_ids = df['transaction_id'].duplicated().sum()
            unique_txn_ids = df['transaction_id'].nunique()
            total_txn_ids = len(df)
            
            logger.info(f"   • Total transaction records: {total_txn_ids:,}")
            logger.info(f"   • Unique transaction IDs: {unique_txn_ids:,}")
            logger.info(f"   • Duplicate transaction IDs: {duplicate_txn_ids:,}")
            logger.info(f"   • Duplication rate: {(duplicate_txn_ids/total_txn_ids)*100:.2f}%")
            
            if duplicate_txn_ids > 0:
                logger.info(f"   ⚠️  Found {duplicate_txn_ids:,} duplicate transaction IDs")
                
                # Step 3: Intelligent duplicate resolution
                logger.info(f"\n🎯 Step 3: Intelligent Duplicate Resolution")
                
                # Create unique transaction IDs for duplicates
                duplicated_mask = df['transaction_id'].duplicated(keep='first')
                duplicate_count = duplicated_mask.sum()
                
                # Generate unique suffixes for duplicates
                df.loc[duplicated_mask, 'transaction_id'] = (
                    df.loc[duplicated_mask, 'transaction_id'] + '_DUP_' + 
                    df.loc[duplicated_mask].groupby('transaction_id').cumcount().add(1).astype(str)
                )
                
                logger.info(f"   ✅ Created unique transaction IDs for {duplicate_count:,} duplicate records")
                
                # Add duplicate flag for analysis
                df['was_duplicate'] = duplicated_mask.astype(int)
            else:
                logger.info("   ✅ No duplicate transaction IDs found")
                df['was_duplicate'] = 0
        else:
            logger.warning("   ⚠️  No transaction_id column found for duplicate analysis")
            df['was_duplicate'] = 0
        
        # Summary
        logger.info(f"\n📊 Duplicate Resolution Summary:")
        logger.info(f"   • Initial records: {initial_count:,}")
        logger.info(f"   • Exact duplicates removed: {exact_duplicates_removed:,}")
        logger.info(f"   • Final records: {len(df):,}")
        logger.info(f"   • Data retention rate: {(len(df)/initial_count)*100:.2f}%")
        
        return df

    def advanced_date_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced date processing with comprehensive parsing, quality validation, and intelligent imputation.
        
        Args:
            df: DataFrame with date columns
            
        Returns:
            DataFrame with properly processed dates
        """
        logger.info("📅 Advanced Date Processing & Standardization...")
        logger.info("=" * 50)
        
        if 'date' not in df.columns:
            logger.warning("⚠️  No 'date' column found in DataFrame")
            return df
        
        # Store original date column for analysis
        df['date_original'] = df['date'].copy()
        
        logger.info("\n🎯 Step 1: Date Format Analysis")
        # Analyze existing date formats
        date_samples = df['date'].dropna().astype(str).head(20).tolist()
        logger.info(f"   📋 Sample date formats found:")
        for i, date_sample in enumerate(date_samples[:10], 1):
            logger.info(f"      {i:2d}. {date_sample}")
        
        # Attempt to parse multiple date formats
        logger.info(f"\n🎯 Step 2: Multi-format Date Parsing")
        date_formats = [
            '%Y-%m-%d',     # 2023-01-15
            '%m/%d/%Y',     # 01/15/2023
            '%d/%m/%Y',     # 15/01/2023
            '%Y/%m/%d',     # 2023/01/15
            '%B %d %Y',     # January 15 2023
            '%d %B %Y',     # 15 January 2023
            '%d-%m-%Y',     # 15-01-2023
        ]
        
        # Primary parsing attempt
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
        
        # Count unparseable dates
        unparseable_dates = df['date'].isna().sum()
        parseable_dates = len(df) - unparseable_dates
        parsing_success_rate = (parseable_dates / len(df)) * 100
        
        logger.info(f"   • Successfully parsed dates: {parseable_dates:,} ({parsing_success_rate:.1f}%)")
        logger.info(f"   • Unparseable dates found: {unparseable_dates:,} ({(unparseable_dates/len(df)*100):.1f}%)")
        
        # Advanced date imputation strategy
        if unparseable_dates > 0:
            logger.info(f"\n🎯 Step 3: Intelligent Date Imputation")
            
            # Use statistical imputation
            valid_dates = df['date'].dropna()
            
            if len(valid_dates) > 0:
                # Multiple imputation strategies
                date_median = valid_dates.median()
                date_mode = valid_dates.mode()[0] if not valid_dates.mode().empty else date_median
                date_mean = valid_dates.mean()
                
                logger.info(f"   📊 Date statistics for imputation:")
                logger.info(f"      • Median date: {pd.to_datetime(date_median).strftime('%Y-%m-%d')}")
                logger.info(f"      • Mode date: {pd.to_datetime(date_mode).strftime('%Y-%m-%d')}")
                logger.info(f"      • Mean date: {pd.to_datetime(date_mean).strftime('%Y-%m-%d')}")
                
                # Use mode for imputation (most frequent date)
                df['date'] = df['date'].fillna(date_mode)
                logger.info(f"   ✅ Filled {unparseable_dates:,} missing dates with mode date: {pd.to_datetime(date_mode).strftime('%Y-%m-%d')}")
                
                # Add imputation flag
                df['date_imputed'] = df['date_original'].isna().astype(int)
            else:
                # Fallback to current date if no valid dates
                fallback_date = pd.Timestamp('2023-01-01')
                df['date'] = df['date'].fillna(fallback_date)
                df['date_imputed'] = 1
                logger.info(f"   ⚠️  Used fallback date {fallback_date.date()} for all missing dates")
        else:
            df['date_imputed'] = 0
            logger.info(f"   ✅ All dates successfully parsed - no imputation needed")
        
        # Date validation and quality checks
        logger.info(f"\n🎯 Step 4: Date Quality Validation")
        current_date = pd.Timestamp.now()
        earliest_reasonable_date = pd.Timestamp('2000-01-01')
        
        # Quality checks
        future_dates = (df['date'] > current_date).sum()
        very_old_dates = (df['date'] < earliest_reasonable_date).sum()
        reasonable_dates = len(df) - future_dates - very_old_dates
        
        logger.info(f"   📊 Date quality analysis:")
        logger.info(f"      • Reasonable dates: {reasonable_dates:,} ({(reasonable_dates/len(df)*100):.1f}%)")
        logger.info(f"      • Future dates: {future_dates:,} ({(future_dates/len(df)*100):.2f}%)")
        logger.info(f"      • Very old dates (pre-2000): {very_old_dates:,} ({(very_old_dates/len(df)*100):.2f}%)")
        
        # Add quality flags
        df['date_is_future'] = (df['date'] > current_date).astype(int)
        df['date_is_very_old'] = (df['date'] < earliest_reasonable_date).astype(int)
        
        # Final date statistics
        logger.info(f"\n📊 Final Date Processing Results:")
        logger.info(f"   ✅ Date standardization complete")
        logger.info(f"   📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        logger.info(f"   📈 Date span: {(df['date'].max() - df['date'].min()).days:,} days")
        logger.info(f"   🎯 Processing success rate: {((len(df) - future_dates - very_old_dates)/len(df)*100):.1f}%")
        
        # Add temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info("\n✅ Advanced date processing completed!")
        
        return df

    def advanced_amount_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced amount processing with comprehensive currency normalization and intelligent imputation.
        
        Args:
            df: DataFrame with amount columns
            
        Returns:
            DataFrame with properly processed amounts
        """
        logger.info("💰 Advanced Amount Processing & Currency Normalization...")
        logger.info("=" * 50)
        
        if 'amount' not in df.columns:
            logger.warning("⚠️  No 'amount' column found in DataFrame")
            return df
        
        # Store original amount for comparison
        df["amount_original"] = df["amount"].copy()
        
        logger.info("\n🎯 Step 1: Currency Format Analysis")
        # Analyze currency formats in the data
        amount_samples = df["amount"].dropna().astype(str).head(20).tolist()
        logger.info(f"   📋 Sample amount formats found:")
        unique_formats = set()
        for i, amount_sample in enumerate(amount_samples[:10], 1):
            logger.info(f"      {i:2d}. {amount_sample}")
            # Extract format pattern
            if "₹" in str(amount_sample):
                unique_formats.add("Rupee symbol (₹)")
            elif "$" in str(amount_sample):
                unique_formats.add("Dollar symbol ($)")
            elif "Rs" in str(amount_sample):
                unique_formats.add("Rs prefix")
            elif "," in str(amount_sample):
                unique_formats.add("Comma separated")
        
        logger.info(f"   🔍 Detected currency formats: {list(unique_formats)}")
        
        logger.info("\n🎯 Step 2: Advanced Currency Cleaning")
        # Comprehensive currency symbol removal
        currency_patterns = [
            r"[₹$£€¥Rs.]",      # Currency symbols
            r"INR|USD|EUR|GBP",  # Currency codes
            r"[,]",              # Thousands separators
            r"\s+"               # Extra whitespace
        ]
        
        df["amount_cleaned"] = df["amount"].astype(str)
        for pattern in currency_patterns:
            df["amount_cleaned"] = df["amount_cleaned"].str.replace(pattern, "", regex=True)
        
        df["amount_cleaned"] = df["amount_cleaned"].str.strip()
        
        # Convert to numeric
        df["amount"] = pd.to_numeric(df["amount_cleaned"], errors="coerce")
        
        # Analyze conversion results
        conversion_success = df["amount"].notna().sum()
        conversion_failures = df["amount"].isna().sum()
        conversion_rate = (conversion_success / len(df)) * 100
        
        logger.info(f"   ✅ Currency conversion results:")
        logger.info(f"      • Successful conversions: {conversion_success:,} ({conversion_rate:.1f}%)")
        logger.info(f"      • Failed conversions: {conversion_failures:,} ({(conversion_failures/len(df)*100):.1f}%)")
        
        # CRITICAL FIX: Cap unrealistic transaction amounts
        logger.info(f"\n🎯 Step 2.5: Realistic Transaction Amount Capping")
        valid_amounts = df["amount"].dropna()
        
        if len(valid_amounts) > 0:
            # Define realistic transaction limits for personal expenses
            MAX_REALISTIC_TRANSACTION = 100000  # ₹1 lakh maximum per transaction
            MIN_REALISTIC_TRANSACTION = 1       # ₹1 minimum per transaction
            
            # Analyze extreme amounts before capping
            extreme_high = (valid_amounts > MAX_REALISTIC_TRANSACTION).sum()
            extreme_low = (valid_amounts < MIN_REALISTIC_TRANSACTION).sum()
            extreme_high_max = valid_amounts[valid_amounts > MAX_REALISTIC_TRANSACTION].max() if extreme_high > 0 else 0
            
            logger.info(f"   📊 Transaction amount analysis:")
            logger.info(f"      • Valid transactions: {len(valid_amounts):,}")
            logger.info(f"      • Unrealistically high (>₹{MAX_REALISTIC_TRANSACTION:,}): {extreme_high:,}")
            logger.info(f"      • Unrealistically low (<₹{MIN_REALISTIC_TRANSACTION:,}): {extreme_low:,}")
            if extreme_high > 0:
                logger.info(f"      • Highest unrealistic amount: ₹{extreme_high_max:,.2f}")
            
            # Cap extreme amounts to realistic levels
            original_amounts = df["amount"].copy()
            df.loc[df["amount"] > MAX_REALISTIC_TRANSACTION, "amount"] = MAX_REALISTIC_TRANSACTION
            df.loc[(df["amount"] > 0) & (df["amount"] < MIN_REALISTIC_TRANSACTION), "amount"] = MIN_REALISTIC_TRANSACTION
            
            # Count and report capping actions
            capped_high = (original_amounts > MAX_REALISTIC_TRANSACTION).sum()
            capped_low = ((original_amounts > 0) & (original_amounts < MIN_REALISTIC_TRANSACTION)).sum()
            total_capped = capped_high + capped_low
            
            # Add capping flags for transparency
            df["amount_capped_high"] = (original_amounts > MAX_REALISTIC_TRANSACTION).astype(int)
            df["amount_capped_low"] = ((original_amounts > 0) & (original_amounts < MIN_REALISTIC_TRANSACTION)).astype(int)
            df["amount_capped"] = (df["amount_capped_high"] | df["amount_capped_low"]).astype(int)
            
            logger.info(f"   ✅ Transaction capping results:")
            logger.info(f"      • High amounts capped to ₹{MAX_REALISTIC_TRANSACTION:,}: {capped_high:,}")
            logger.info(f"      • Low amounts capped to ₹{MIN_REALISTIC_TRANSACTION:,}: {capped_low:,}")
            logger.info(f"      • Total transactions capped: {total_capped:,} ({(total_capped/len(df)*100):.2f}%)")
            
            # Show new amount statistics
            new_valid_amounts = df["amount"].dropna()
            logger.info(f"   📊 Post-capping amount statistics:")
            logger.info(f"      • New Min: ₹{new_valid_amounts.min():,.2f}")
            logger.info(f"      • New Max: ₹{new_valid_amounts.max():,.2f}")
            logger.info(f"      • New Mean: ₹{new_valid_amounts.mean():,.2f}")
            logger.info(f"      • New Median: ₹{new_valid_amounts.median():,.2f}")
        else:
            logger.warning("   ⚠️  No valid amounts found for capping analysis")
            df["amount_capped_high"] = 0
            df["amount_capped_low"] = 0
            df["amount_capped"] = 0
        
        # Handle missing amounts with intelligent imputation
        if conversion_failures > 0:
            logger.info(f"\n🎯 Step 3: Intelligent Amount Imputation")
            
            valid_amounts = df["amount"].dropna()
            if len(valid_amounts) > 0:
                # Statistical measures for imputation
                amount_median = valid_amounts.median()
                amount_mean = valid_amounts.mean()
                amount_mode = valid_amounts.mode()[0] if not valid_amounts.mode().empty else amount_median
                
                logger.info(f"   📊 Amount statistics for imputation:")
                logger.info(f"      • Median: ₹{amount_median:,.2f}")
                logger.info(f"      • Mean: ₹{amount_mean:,.2f}")
                logger.info(f"      • Mode: ₹{amount_mode:,.2f}")
                
                # Use median for imputation (robust to outliers)
                df["amount"] = df["amount"].fillna(amount_median)
                logger.info(f"   ✅ Filled {conversion_failures:,} missing amounts with median: ₹{amount_median:,.2f}")
                
                # Add imputation flag
                df["amount_imputed"] = df["amount_original"].isna().astype(int)
            else:
                df["amount"] = df["amount"].fillna(1000)  # Fallback value
                df["amount_imputed"] = 1
        else:
            df["amount_imputed"] = 0
        
        logger.info("\n🎯 Step 4: Amount Quality Validation")
        # Quality checks and corrections
        negative_amounts = (df["amount"] < 0).sum()
        zero_amounts = (df["amount"] == 0).sum()
        positive_amounts = (df["amount"] > 0).sum()
        
        logger.info(f"   📊 Amount quality analysis:")
        logger.info(f"      • Positive amounts: {positive_amounts:,} ({(positive_amounts/len(df)*100):.1f}%)")
        logger.info(f"      • Zero amounts: {zero_amounts:,} ({(zero_amounts/len(df)*100):.2f}%)")
        logger.info(f"      • Negative amounts: {negative_amounts:,} ({(negative_amounts/len(df)*100):.2f}%)")
        
        # Handle negative and zero amounts
        if negative_amounts > 0 or zero_amounts > 0:
            invalid_amounts = negative_amounts + zero_amounts
            median_replacement = df[df["amount"] > 0]["amount"].median() if positive_amounts > 0 else 1000
            
            df.loc[df["amount"] <= 0, "amount"] = median_replacement
            logger.info(f"   ✅ Replaced {invalid_amounts:,} invalid amounts with median: ₹{median_replacement:,.2f}")
            
            # Add quality flags
            df["amount_was_invalid"] = ((df["amount_original"].astype(str).str.contains(r"^[0-]|^-", na=False)) | 
                                       (pd.to_numeric(df["amount_original"], errors="coerce") <= 0)).astype(int)
        else:
            df["amount_was_invalid"] = 0
        
        # Clean up temporary columns
        df.drop(['amount_cleaned'], axis=1, inplace=True, errors='ignore')
        
        logger.info("\n✅ Advanced amount processing completed!")
        
        return df

    def _fuzzy_standardize(self, series: pd.Series, valid_list: List[str], threshold: int = 80) -> List[str]:
        """
        Apply fuzzy matching to standardize category names.
        
        Args:
            series: Series of category names to standardize
            valid_list: List of valid category names
            threshold: Minimum similarity score for matching
            
        Returns:
            List of standardized category names
        """
        cleaned = []
        for val in series:
            if pd.isna(val):
                cleaned.append("Others")
                continue
            
            if FUZZY_MATCHING_AVAILABLE:
                match, score, _ = process.extractOne(str(val), valid_list)
                cleaned.append(match if score >= threshold else "Others")
            else:
                # Fallback to exact matching
                val_lower = str(val).lower()
                best_match = None
                for valid_cat in valid_list:
                    if val_lower in valid_cat.lower() or valid_cat.lower() in val_lower:
                        best_match = valid_cat
                        break
                cleaned.append(best_match if best_match else "Others")
        
        return cleaned
    
    def standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced category standardization using fuzzy matching and intelligent grouping.
        
        Args:
            df: DataFrame with category columns
            
        Returns:
            DataFrame with standardized categories
        """
        logger.info("🏷️  Advanced Category Standardization with Fuzzy Matching...")
        logger.info("=" * 50)
        
        if 'category' not in df.columns:
            logger.warning("⚠️  No 'category' column found in DataFrame")
            return df
        
        # Keep original copy
        df['category_original'] = df['category'].astype(str).str.strip()
        
        # Analyze current categories
        logger.info("\n🎯 Step 1: Category Analysis")
        unique_categories = df['category'].value_counts()
        logger.info(f"   • Total unique categories: {len(unique_categories)}")
        logger.info(f"   • Top 10 categories:")
        for cat, count in unique_categories.head(10).items():
            percentage = (count / len(df)) * 100
            logger.info(f"     - {cat}: {count:,} transactions ({percentage:.1f}%)")
        
        # Define valid master categories (comprehensive list)
        valid_categories = [
            'Education', 'Housing', 'Income', 'Food & Dining', 'Entertainment',
            'Bills & Utilities', 'Others', 'Transportation', 'Healthcare', 
            'Travel', 'Savings', 'Shopping', 'Personal Care', 'Financial'
        ]
        
        logger.info(f"\n🎯 Step 2: Fuzzy Category Standardization")
        logger.info(f"   🎯 Using {'fuzzy matching' if FUZZY_MATCHING_AVAILABLE else 'basic string matching'}")
        
        # Apply fuzzy cleaning with intelligent matching
        df['category'] = self._fuzzy_standardize(df['category_original'], valid_categories, threshold=80)
        
        # Report standardization results
        cats_before = df['category_original'].nunique()
        cats_after = df['category'].nunique()
        df['category_was_standardized'] = (df['category_original'].str.lower() != df['category'].str.lower()).astype(int)
        
        standardized_count = df['category_was_standardized'].sum()
        reduction_percentage = ((cats_before - cats_after) / cats_before * 100) if cats_before > 0 else 0
        
        logger.info(f"\n📊 Category Standardization Results:")
        logger.info(f"   ✅ Category standardization complete")
        logger.info(f"   📊 Categories reduced from {cats_before} to {cats_after} ({reduction_percentage:.1f}% reduction)")
        logger.info(f"   🔄 Entries standardized: {standardized_count:,} ({(standardized_count/len(df)*100):.1f}%)")
        logger.info(f"   📋 Final categories: {sorted(df['category'].unique())}")
        
        # Final category distribution
        final_categories = df['category'].value_counts()
        logger.info(f"\n   • Standardized category distribution:")
        for cat, count in final_categories.items():
            percentage = (count / len(df)) * 100
            logger.info(f"     - {cat}: {count:,} transactions ({percentage:.1f}%)")
        
        return df
    
    def standardize_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize location names using fuzzy matching.
        
        Args:
            df: DataFrame with location columns
            
        Returns:
            DataFrame with standardized locations
        """
        if 'location' not in df.columns:
            return df
        
        logger.info("🗺️  Standardizing Location Names...")
        
        # Keep original copy
        df['location_original'] = df['location'].astype(str).str.strip()
        
        # Define major Indian cities and common locations
        valid_locations = [
            'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Chennai', 
            'Kolkata', 'Surat', 'Pune', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur',
            'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Pimpri-Chinchwad',
            'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana', 'Agra', 'Nashik',
            'Online', 'Unknown', 'Others'
        ]
        
        # Apply fuzzy standardization
        df['location'] = self._fuzzy_standardize(df['location_original'], valid_locations, threshold=85)
        
        locations_before = df['location_original'].nunique()
        locations_after = df['location'].nunique()
        standardized_locations = (df['location_original'].str.lower() != df['location'].str.lower()).sum()
        
        logger.info(f"   ✅ Locations reduced from {locations_before} to {locations_after}")
        logger.info(f"   🔄 Location entries standardized: {standardized_locations:,}")
        
        return df
    
    def standardize_payment_modes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize payment mode names using fuzzy matching.
        
        Args:
            df: DataFrame with payment_mode columns
            
        Returns:
            DataFrame with standardized payment modes
        """
        if 'payment_mode' not in df.columns:
            return df
        
        logger.info("💳 Standardizing Payment Modes...")
        
        # Keep original copy
        df['payment_mode_original'] = df['payment_mode'].astype(str).str.strip()
        
        # Define standard payment modes
        valid_payment_modes = [
            'Credit Card', 'Debit Card', 'Cash', 'Bank Transfer', 'UPI', 
            'Net Banking', 'Mobile Wallet', 'Cheque', 'Unknown', 'Others'
        ]
        
        # Apply fuzzy standardization
        df['payment_mode'] = self._fuzzy_standardize(df['payment_mode_original'], valid_payment_modes, threshold=75)
        
        modes_before = df['payment_mode_original'].nunique()
        modes_after = df['payment_mode'].nunique()
        standardized_modes = (df['payment_mode_original'].str.lower() != df['payment_mode'].str.lower()).sum()
        
        logger.info(f"   ✅ Payment modes reduced from {modes_before} to {modes_after}")
        logger.info(f"   🔄 Payment mode entries standardized: {standardized_modes:,}")
        
        return df

    def detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)

    def detect_outliers_zscore(self, series: pd.Series, threshold: float = 3) -> pd.Series:
        """Detect outliers using Z-score method"""
        try:
            from scipy import stats
            z_scores = np.abs(stats.zscore(series.dropna()))
            # Create a boolean series with same index as original
            outliers = pd.Series(False, index=series.index)
            outliers.loc[series.dropna().index] = z_scores > threshold
            return outliers
        except Exception:
            # Fallback to manual z-score calculation
            mean_val = series.mean()
            std_val = series.std()
            if std_val == 0:
                return pd.Series(False, index=series.index)
            z_scores = np.abs((series - mean_val) / std_val)
            return z_scores > threshold

    def detect_outliers_modified_zscore(self, series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """Detect outliers using Modified Z-score method"""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            return pd.Series(False, index=series.index)
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold

    def detect_and_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔍 Advanced Outlier Detection & Treatment using multiple methods.
        
        Args:
            df: DataFrame with amount data
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info("🔍 Advanced Outlier Detection & Treatment...")
        logger.info("=" * 50)
        
        if 'amount' not in df.columns:
            logger.warning("⚠️  No 'amount' column found for outlier detection")
            return df
        
        # Multiple outlier detection methods
        logger.info("\n🎯 Step 1: Multi-Method Outlier Detection")
        amount_series = df["amount"]
        
        outliers_iqr = self.detect_outliers_iqr(amount_series)
        outliers_zscore = self.detect_outliers_zscore(amount_series)
        outliers_modified_zscore = self.detect_outliers_modified_zscore(amount_series)
        
        logger.info(f"   📊 Outlier detection results:")
        logger.info(f"      • IQR method: {outliers_iqr.sum():,} outliers ({outliers_iqr.sum()/len(df)*100:.2f}%)")
        logger.info(f"      • Z-score method: {outliers_zscore.sum():,} outliers ({outliers_zscore.sum()/len(df)*100:.2f}%)")
        logger.info(f"      • Modified Z-score: {outliers_modified_zscore.sum():,} outliers ({outliers_modified_zscore.sum()/len(df)*100:.2f}%)")
        
        # Consensus outlier detection
        consensus_outliers = (outliers_iqr.astype(int) + outliers_zscore.astype(int) + 
                             outliers_modified_zscore.astype(int)) >= 2
        logger.info(f"      • Consensus outliers: {consensus_outliers.sum():,} outliers ({consensus_outliers.sum()/len(df)*100:.2f}%)")
        
        # Outlier treatment
        logger.info("\n🎯 Step 2: Outlier Treatment Strategy")
        if consensus_outliers.sum() > 0:
            outlier_amounts = df.loc[consensus_outliers, "amount"]
            logger.info(f"   💰 Outlier amount statistics:")
            logger.info(f"      • Min outlier: ₹{outlier_amounts.min():,.2f}")
            logger.info(f"      • Max outlier: ₹{outlier_amounts.max():,.2f}")
            logger.info(f"      • Mean outlier: ₹{outlier_amounts.mean():,.2f}")
            
            # Cap extreme outliers at 99.5th percentile
            cap_value = df["amount"].quantile(0.995)
            extreme_outliers = df["amount"] > cap_value
            
            if extreme_outliers.sum() > 0:
                df.loc[extreme_outliers, "amount"] = cap_value
                logger.info(f"   ✅ Capped {extreme_outliers.sum():,} extreme outliers at ₹{cap_value:,.2f}")
        
        # Add outlier flags for analysis
        df["is_outlier_amount"] = consensus_outliers.astype(int)
        df["outlier_method_count"] = (outliers_iqr.astype(int) + outliers_zscore.astype(int) + 
                                      outliers_modified_zscore.astype(int))
        
        # Outlier analysis for other numeric columns
        logger.info("\n🎯 Step 3: Extended Outlier Analysis")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_columns:
            if col not in ["amount", "is_outlier_amount", "outlier_method_count", "amount_imputed", "amount_was_invalid"]:
                if df[col].notna().sum() > 10:  # Only analyze columns with sufficient data
                    col_outliers = self.detect_outliers_iqr(df[col].dropna())
                    outlier_summary[col] = {
                        "count": col_outliers.sum(),
                        "percentage": (col_outliers.sum() / len(df[col].dropna())) * 100
                    }
        
        if outlier_summary:
            logger.info("   📊 Outlier summary for numeric columns:")
            for col, stats in outlier_summary.items():
                logger.info(f"      • {col}: {stats['count']:,} outliers ({stats['percentage']:.2f}%)")
        
        # Final statistics
        initial_count = len(df)
        logger.info(f"\n📊 Outlier Handling Summary:")
        logger.info(f"   • Total records: {initial_count:,}")
        logger.info(f"   • Consensus outliers flagged: {consensus_outliers.sum():,}")
        logger.info(f"   • Extreme outliers capped: {extreme_outliers.sum() if 'extreme_outliers' in locals() else 0:,}")
        
        if len(df) > 0:
            logger.info(f"   • Final amount range: ₹{df['amount'].min():,.2f} to ₹{df['amount'].max():,.2f}")
            logger.info(f"   • Final average amount: ₹{df['amount'].mean():,.2f}")
        
        logger.info("\n✅ Advanced outlier detection completed!")
        
        return df

    def comprehensive_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute comprehensive data cleaning pipeline.
        
        Args:
            df: Raw combined DataFrame
            
        Returns:
            Thoroughly cleaned DataFrame
        """
        logger.info("🧹 Comprehensive Data Cleaning Pipeline...")
        logger.info("=" * 60)
        
        # Store initial statistics
        initial_shape = df.shape
        logger.info(f"🚀 Starting with {initial_shape[0]:,} records and {initial_shape[1]} columns")
        
        # Step 1: Advanced duplicate detection
        df = self.advanced_duplicate_detection(df)
        
        # Step 2: Advanced date processing
        df = self.advanced_date_processing(df)
        
        # Step 3: Advanced amount processing
        df = self.advanced_amount_processing(df)
        
        # Step 4: Category standardization with fuzzy matching
        df = self.standardize_categories(df)
        
        # Step 5: Location standardization
        df = self.standardize_locations(df)
        
        # Step 6: Payment mode standardization
        df = self.standardize_payment_modes(df)
        
        # Step 7: Outlier detection and handling
        df = self.detect_and_handle_outliers(df)
        
        # Step 6: Final data validation
        logger.info("✅ Final Data Validation & Quality Check...")
        
        # Remove any remaining rows with critical missing values
        critical_columns = ['date', 'amount', 'category']
        before_critical_cleaning = len(df)
        df = df.dropna(subset=critical_columns)
        critical_cleaned = before_critical_cleaning - len(df)
        
        if critical_cleaned > 0:
            logger.info(f"   • Removed {critical_cleaned:,} records with missing critical data")
        
        # Final statistics
        final_shape = df.shape
        data_retention = (final_shape[0] / initial_shape[0]) * 100
        
        logger.info(f"\n🏁 Comprehensive Cleaning Complete!")
        logger.info(f"=" * 60)
        logger.info(f"📊 Final Dataset Statistics:")
        logger.info(f"   • Initial records: {initial_shape[0]:,}")
        logger.info(f"   • Final records: {final_shape[0]:,}")
        logger.info(f"   • Data retention rate: {data_retention:.2f}%")
        logger.info(f"   • Columns: {final_shape[1]}")
        
        if len(df) > 0:
            logger.info(f"   • Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            logger.info(f"   • Amount range: ₹{df['amount'].min():,.2f} to ₹{df['amount'].max():,.2f}")
            logger.info(f"   • Categories: {df['category'].nunique()} unique")
            logger.info(f"   • Total transaction value: ₹{df['amount'].sum():,.2f}")
            
            if 'user_id' in df.columns:
                logger.info(f"   • Unique users: {df['user_id'].nunique():,}")
        
        # Store processed data
        self.processed_data = df
        
        return df

    def prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare cleaned data for time series forecasting.
        
        Args:
            df: Cleaned transaction data
            
        Returns:
            Time series ready data
        """
        logger.info("📈 Preparing Time Series Data for Forecasting...")
        logger.info("=" * 50)
        
        # Create daily aggregations
        logger.info("🎯 Creating daily expense aggregations...")
        
        # Group by date and category for daily sums
        daily_data = df.groupby(['date', 'category']).agg({
            'amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        daily_data.columns = ['date', 'category', 'daily_expense', 'transaction_count']
        
        # Create pivot table for category-wise time series
        pivot_data = daily_data.pivot(index='date', columns='category', values='daily_expense')
        pivot_data = pivot_data.fillna(0)
        
        # Add total daily expense
        pivot_data['total_daily_expense'] = pivot_data.sum(axis=1)
        
        # Reset index to make date a column
        pivot_data = pivot_data.reset_index()
        
        # Sort by date
        pivot_data = pivot_data.sort_values('date')
        
        # CRITICAL FIX: Daily Aggregation Outlier Detection & Treatment
        logger.info(f"\n🎯 Daily Aggregation Outlier Detection & Treatment")
        
        # Analyze daily totals before treatment
        daily_totals = pivot_data['total_daily_expense']
        logger.info(f"   📊 Daily expense analysis before treatment:")
        logger.info(f"      • Mean daily expense: ₹{daily_totals.mean():,.2f}")
        logger.info(f"      • Median daily expense: ₹{daily_totals.median():,.2f}")
        logger.info(f"      • 95th percentile: ₹{daily_totals.quantile(0.95):,.2f}")
        logger.info(f"      • Max daily expense: ₹{daily_totals.max():,.2f}")
        
        # REMOVED: Daily expense capping to preserve natural expense patterns
        # This allows the dashboard to show realistic variations instead of artificial ₹50k ceiling
        logger.info(f"   ✅ Preserving natural daily expense patterns (no artificial capping)")
        logger.info(f"      • Max daily expense: ₹{pivot_data['total_daily_expense'].max():,.2f}")
        logger.info(f"      • Mean daily expense: ₹{pivot_data['total_daily_expense'].mean():,.2f}")
        logger.info(f"      • 95th percentile: ₹{pivot_data['total_daily_expense'].quantile(0.95):,.2f}")
        
        # Add tracking columns for consistency (but no capping applied)
        pivot_data['daily_expense_capped'] = 0
        pivot_data['original_daily_expense'] = pivot_data['total_daily_expense']
        
        logger.info(f"✅ Time series data prepared:")
        logger.info(f"   • Date range: {pivot_data['date'].min().strftime('%Y-%m-%d')} to {pivot_data['date'].max().strftime('%Y-%m-%d')}")
        logger.info(f"   • Total days: {len(pivot_data):,}")
        logger.info(f"   • Categories: {len([col for col in pivot_data.columns if col not in ['date', 'total_daily_expense']])} + total")
        
        return pivot_data

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data into train, validation, and test sets.
        
        Args:
            df: Time series data
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        logger.info("🔄 Splitting data for training/validation/testing...")
        
        # Sort by date to ensure temporal order
        df_sorted = df.sort_values('date')
        
        # Calculate split points
        total_len = len(df_sorted)
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        train_idx = int(total_len * train_split)
        val_idx = int(total_len * (train_split + val_split))
        
        # Split data
        train_data = df_sorted[:train_idx].copy()
        val_data = df_sorted[train_idx:val_idx].copy()
        test_data = df_sorted[val_idx:].copy()
        
        logger.info(f"✅ Data split completed:")
        logger.info(f"   • Training: {len(train_data):,} days ({len(train_data)/total_len*100:.1f}%)")
        logger.info(f"   • Validation: {len(val_data):,} days ({len(val_data)/total_len*100:.1f}%)")
        logger.info(f"   • Testing: {len(test_data):,} days ({len(test_data)/total_len*100:.1f}%)")
        
        return train_data, val_data, test_data

    def save_processed_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                          test_data: pd.DataFrame, processed_raw: Optional[pd.DataFrame] = None) -> None:
        """
        Save all processed data to files.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset  
            test_data: Test dataset
            processed_raw: Optional cleaned raw data
        """
        logger.info("💾 Saving processed data files...")
        
        # Create processed directory
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Save time series splits
        train_data.to_csv(self.processed_data_path / "train_data.csv", index=False)
        val_data.to_csv(self.processed_data_path / "val_data.csv", index=False)
        test_data.to_csv(self.processed_data_path / "test_data.csv", index=False)
        
        logger.info("✅ Time series data saved:")
        logger.info(f"   • {self.processed_data_path / 'train_data.csv'}")
        logger.info(f"   • {self.processed_data_path / 'val_data.csv'}")
        logger.info(f"   • {self.processed_data_path / 'test_data.csv'}")
        
        # Save cleaned raw data if provided
        if processed_raw is not None:
            processed_raw.to_csv(self.processed_data_path / "cleaned_transactions.csv", index=False)
            logger.info(f"   • {self.processed_data_path / 'cleaned_transactions.csv'}")
        
        # Save data summary
        summary = {
            'processing_date': datetime.now().isoformat(),
            'datasets_processed': list(self.datasets.keys()) if self.datasets else [],
            'total_original_records': sum(df.shape[0] for df in self.datasets.values()) if self.datasets else 0,
            'final_clean_records': len(processed_raw) if processed_raw is not None else 0,
            'time_series_days': len(train_data) + len(val_data) + len(test_data),
            'train_days': len(train_data),
            'val_days': len(val_data),
            'test_days': len(test_data)
        }
        
        import json
        with open(self.processed_data_path / "processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"   • {self.processed_data_path / 'processing_summary.json'}")
        logger.info("💾 All processed data saved successfully!")

    def run_complete_preprocessing_pipeline(self) -> None:
        """
        Execute the complete advanced preprocessing pipeline.
        """
        logger.info("🚀 Starting Complete Advanced Preprocessing Pipeline...")
        logger.info("=" * 70)
        
        try:
            # Step 1: Load multiple datasets
            self.load_multiple_datasets()
            
            # Step 2: Merge datasets intelligently
            combined_data = self.merge_datasets()
            
            # Step 3: Comprehensive data cleaning
            cleaned_data = self.comprehensive_data_cleaning(combined_data)
            
            # Step 4: Prepare time series data
            time_series_data = self.prepare_time_series_data(cleaned_data)
            
            # Step 5: Split data for modeling
            train_data, val_data, test_data = self.split_data(time_series_data)
            
            # Step 6: Save all processed data
            self.save_processed_data(train_data, val_data, test_data, cleaned_data)
            
            # Final success message
            logger.info("\n🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            logger.info("✅ All data has been processed, cleaned, and prepared for modeling")
            logger.info(f"📁 Processed data saved to: {self.processed_data_path}")
            logger.info("🚀 Ready for feature engineering and model training!")
            
        except Exception as e:
            logger.error(f"❌ Preprocessing pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the advanced preprocessing pipeline."""
    try:
        # Initialize preprocessor
        preprocessor = AdvancedDataPreprocessor()
        
        # Run complete pipeline
        preprocessor.run_complete_preprocessing_pipeline()
        
    except Exception as e:
        logger.error(f"Failed to run preprocessing pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()