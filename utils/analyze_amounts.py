import pandas as pd
import numpy as np

print('=== ANALYZING RAW DATA AMOUNTS ===')

# Check budgetwise_finance_dataset.csv
print('\n1. budgetwise_finance_dataset.csv:')
df1 = pd.read_csv('data/raw/budgetwise_finance_dataset.csv')
print(f'Amount column dtype: {df1["amount"].dtype}')
print(f'Sample amount values: {df1["amount"].head(10).tolist()}')

# Try to convert to numeric
df1["amount_clean"] = pd.to_numeric(df1["amount"], errors="coerce")
valid_amounts = df1["amount_clean"].dropna()
print(f'Valid amounts after conversion: {len(valid_amounts)}/{len(df1)}')
print(f'Min: ₹{valid_amounts.min():,.2f}')
print(f'Max: ₹{valid_amounts.max():,.2f}')
print(f'Mean: ₹{valid_amounts.mean():,.2f}')
print(f'Median: ₹{valid_amounts.median():,.2f}')

# Check for currency symbols or formatting issues
print(f'Non-numeric amount samples: {df1[df1["amount_clean"].isna()]["amount"].head().tolist()}')

print('\n' + '='*60)

# Check synthetic dirty
print('\n2. budgetwise_synthetic_dirty.csv:')
df2 = pd.read_csv('data/raw/budgetwise_synthetic_dirty.csv')
print(f'Amount column dtype: {df2["amount"].dtype}')
print(f'Sample amount values: {df2["amount"].head(10).tolist()}')

df2["amount_clean"] = pd.to_numeric(df2["amount"], errors="coerce")
valid_amounts2 = df2["amount_clean"].dropna()
print(f'Valid amounts after conversion: {len(valid_amounts2)}/{len(df2)}')
print(f'Min: ₹{valid_amounts2.min():,.2f}')
print(f'Max: ₹{valid_amounts2.max():,.2f}')
print(f'Mean: ₹{valid_amounts2.mean():,.2f}')
print(f'Median: ₹{valid_amounts2.median():,.2f}')

print('\n' + '='*60)

# Check processed data to see the aggregation
print('\n3. PROCESSED DATA ANALYSIS:')
train = pd.read_csv('data/processed/train_data.csv')
print(f'Processed train data shape: {train.shape}')
print(f'Columns: {list(train.columns)}')

if 'total_daily_expense' in train.columns:
    print(f'\nTotal daily expense stats:')
    print(f'Min: ₹{train["total_daily_expense"].min():,.2f}')
    print(f'Max: ₹{train["total_daily_expense"].max():,.2f}')
    print(f'Mean: ₹{train["total_daily_expense"].mean():,.2f}')
    print(f'Median: ₹{train["total_daily_expense"].median():,.2f}')
    
    # Show some sample daily totals
    print(f'\nSample daily expenses:')
    for i, row in train.head(10).iterrows():
        print(f'{row["date"]}: ₹{row["total_daily_expense"]:,.2f}')

print('\n' + '='*60)
print('\n4. DAILY AGGREGATION ANALYSIS:')
# Check if individual expenses are being summed incorrectly
print('Raw data individual transaction amounts vs processed daily totals...')
print(f'Raw dataset 1 mean transaction: ₹{valid_amounts.mean():,.2f}')
print(f'Raw dataset 2 mean transaction: ₹{valid_amounts2.mean():,.2f}')
print(f'Processed mean daily total: ₹{train["total_daily_expense"].mean():,.2f}')

# Calculate expected daily amounts
transactions_per_day_estimate = len(df1) / 365  # rough estimate
print(f'Estimated transactions per day: {transactions_per_day_estimate:.1f}')
expected_daily = valid_amounts.mean() * transactions_per_day_estimate
print(f'Expected daily total (rough): ₹{expected_daily:,.2f}')