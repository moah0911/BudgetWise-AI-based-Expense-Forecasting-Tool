import pandas as pd
import numpy as np

# Load the processed data
combined_data = pd.concat([
    pd.read_csv('data/processed/train_data.csv'),
    pd.read_csv('data/processed/val_data.csv'),
    pd.read_csv('data/processed/test_data.csv')
])

print('=== DATA STATISTICS ===')
print(f'Total records: {len(combined_data)}')
print('\n=== TOTAL_DAILY_EXPENSE STATS ===')
print(combined_data['total_daily_expense'].describe())

print('\n=== EXTREME VALUES ===')
min_val = combined_data['total_daily_expense'].min()
max_val = combined_data['total_daily_expense'].max()
p95 = combined_data['total_daily_expense'].quantile(0.95)
p99 = combined_data['total_daily_expense'].quantile(0.99)

print(f'Min: Rs {min_val:,.2f}')
print(f'Max: Rs {max_val:,.2f}')
print(f'95th percentile: Rs {p95:,.2f}')
print(f'99th percentile: Rs {p99:,.2f}')

print('\n=== TOP 10 HIGHEST VALUES ===')
top_10 = combined_data.nlargest(10, 'total_daily_expense')[['date', 'total_daily_expense']]
for _, row in top_10.iterrows():
    print(f'{row["date"]}: Rs {row["total_daily_expense"]:,.2f}')

print('\n=== OUTLIER ANALYSIS ===')
Q1 = combined_data['total_daily_expense'].quantile(0.25)
Q3 = combined_data['total_daily_expense'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = combined_data[(combined_data['total_daily_expense'] < lower_bound) | 
                        (combined_data['total_daily_expense'] > upper_bound)]

print(f'IQR-based outlier count: {len(outliers)}')
print(f'Outlier percentage: {len(outliers)/len(combined_data)*100:.2f}%')
print(f'Upper bound for normal values: Rs {upper_bound:,.2f}')