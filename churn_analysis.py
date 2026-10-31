import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load the dataset
file_name = 'churn_dataset.csv'
df = pd.read_csv(file_name)

print('--- 1. Dataset Preview ---')
print(df.head(3))
print(f'Total Customers: {len(df)}')

# Standardize column headers to lowercase
df.columns = df.columns.str.strip().str.lower()


# Helper function for exact column matching
def find_col(possible_names):
  for name in possible_names:
    if name in df.columns:
      return name
  return None


# Explicitly map exact column names from your dataset
churn_col = 'churn'
tenure_col = 'tenuremonths'
monthly_col = 'monthlycharges'
contract_col = 'contracttype'
payment_col = 'paymentmethod'

print(
    f'\nMapped Columns -> Churn: {churn_col}, Tenure: {tenure_col}, Monthly:'
    f' {monthly_col}, Contract: {contract_col}, Payment: {payment_col}'
)

# 2. Calculate Key Retention & Churn Metrics
if churn_col in df.columns:
  df['churn_numeric'] = df[churn_col].apply(
      lambda x: 1 if str(x).strip().lower() in ['yes', 'true', '1', 'churn'] else 0
  )
else:
  raise ValueError("Could not find 'churn' column.")

total_customers = len(df)
churned_customers = int(df['churn_numeric'].sum())
churn_rate = (churned_customers / total_customers) * 100

print('\n--- 2. Core Business Metrics ---')
print(f'Total Churned Customers: {churned_customers}')
print(f'Baseline Churn Rate: {churn_rate:.2f}%')

# Monthly Recurring Revenue (MRR) Loss
if monthly_col in df.columns:
  df[monthly_col] = pd.to_numeric(df[monthly_col], errors='coerce').fillna(0)
  mrr_loss = df[df['churn_numeric'] == 1][monthly_col].sum()
  print(f'Monthly Recurring Revenue (MRR) Loss: ${mrr_loss:,.2f}')

# Customer Lifetime Value (CLV) approximation
if monthly_col in df.columns and tenure_col in df.columns:
  df[tenure_col] = pd.to_numeric(df[tenure_col], errors='coerce').fillna(0)
  arpu = df[monthly_col].mean()
  avg_tenure = df[tenure_col].mean()
  estimated_clv = arpu * avg_tenure
  print(f'Estimated Customer Lifetime Value (CLV): ${estimated_clv:,.2f}')

# 3. Group Churn Rate by Key Dimensions
print('\n--- 3. Churn Breakdown by Category ---')
for col in [contract_col, payment_col]:
  if col in df.columns:
    print(f'\nChurn Rate by {col.title()}:')
    breakdown = (
        df.groupby(col)['churn_numeric'].mean() * 100
    ).reset_index()
    breakdown.columns = [col, 'churn_rate_percentage']
    print(breakdown)

# 4. Generate & Save Visualization Charts
sns.set_theme(style='whitegrid')
plt.figure(figsize=(12, 5))

# Chart 1: Churn by Contract Type
if contract_col in df.columns:
  plt.subplot(1, 2, 1)
  sns.barplot(
      x=contract_col,
      y='churn_numeric',
      data=df,
      estimator=lambda x: sum(x) / len(x) * 100,
      hue=contract_col,
      palette='Blues_d',
      legend=False,
  )
  plt.title('Churn Rate (%) by Contract Type')
  plt.ylabel('Churn Rate (%)')
  plt.xlabel('Contract Type')
  plt.xticks(rotation=15)

# Chart 2: Tenure vs Churn Boxplot
if tenure_col in df.columns and contract_col in df.columns:
  plt.subplot(1, 2, 2)
  sns.boxplot(
      x='churn_numeric',
      y=tenure_col,
      hue='churn_numeric',
      data=df,
      palette='Set2',
      legend=False,
  )
  plt.title('Tenure Distribution by Churn Status')
  plt.ylabel('Tenure (Months)')
  plt.xlabel('Churn Status (0 = Retained, 1 = Churned)')

plt.tight_layout()
plt.savefig('churn_analysis_charts.png')
print(
    '\nSuccess! Updated visualizations saved as "churn_analysis_charts.png" in'
    ' your folder.'
)