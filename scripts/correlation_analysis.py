import pandas as pd
import numpy as np
from scipy import stats

print("="*60)
print("CORRELATION: MUSHRA vs OBJECTIVE METRICS")
print("="*60)

df_bss = pd.read_csv('results/bss_eval_results_processed.csv')

# Calculate average SDR per model (excluding reference)
sdr_by_model = df_bss[df_bss['model']].groupby('model')['SDR'].mean()

print("\nAverage SDR by model:")
print(sdr_by_model.sort_values(ascending=False))