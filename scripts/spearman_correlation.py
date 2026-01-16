import pandas as pd
import numpy as np
from scipy.stats import spearmanr

mushra_df = pd.read_excel(r'results\qualtrics_export_December 7, 2025_15.22.xlsx')
bss_df = pd.read_excel(r'results\bss_eval_results_processed_wkbk.xlsx')
fad_df = pd.read_excel(r'results\fad_scores_workbook.xlsx')


print(mushra_df.head())
print(mushra_df.columns)

metadata_cols = ['Playback Device', 'Environment_Quality_1', 'MSS_Familiarity_1',
       'Assessment_Familiari_1']

rating_cols = [col for col in mushra_df.columns if col not in metadata_cols]

print(f'Found {len(rating_cols)} rating columns!')
print(rating_cols)

trial_data = []

track_map = {
    'NG': 'nogravity',
    'Monstaclat': 'monstaclat',
    'Celebrate': 'celebrate',
    'JG': 'jackiesgarage',
    'TF': 'thisfeeling',
    'DropNoir': 'dropnoir'
}

model_map = {
    'Spl': 'spleeter',
    'Anch': 'anchor',
    'HTD': 'htdemucs',
    'Ref': 'reference',
    'Dv2': 'dv2'
}

for col in rating_cols:
    col_name = col.split('_')
    track = track_map.get(col_name[0], col_name[0].lower())
    stem = col_name[1].lower()
    model = model_map.get(col_name[2], col_name[2].lower())
    
    mean_score = mushra_df[col].mean()
    
    trial_data.append({
        'track': f"{track}_{stem}",
        'model': model,
        'mushra_mean': mean_score
    })

trial_df = pd.DataFrame(trial_data)

print(trial_df.head(10))

# ============================================================
# PART 2: PREPARE SDR DATA (TRIAL-LEVEL)
# ============================================================

bss_df['track'] = bss_df['reference_file'].str.replace('.wav', '')
bss_df['model'] = bss_df['model'].str.lower()

# Merge MUSHRA with SDR
trial_level = trial_df.merge(bss_df[['track', 'model', 'SDR']], 
                              on=['track', 'model'], 
                              how='inner')

print(f"Matched {len(trial_level)} trials for trial-level analysis")
print(trial_level.head(30))

# ============================================================
# PART 3: TRIAL-LEVEL CORRELATION (MUSHRA vs SDR)
# ============================================================

print(f"\nOriginal trial count: {len(trial_level)}")

# Exclude reference and anchor
trial_level_models_only = trial_level[~trial_level['model'].isin(['reference', 'anchor'])]

print(f"After excluding reference and anchor: {len(trial_level_models_only)}")
print(f"Models included: {trial_level_models_only['model'].unique()}")

# Now run correlation on filtered data
corr_trial_sdr, p_trial_sdr = spearmanr(trial_level_models_only['mushra_mean'], 
                                         trial_level_models_only['SDR'])

print("\n" + "="*60)
print("TRIAL-LEVEL ANALYSIS - MODELS ONLY (n={})".format(len(trial_level_models_only)))
print("="*60)
print(f"\nMUSHRA vs SDR (htdemucs, dv2, spleeter only):")
print(f"  Spearman ρ = {corr_trial_sdr:.4f}")
print(f"  p-value = {p_trial_sdr:.6f}")
print(f"  p-value (scientific): {p_trial_sdr:.4e}")
if p_trial_sdr < 0.05:
    print("  ✓ Statistically significant at α = 0.05")

import matplotlib.pyplot as plt

# Create the plot using only the filtered data (models only)
plt.figure(figsize=(10, 6))

# Define colors for the three models
colors = {
    'htdemucs': 'green',
    'dv2': 'blue', 
    'spleeter': 'orange'
}

# Plot each model with its own color
for model in trial_level_models_only['model'].unique():
    model_data = trial_level_models_only[trial_level_models_only['model'] == model]
    plt.scatter(model_data['SDR'], model_data['mushra_mean'], 
               label=model, s=100, alpha=0.7, color=colors.get(model, 'gray'))

plt.xlabel('SDR (dB)', fontsize=12)
plt.ylabel('Mean MUSHRA Score', fontsize=12)
plt.title(f'Trial-Level: MUSHRA vs SDR - Models Only\n(ρ = {corr_trial_sdr:.3f}, p = {p_trial_sdr:.4f}, n={len(trial_level_models_only)})', 
         fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('analysis/mushra_sdr_correlation_models_only.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'mushra_sdr_correlation_models_only.png'")


# Model Level Spearman

model_mushra = trial_df.groupby('model')['mushra_mean'].mean().reset_index()
model_mushra.columns = ['model', 'avg_mushra']
model_mushra['model'] = model_mushra['model'].str.lower()

print('\n Model-Level MUSHRA Scores:')
print(model_mushra)


print('\n Extracting BSS Eval Data...')
print(bss_df[['reference_file', 'model', 'SDR']].head(10))

model_sdr = bss_df.groupby('model')['SDR'].mean().reset_index()
model_sdr.columns = ['model', 'avg_sdr']


print('\n Extracting FAD Data...')
print(fad_df[['model', 'FAD (pann)', 'FAD (vggish)']])

model_metrics = model_mushra.merge(model_sdr, on='model')
model_metrics = model_metrics.merge(fad_df, on = 'model')

print('\n Combining Model Metrics')
print(model_metrics)

print("\n" + "="*60)
print("MODEL-LEVEL SPEARMAN CORRELATIONS")
print("="*60)

model_metrics_only = model_metrics[~model_metrics['model'].isin(['reference', 'anchor'])]

print("\n" + "="*60)
print("MODEL-LEVEL ANALYSIS - MODELS ONLY (n={})".format(len(model_metrics_only)))
print("="*60)

print("\nModel-Level Metrics (htdemucs, dv2, spleeter only):")
print(model_metrics_only)

summary_data = []

# MUSHRA vs SDR (models only)
corr_model_sdr, p_model_sdr = spearmanr(model_metrics_only['avg_mushra'], 
                                         model_metrics_only['avg_sdr'])

print(f"\nMUSHRA vs SDR:")
print(f"  Spearman ρ = {corr_model_sdr:.4f}")
print(f"  p-value = {p_model_sdr:.4f}")

summary_data.append({
    'Comparison': 'MUSHRA vs SDR',
    'Level': 'Model',
    'n': len(model_metrics_only),
    'Spearman ρ': corr_model_sdr,
    'p-value': p_model_sdr
})

# MUSHRA vs FAD (models only)
corr_fad_pann, p_fad_pann = spearmanr(model_metrics_only['avg_mushra'], 
                                       model_metrics_only['FAD (pann)'])

print(f"\nMUSHRA vs FAD (PANN):")
print(f"  Spearman ρ = {corr_fad_pann:.4f}")
print(f"  p-value = {p_fad_pann:.4f}")

summary_data.append({
    'Comparison': 'MUSHRA vs FAD (PANN)',
    'Level': 'Model',
    'n': len(model_metrics_only),
    'Spearman ρ': corr_fad_pann,
    'p-value': p_fad_pann
})

corr_fad_vgg, p_fad_vgg = spearmanr(model_metrics_only['avg_mushra'], 
                                     model_metrics_only['FAD (vggish)'])

print(f"\nMUSHRA vs FAD (VGGish):")
print(f"  Spearman ρ = {corr_fad_vgg:.4f}")
print(f"  p-value = {p_fad_vgg:.4f}")

summary_data.append({
    'Comparison': 'MUSHRA vs FAD (VGGish)',
    'Level': 'Model',
    'n': len(model_metrics_only),
    'Spearman ρ': corr_fad_vgg,
    'p-value': p_fad_vgg
})

# SDR vs FAD (PANN)

corr_sdr_fad_pann, p_sdr_fad_pann = spearmanr(model_metrics_only['avg_sdr'], 
                                               model_metrics_only['FAD (pann)'])
summary_data.append({
    'Comparison': 'SDR vs FAD (PANN)',
    'Level': 'Model',
    'n': len(model_metrics_only),
    'Spearman ρ': corr_sdr_fad_pann,
    'p-value': p_sdr_fad_pann
})

print(f"\n SDR vs FAD (PANN):")
print(f"  Spearman ρ = {corr_sdr_fad_pann:.4f}")
print(f"  p-value = {p_sdr_fad_pann:.4f}")

# SDR vs FAD (VGGish)
corr_sdr_fad_vgg, p_sdr_fad_vgg = spearmanr(model_metrics_only['avg_sdr'], 
                                             model_metrics_only['FAD (vggish)'])

print(f"\n SDR vs FAD (VGGish):")
print(f"  Spearman ρ = {corr_sdr_fad_vgg:.4f}")
print(f"  p-value = {p_sdr_fad_vgg:.4f}")

summary_data.append({
    'Comparison': 'SDR vs FAD (VGGish)',
    'Level': 'Model',
    'n': len(model_metrics_only),
    'Spearman ρ': corr_sdr_fad_vgg,
    'p-value': p_sdr_fad_vgg
})


summary_df = pd.DataFrame(summary_data)

# Format the table nicely
summary_df['Spearman ρ'] = summary_df['Spearman ρ'].apply(lambda x: f'{x:.4f}')
summary_df['p-value'] = summary_df['p-value'].apply(lambda x: f'{x:.4e}' if x < 0.001 else f'{x:.4f}')

summary_df.to_csv('analysis/correlation_summary_table.csv', index=False)
print("\n✓ Table saved as 'correlation_summary_table.csv'")

# Also save to Excel with better formatting
with pd.ExcelWriter('analysis/correlation_summary_table.xlsx', engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
print("✓ Table saved as 'correlation_summary_table.xlsx'")