import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\sahan\OneDrive\Documents\Python\MUSHRA\results\qualtrics_export_December 7, 2025_15.22.xlsx")

print(df.head())
print(df.columns)

metadata_cols = ['Playback Device', 'Environment_Quality_1', 'MSS_Familiarity_1',
       'Assessment_Familiari_1']

rating_cols = [col for col in df.columns if col not in metadata_cols]

print(f'Found {len(rating_cols)} rating columns!')
print(rating_cols)

tracks = []
models = []

for col in rating_cols:
    col_name = col.split('_')
    if col_name[2] == 'Spl':
        col_name[2] = 'Spleeter'
    if col_name[2] == 'Anch':
        col_name[2] = 'Anchor'
    if col_name[2] == 'HTD':
        col_name[2] = 'HTDemucs'
    if col_name[2] == 'Ref':
        col_name[2] = 'Reference'             
    if col_name[0] not in tracks:
        tracks.append(col_name[0] + '_' + col_name[1])
    if col_name[2] not in models:   
        models.append(col_name[2])

pass

print('Unique tracks:', set(tracks))
print('Unique models:', set(models))

model_ratings = {
    'htdemucs': [],
    'reference': [],
    'spleeter': [],
    'anchor': [],
    'dv2': []
}

for col in rating_cols:
    col_name = col.split('_')
    
    # Standardize the model name
    if col_name[2] in ['Spl', 'Spleeter']:
        model = 'spleeter'
    elif col_name[2] in ['Anch', 'Anchor']:
        model = 'anchor'
    elif col_name[2] in ['HTD', 'HTDemucs']:
        model = 'htdemucs'
    elif col_name[2] in ['Ref', 'Reference']:  
        model = 'reference'
    elif col_name[2] in ['Dv2']:  
        model = 'dv2'
    else:
        print(f"Warning: Unknown model '{col_name[2]}' in column {col}")
        continue
    
    # Get all ratings for this column
    ratings = df[col].values
    
    # Add to the appropriate list
    model_ratings[model].extend(ratings)

# Calculate and print averages
print("="*60)
print("RAW AVERAGE RATINGS BY MODEL")
print("="*60)

for model, ratings in model_ratings.items():
    avg = np.mean(ratings)
    std = np.std(ratings)
    count = len(ratings)
    
    print(f"{model:12s}: Mean={avg:5.2f}, Std={std:5.2f}, N={count}")

print("\n" + "="*60)
print("SCALE USAGE PER PARTICIPANT")
print("="*60)

for participant_id in range(len(df)):
    # TODO: Get all ratings for this participant
    participant_ratings = df.iloc[participant_id][rating_cols].values
    
    # TODO: Calculate min, max, range, mean, std
    min_rating = np.min(participant_ratings)
    max_rating = np.max(participant_ratings)
    rating_range = max_rating - min_rating
    mean_rating = np.mean(participant_ratings)
    std_rating = np.std(participant_ratings)
    
    print(f"Participant {participant_id + 1}:")
    print(f"  Range: {min_rating:.0f} - {max_rating:.0f} (span: {rating_range:.0f})")
    print(f"  Mean: {mean_rating:.2f}, Std: {std_rating:.2f}")

def z_score_normalize_participant(ratings):
    mean_rating = np.mean(ratings)
    std_rating = np.std(ratings)
    if std_rating == 0:
        return ratings - mean_rating
    z_scores = (ratings - mean_rating)/std_rating
    return z_scores

print("\n" + "="*60)
print("WEIGHTED Z-SCORES PER PARTICIPANT")
print("="*60)

# Create a list to store all normalized data
normalized_data = []

print("="*60)
print("NORMALIZING ALL PARTICIPANTS")
print("="*60)

# Loop through each participant
for participant_id in range(len(df)):
    print(f"\nProcessing Participant {participant_id + 1}...")
    
    # TODO 1: Get all ratings for this participant
    participant_ratings = df.iloc[participant_id][rating_cols].values
    
    # TODO 2: Apply your normalization function
    normalized_ratings = z_score_normalize_participant(participant_ratings)
    
    # Check: Print before/after for first participant
    # if participant_id == 0:
    #     print(f"  Original ratings (first 5): {participant_ratings[:5]}")
    #     print(f"  Normalized (first 5): {np.round(normalized_ratings[:5], 2)}")
    #     print(f"  Mean of normalized: {np.mean(normalized_ratings):.10f}")
    #     print(f"  Std of normalized: {np.std(normalized_ratings):.10f}")
    
    # TODO 3: For each rating, store the data with track/model info
    for col_idx, col in enumerate(rating_cols):
        # Parse column to get track and model
        col_name = col.split('_')
        
        # Extract track name
        track = col_name[0] + '_' + col_name[1]
        
        # Extract model name
        if col_name[2] in ['HTD', 'HTDemucs']:
            model = 'htdemucs'
        elif col_name[2] in ['Ref', 'Reference']:
            model = 'reference'
        elif col_name[2] in ['Spl', 'Spleeter']:
            model = 'spleeter'
        elif col_name[2] in ['Anch', 'Anchor']:
            model = 'anchor'
        elif col_name[2] in ['Dv2']:
            model = 'dv2'
        else:
            print(f"Warning: Unknown model in {col}")
            model = 'unknown'
        
        # Store the data
        normalized_data.append({
            'participant': participant_id + 1,
            'track': track,
            'model': model,
            'rating_raw': participant_ratings[col_idx],
            'rating_zscore': normalized_ratings[col_idx]
        })

print("\n✅ Normalization complete!")
print(f"Total data points: {len(normalized_data)}")

# Convert to DataFrame for easier analysis
df_normalized = pd.DataFrame(normalized_data)

print("\n" + "="*60)
print("STEP 8: NORMALIZED AVERAGES BY MODEL")
print("="*60)

# TODO 1: Calculate average z-score for each model
# Hint: Use groupby on df_normalized

normalized_averages = df_normalized.groupby('model')['rating_zscore'].agg(['mean', 'std', 'count'])

# Sort by mean (highest to lowest)
normalized_averages = normalized_averages.sort_values('mean', ascending=False)

print("\nZ-Score Normalized Averages:")
print(normalized_averages)

# TODO 2: Create a comparison table
# Compare raw averages (from Step 4) with normalized averages

print("\n" + "="*60)
print("COMPARISON: RAW vs NORMALIZED")
print("="*60)

# You'll need to manually fill in the raw averages from Step 4
# Or calculate them again from df_normalized using 'rating_raw'

raw_averages = df_normalized.groupby('model')['rating_raw'].agg(['mean', 'std'])
raw_averages = raw_averages.sort_values('mean', ascending=False)

# Create comparison
comparison = pd.DataFrame({
    'Raw_Mean': raw_averages['mean'],
    'Raw_Std': raw_averages['std'],
    'Zscore_Mean': normalized_averages['mean'],
    'Zscore_Std': normalized_averages['std']
})

print(comparison.round(2))

# TODO 3: Check if rankings changed
print("\n" + "="*60)
print("RANKING COMPARISON")
print("="*60)

raw_ranking = raw_averages.index.tolist()
zscore_ranking = normalized_averages.index.tolist()

print("Raw Rankings:")
for i, model in enumerate(raw_ranking, 1):
    print(f"  {i}. {model}")

print("\nZ-Score Rankings:")
for i, model in enumerate(zscore_ranking, 1):
    print(f"  {i}. {model}")

if raw_ranking == zscore_ranking:
    print("\n✅ Rankings are IDENTICAL! Normalization preserved the order.")
else:
    print("\n⚠️ Rankings changed after normalization!")



# Get ratings for all models (z-scored)
htdemucs_ratings = df_normalized[df_normalized['model'] == 'htdemucs']['rating_zscore'].values
reference_ratings = df_normalized[df_normalized['model'] == 'reference']['rating_zscore'].values
spleeter_ratings = df_normalized[df_normalized['model'] == 'spleeter']['rating_zscore'].values
dv2_ratings = df_normalized[df_normalized['model'] == 'dv2']['rating_zscore'].values


print("="*60)
print("STATISTICAL TEST: htdemucs vs reference")
print("="*60)
print(f"Sample size: {len(htdemucs_ratings)} ratings each")
print(f"\nhtdemucs:  Mean={np.mean(htdemucs_ratings):.3f}, SD={np.std(htdemucs_ratings):.3f}")
print(f"reference: Mean={np.mean(reference_ratings):.3f}, SD={np.std(reference_ratings):.3f}")
print(f"Difference: {np.mean(htdemucs_ratings) - np.mean(reference_ratings):.3f}")

# Paired t-test (same participants rated both)
t_stat, p_value = stats.ttest_rel(htdemucs_ratings, reference_ratings)

print(f"\nPaired t-test results:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  degrees of freedom: {len(htdemucs_ratings) - 1}")

# Interpretation
if p_value < 0.05:
    print(f"\n✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
    if t_stat > 0:
        print("   → htdemucs rated significantly HIGHER than reference")
    else:
        print("   → reference rated significantly HIGHER than htdemucs")
else:
    print(f"\n⚠️ NOT statistically significant (p ≥ 0.05)")
    print("   → No reliable difference between htdemucs and reference")
    print("   → They're perceived as similar quality")

# More t-tests

print("="*60)
print("MORE STATISTICAL TESTS: hybrid-domain vs single-domain models")
print("="*60)

t_stat_spl, p_value_spl = stats.ttest_rel(htdemucs_ratings, spleeter_ratings)
print(f"\nPaired t-test results (htdemucs v. spleeter):")
print(f"  t-statistic: {t_stat_spl:.3f}")
print(f"  p-value: {p_value_spl:.4f}")
print(f"  degrees of freedom: {len(htdemucs_ratings) - 1}")

if p_value_spl < 0.05:
    print(f"\n✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
    if t_stat_spl > 0:
        print("   → htdemucs significantly HIGHER than spleeter")
    else:
        print("   → spleeter rated significantly HIGHER than htdemucs")
else:
    print(f"\n⚠️ NOT statistically significant (p ≥ 0.05)")
    print("   → No reliable difference between htdemucs and spleeter")
    print("   → They're perceived as similar quality")

t_stat_dv2, p_value_dv2 = stats.ttest_rel(htdemucs_ratings, dv2_ratings)
print(f"\nPaired t-test results (htdemucs v. dv2):")
print(f"  t-statistic: {t_stat_dv2:.3f}")
print(f"  p-value: {p_value_dv2:.4f}")
print(f"  degrees of freedom: {len(htdemucs_ratings) - 1}")

if p_value_dv2 < 0.05:
    print(f"\n✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
    if t_stat_dv2 > 0:
        print("   → htdemucs rated significantly HIGHER than dv2")
    else:
        print("   → dv2 rated significantly HIGHER than htdemucs")
else:
    print(f"\n⚠️ NOT statistically significant (p ≥ 0.05)")
    print("   → No reliable difference between htdemucs and dv2")
    print("   → They're perceived as similar quality")

# Effect size (Cohen's d)
# mean_diff = np.mean(htdemucs_ratings - reference_ratings)
# std_diff = np.std(htdemucs_ratings - reference_ratings)
# cohens_d = mean_diff / std_diff

# print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
# if abs(cohens_d) < 0.2:
#     print("  → Small effect")
# elif abs(cohens_d) < 0.5:
#     print("  → Medium effect")
# else:
#     print("  → Large effect")

print("\n" + "="*60)
print("BREAKDOWN BY STEM TYPE")
print("="*60)

# Add stem type column
df_normalized['stem_type'] = df_normalized['track'].apply(
    lambda x: 'Drums' if 'Drums' in x else 'bass'
)

# Compare by stem type
for stem_type in ['Drums', 'bass']:
    print(f"\n{stem_type.upper()}:")
    
    stem_data = df_normalized[df_normalized['stem_type'] == stem_type]
    
    htd_stem = stem_data[stem_data['model'] == 'htdemucs']['rating_zscore']
    ref_stem = stem_data[stem_data['model'] == 'reference']['rating_zscore']
    
    print(f"  htdemucs:  {np.mean(htd_stem):.3f}")
    print(f"  reference: {np.mean(ref_stem):.3f}")
    print(f"  difference: {np.mean(htd_stem) - np.mean(ref_stem):.3f}")
    
    t_stat, p_value = stats.ttest_rel(htd_stem, ref_stem)
    print(f"  p-value: {p_value:.4f}", end="")
    if p_value < 0.05:
        print(" ✅ Significant")
    else:
        print(" ⚠️ Not significant")

  

print("Creating bar chart...")

# Calculate statistics
model_stats = df_normalized.groupby('model')['rating_zscore'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
model_stats['se'] = model_stats['std'] / np.sqrt(model_stats['count'])

# Calculate 95% CI
model_stats['ci'] = model_stats['se'] * 1.96

print("Model statistics:")
print(model_stats)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

models = model_stats.index
x_pos = np.arange(len(models))
means = model_stats['mean']
ses = model_stats['se']

bars = ax.bar(x_pos, means, yerr=ses, capsize=5,
               color='skyblue', edgecolor='black', linewidth=1.5,
               error_kw={'linewidth': 2, 'ecolor': 'black'})

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Z-Score', fontsize=14, fontweight='bold')
ax.set_title('MUSHRA Ratings by Model', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=12)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (mean, se) in enumerate(zip(means, ses)):
    ax.text(i, mean + se + 0.1, f'{mean:.2f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add caption with sample size
n_per_model = int(model_stats['count'].iloc[0])
ax.text(0.5, -0.15, f'Error bars represent standard error (n={n_per_model} per model)', 
        ha='center', va='top', transform=ax.transAxes, fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('results/mushra_barchart_se.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: results/mushra_barchart_se.png")
plt.show()

# ANOVA

print("\n" + "="*60)
print('5 x 2 ANOVA: MUSHRA Ratings by Model and Stem Type')
print("\n" + "="*60)

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# df_anova = df_normalized.copy() # 5 x 2 ANOVA
# df_anova = df_normalized[(df_normalized['model'] != 'reference')].copy() # 4 x 2 ANOVA
df_anova = df_normalized[(df_normalized['model'] != 'anchor') & (df_normalized['model'] != 'reference')].copy() # 3 x 2 ANOVA

df_anova['stem_type'] = df_anova['stem_type'].str.lower()

print(f'Total observations: {len(df_anova)}')
print(f'Models: {df_anova['model'].unique()}')
print(f'\nStem Types: {df_anova['stem_type'].unique()}')

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

print("\nMean Z-Score by Model:")
model_means_anova = df_anova.groupby('model')['rating_zscore'].agg(['mean', 'std', 'count'])
print(model_means_anova.sort_values('mean', ascending=False))

print("\nMean Z-Score by Stem Type:")
stem_means_anova = df_anova.groupby('stem_type')['rating_zscore'].agg(['mean', 'std', 'count'])
print(stem_means_anova)

print("\nMean Z-Score by Model × Stem Type:")
interaction_means_anova = df_anova.groupby(['model', 'stem_type'])['rating_zscore'].mean().unstack()
print(interaction_means_anova)

print("\n" + "="*60)
print("ANOVA RESULTS")
print("="*60)

model_anova = ols('rating_zscore ~ C(model) + C(stem_type) + C(model):C(stem_type)',
                  data = df_anova).fit()

anova_results = anova_lm(model_anova, typ = 2)

print(anova_results)

p_model = anova_results.loc['C(model)', 'PR(>F)']
p_stem = anova_results.loc['C(stem_type)', 'PR(>F)']
p_interaction = anova_results.loc['C(model):C(stem_type)', 'PR(>F)']

print(f"\nMain Effect of Model: F = {anova_results.loc['C(model)', 'F']:.2f}, p = {p_model:.4f}")
print(f"\nMain Effect of Stem Type: F = {anova_results.loc['C(stem_type)', 'F']:.2f}, p = {p_stem:.4f}")
print(f"\nInteraction (Model × Stem Type): F = {anova_results.loc['C(model):C(stem_type)', 'F']:.2f}, p = {p_interaction:.4f}")

# Visualization: Interaction plot
print("\n" + "="*60)
print("CREATING INTERACTION PLOT")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 6))

for stem in df_anova['stem_type'].unique():
    stem_data = df_anova[df_anova['stem_type'] == stem]
    means = stem_data.groupby('model')['rating_zscore'].mean().sort_values(ascending=False)
    ax.plot(range(len(means)), means.values, marker='o', linewidth=2, 
            markersize=10, label=stem.capitalize())

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Z-Score', fontsize=14, fontweight='bold')
ax.set_title('MUSHRA Interaction Plot: Model × Stem Type', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(means)))
ax.set_xticklabels(means.index, rotation=45, ha='right', fontsize=12)
ax.legend(title='Stem Type', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('results/mushra_interaction_plot.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/mushra_interaction_plot.png")
plt.show()

# Save ANOVA results
anova_summary_df = pd.DataFrame({
    'Effect': ['Model', 'Stem Type', 'Model × Stem Type'],
    'F_statistic': [
        anova_results.loc['C(model)', 'F'],
        anova_results.loc['C(stem_type)', 'F'],
        anova_results.loc['C(model):C(stem_type)', 'F']
    ],
    'p_value': [p_model, p_stem, p_interaction],
    'significant': [p_model < 0.05, p_stem < 0.05, p_interaction < 0.05]
})







# Saving results to .csv

raw_stats = []
for model, ratings in model_ratings.items():
    raw_stats.append({
        'model': model,
        'mean': np.mean(ratings),
        'std': np.std(ratings),
        'count': len(ratings)
    })

df_raw_stats = pd.DataFrame(raw_stats)

participant_stats = []
for participant_id in range(len(df)):
    participant_ratings = df.iloc[participant_id][rating_cols].values
    
    participant_stats.append({
        'participant': participant_id + 1,
        'min': np.nanmin(participant_ratings),
        'max': np.nanmax(participant_ratings),
        'range': np.nanmax(participant_ratings) - np.nanmin(participant_ratings),
        'mean': np.nanmean(participant_ratings),
        'std': np.nanstd(participant_ratings)
    })

df_participant_stats = pd.DataFrame(participant_stats)

ttest_results = pd.DataFrame({
    'comparison': ['htdemucs vs reference'],
    'htdemucs_mean': [np.mean(htdemucs_ratings)],
    'htdemucs_std': [np.std(htdemucs_ratings)],
    'reference_mean': [np.mean(reference_ratings)],
    'reference_std': [np.std(reference_ratings)],
    'mean_difference': [np.mean(htdemucs_ratings) - np.mean(reference_ratings)],
    't_statistic': [t_stat],
    'p_value': [p_value],
    'df': [len(htdemucs_ratings) - 1],
    'significant': [p_value < 0.05]
})

stem_breakdown = []
for stem_type in ['Drums', 'bass']:
    stem_data = df_normalized[df_normalized['stem_type'] == stem_type]
    
    htd_stem = stem_data[stem_data['model'] == 'htdemucs']['rating_zscore']
    ref_stem = stem_data[stem_data['model'] == 'reference']['rating_zscore']
    
    t_stat_stem, p_value_stem = stats.ttest_rel(htd_stem, ref_stem)
    
    stem_breakdown.append({
        'stem_type': stem_type,
        'htdemucs_mean': np.mean(htd_stem),
        'reference_mean': np.mean(ref_stem),
        'difference': np.mean(htd_stem) - np.mean(ref_stem),
        'p_value': p_value_stem,
        'significant': p_value_stem < 0.05
    })

df_stem_breakdown = pd.DataFrame(stem_breakdown)


with pd.ExcelWriter('analysis/qualtrics_analysis.xlsx', engine='openpyxl') as writer:
    df_raw_stats.to_excel(writer, sheet_name='Raw_Model_Averages', index=False)
    df_participant_stats.to_excel(writer, sheet_name='Participant_Stats', index=False)
    df_normalized.to_excel(writer, sheet_name='All_Normalized_Data', index=False)
    normalized_averages.to_excel(writer, sheet_name='Normalized_Averages')
    comparison.to_excel(writer, sheet_name='Raw_vs_Normalized')
    ttest_results.to_excel(writer, sheet_name='TTest_Results', index=False)
    df_stem_breakdown.to_excel(writer, sheet_name='Stem_Breakdown', index=False)
    anova_results.to_excel(writer, sheet_name='Anova Results', index=False)
    anova_summary_df.to_excel(writer, sheet_name='Anova Results', index=False)
    interaction_means_anova.to_excel(writer, sheet_name='Anova Results', index=False)
    

print("\n✅ ALL RESULTS SAVED TO: analysis/qualtrics_analysis.xlsx")


# print("\n" + "="*60)
# print("STEP 9: COMPARISON WITH OBJECTIVE METRICS")
# print("="*60)

# # Load objective metrics
# print("\nLoading objective metric data...")

# # Load BSS Eval (SDR) results
# df_bss = pd.read_csv('results/bss_eval_results_processed.csv')
# print(f"  ✓ Loaded SDR data: {len(df_bss)} rows")

# # Load FAD results (use the processed sheet with both embeddings)
# df_fad = pd.read_excel('results/fad_scores_workbook.xlsx', 
#                        sheet_name='fad_scores_pann_processed')
# print(f"  ✓ Loaded FAD data: {len(df_fad)} models")
# print(f"    Available: {list(df_fad['model'].values)}")

# # Calculate model-level averages
# print("\nCalculating model-level averages...")

# # MUSHRA: Already have this from Step 8
# mushra_by_model = df_normalized.groupby('model')['rating_zscore'].mean()
# print(f"  MUSHRA models: {list(mushra_by_model.index)}")

# # SDR: Average across all trials per model
# sdr_by_model = df_bss.groupby('model')['SDR'].mean()
# print(f"  SDR models: {list(sdr_by_model.index)}")

# # FAD: Already at model level (use VGGish as standard)
# fad_vggish_dict = dict(zip(df_fad['model'], df_fad['FAD (vggish)']))
# fad_pann_dict = dict(zip(df_fad['model'], df_fad['FAD (pann)']))
# print(f"  FAD models: {list(fad_vggish_dict.keys())}")


# print("\n" + "="*60)
# print("COMPARISON TABLES")
# print("="*60)

# # Table 1: MUSHRA vs SDR (with reference, n=5)
# comparison_sdr = pd.DataFrame({
#     'model': ['reference', 'htdemucs', 'dv2', 'spleeter', 'anchor']
# })
# comparison_sdr['MUSHRA'] = comparison_sdr['model'].map(mushra_by_model)
# comparison_sdr['SDR'] = comparison_sdr['model'].map(sdr_by_model)

# print("\n1. MUSHRA vs SDR (n=5 models, including reference):")
# print(comparison_sdr.sort_values('MUSHRA', ascending=False).to_string(index=False))

# # Table 2: MUSHRA vs FAD (with reference, n=5)
# comparison_fad = pd.DataFrame({
#     'model': ['reference', 'htdemucs', 'dv2', 'spleeter', 'anchor']
# })
# comparison_fad['MUSHRA'] = comparison_fad['model'].map(mushra_by_model)
# comparison_fad['FAD_VGGish'] = comparison_fad['model'].map(fad_vggish_dict)
# comparison_fad['FAD_PANN'] = comparison_fad['model'].map(fad_pann_dict)

# print("\n2. MUSHRA vs FAD (n=5 models):")
# print(comparison_fad.sort_values('MUSHRA', ascending=False).to_string(index=False))

# print("\n" + "="*60)
# print("CORRELATION ANALYSIS")
# print("="*60)

# # Correlation 1: MUSHRA vs SDR
# print("\n1. MUSHRA vs SDR (n=5)")
# print("-" * 40)

# spearman_sdr, p_spearman_sdr = stats.spearmanr(
#     comparison_sdr['MUSHRA'], 
#     comparison_sdr['SDR']
# )
# pearson_sdr, p_pearson_sdr = stats.pearsonr(
#     comparison_sdr['MUSHRA'], 
#     comparison_sdr['SDR']
# )

# print(f"Spearman ρ = {spearman_sdr:.3f}, p = {p_spearman_sdr:.4f}")
# print(f"Pearson  r = {pearson_sdr:.3f}, p = {p_pearson_sdr:.4f}")

# if p_spearman_sdr < 0.05:
#     print("✅ Statistically significant")
# else:
#     print("⚠️  Not statistically significant")

# # Correlation 2: MUSHRA vs FAD (VGGish)
# print("\n2. MUSHRA vs FAD-VGGish (n=4)")
# print("-" * 40)

# spearman_fad_vgg, p_spearman_fad_vgg = stats.spearmanr(
#     comparison_fad['MUSHRA'], 
#     comparison_fad['FAD_VGGish']
# )
# pearson_fad_vgg, p_pearson_fad_vgg = stats.pearsonr(
#     comparison_fad['MUSHRA'], 
#     comparison_fad['FAD_VGGish']
# )

# print(f"Spearman ρ = {spearman_fad_vgg:.3f}, p = {p_spearman_fad_vgg:.4f}")
# print(f"Pearson  r = {pearson_fad_vgg:.3f}, p = {p_pearson_fad_vgg:.4f}")

# if p_spearman_fad_vgg < 0.05:
#     print("✅ Statistically significant")
# else:
#     print("⚠️  Not statistically significant (n=4 limits power)")

# # Correlation 3: MUSHRA vs FAD (PANN) - Optional
# print("\n3. MUSHRA vs FAD-PANN (n=4)")
# print("-" * 40)

# spearman_fad_pann, p_spearman_fad_pann = stats.spearmanr(
#     comparison_fad['MUSHRA'], 
#     comparison_fad['FAD_PANN']
# )
# pearson_fad_pann, p_pearson_fad_pann = stats.pearsonr(
#     comparison_fad['MUSHRA'], 
#     comparison_fad['FAD_PANN']
# )

# print(f"Spearman ρ = {spearman_fad_pann:.3f}, p = {p_spearman_fad_pann:.4f}")
# print(f"Pearson  r = {pearson_fad_pann:.3f}, p = {p_pearson_fad_pann:.4f}")

# if p_spearman_fad_pann < 0.05:
#     print("✅ Statistically significant")
# else:
#     print("⚠️  Not statistically significant (n=4 limits power)")


# # Bonus: Do objective metrics agree with each other?
# print("\n4. SDR vs FAD-VGGish (n=4)")
# print("-" * 40)

# comparison_obj = comparison_fad.copy()
# comparison_obj['SDR'] = comparison_obj['model'].map(sdr_by_model)

# spearman_obj, p_spearman_obj = stats.spearmanr(
#     comparison_obj['SDR'], 
#     comparison_obj['FAD_VGGish']
# )

# print(f"Spearman ρ = {spearman_obj:.3f}, p = {p_spearman_obj:.4f}")
# print("(Do objective metrics rank models consistently?)")

# # ============================================================
# # Save correlation results
# # ============================================================

# correlation_summary = pd.DataFrame({
#     'Comparison': [
#         'MUSHRA vs SDR', 
#         'MUSHRA vs FAD-VGGish', 
#         'MUSHRA vs FAD-PANN',
#         'SDR vs FAD-VGGish'
#     ],
#     'n': [5, 4, 4, 4],
#     'Spearman_rho': [
#         spearman_sdr, 
#         spearman_fad_vgg, 
#         spearman_fad_pann,
#         spearman_obj
#     ],
#     'Spearman_p': [
#         p_spearman_sdr, 
#         p_spearman_fad_vgg, 
#         p_spearman_fad_pann,
#         p_spearman_obj
#     ],
#     'Pearson_r': [
#         pearson_sdr, 
#         pearson_fad_vgg, 
#         pearson_fad_pann,
#         np.nan  # Only Spearman for SDR vs FAD
#     ],
#     'Pearson_p': [
#         p_pearson_sdr, 
#         p_pearson_fad_vgg, 
#         p_pearson_fad_pann,
#         np.nan
#     ]
# })

# correlation_summary.to_csv('analysis/correlation_summary.csv', index=False)
# print("\n✅ Saved: analysis/correlation_summary.csv")

# comparison_sdr.to_csv('analysis/comparison_mushra_sdr.csv', index=False)
# comparison_fad.to_csv('analysis/comparison_mushra_fad.csv', index=False)
# print("✅ Saved comparison tables")

# print("\n" + "="*60)
# print("CORRELATION ANALYSIS COMPLETE")
# print("="*60)