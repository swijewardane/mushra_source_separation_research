import numpy as np
import mir_eval
import librosa
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def calculate_bss_metrics(reference_path, estimated_path, target_sr=44100):
    try:
        reference, sr = librosa.load(reference_path, sr=target_sr, mono=True)
        estimated, sr = librosa.load(estimated_path, sr=target_sr, mono=True)
        
        min_len = min(len(reference), len(estimated))
        reference = reference[:min_len]
        estimated = estimated[:min_len]
        
        sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
            reference[np.newaxis, :],
            estimated[np.newaxis, :])
        
        return {
            'SDR': float(sdr[0]),
            'SIR': float(sir[0]),
            'SAR': float(sar[0])
        }
        
    except Exception as e:
        print(f"Error processing {estimated_path.name}: {e}")
        return None
    
def extract_model_name(filename):
    models = ['htdemucs', 'anchor', 'dv2', 'spleeter', 'reference']
    stem = filename.stem
    parts = stem.split('_')
    
    if len(parts) >= 2 and parts[1] in models:
        model = parts[1]  # Gets Model Name
        return model 
    else:
        return stem
    
def batch_evaluate(references_dir, estimated_dir, output_csv): 
    references_dir = Path(references_dir)
    estimated_dir = Path(estimated_dir)
    
    results = []
    reference_files = sorted(references_dir.glob('*.wav'))
    
    print(f"Found {len(reference_files)} reference files\n")
    
    for ref_file in reference_files:
        ref_stem = ref_file.stem
        print(f"Processing reference: {ref_file.name}")
        
        estimates_folder = estimated_dir / ref_stem
        
        if not estimates_folder.exists():
            print(f"  ⚠️  No folder found: {estimates_folder}")
            continue
        
        estimated_files = sorted(estimates_folder.glob('*.wav'))
        
        if not estimated_files:
            print(f"  ⚠️  No .wav files found in {estimates_folder}")
            continue
        
        print(f"  Found {len(estimated_files)} estimated versions")
        
        for est_file in estimated_files:
            
            model_name = extract_model_name(est_file)
            
            print(f"    - {model_name}: ", end="")
            
            metrics = calculate_bss_metrics(ref_file, est_file)
            
            if metrics:
                results.append({
                    'reference_file': ref_file.name,
                    'reference_stem': ref_stem,
                    'model': model_name,
                    'estimated_file': est_file.name,
                    'SDR': metrics['SDR'],
                    'SIR': metrics['SIR'],
                    'SAR': metrics['SAR']
                })
                
                print(f"SDR: {metrics['SDR']:6.2f} dB, SIR: {metrics['SIR']:6.2f} dB, SAR: {metrics['SAR']:6.2f} dB")
            else:
                print("FAILED")
        
        print()
    df = pd.DataFrame(results)
        
    if df.empty:
        print("⚠️  No results to save!")
        return df
    
    # Create results directory if it doesn't exist
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}\n")
    
    # Print summary statistics by model
    # print("=== Summary Statistics by Model ===")
    # summary = df.groupby('model')[['SDR', 'SIR', 'SAR']].agg(['mean', 'std'])
    # print(summary)
    # print()
    
    # # Print summary statistics by reference
    # print("=== Summary Statistics by Reference ===")
    # ref_summary = df.groupby('reference_stem')[['SDR', 'SIR', 'SAR']].mean()
    # print(ref_summary)
    
    return df

if __name__ == "__main__":
    # Set your paths
    references_dir = "references"
    estimated_dir = "estimated_sources"
    output_csv = "results/bss_eval_results.csv"
    
    # Run batch evaluation
    results_df = batch_evaluate(references_dir, estimated_dir, output_csv)              