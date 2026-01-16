import sys
import numpy as np
from frechet_audio_distance import FrechetAudioDistance
from pathlib import Path
import pandas as pd
import warnings
import shutil
import tempfile
warnings.filterwarnings('ignore')

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from batch_bss_eval import extract_model_name

def collect_files_by_model(references_dir, estimated_dir):
    
    references_dir = Path(references_dir)
    estimated_dir = Path(estimated_dir)
    
    ref_files = sorted(references_dir.glob('*.wav'))
    
    model_files = {}
    
    print(f"Found {len(ref_files)} reference files")
    print("\nCollecting estimated files by model...")
    
    for ref_file in ref_files:
        ref_stem = ref_file.stem
        estimates_folder = estimated_dir/ref_stem
        
        if not estimates_folder.exists():
            print(f' No folder found for {ref_stem}')
            continue
        
        estimated_files = sorted(estimates_folder.glob('*.wav'))
        
        for est_file in estimated_files:
            model_name = extract_model_name(est_file)
            
            if model_name not in model_files:
                model_files[model_name] = []
            
            model_files[model_name].append(est_file)
    
    print("\nFiles collected:")
    print(f"  References: {len(ref_files)} files")
    for model, files in model_files.items():
        print(f"  {model}: {len(files)} files")
    
    return ref_files, model_files
    
def calculate_fad_scores(references_dir, estimated_dir, sample_rate=16000, 
                        model_name="vggish"):
    """
    Calculate FAD scores for each model
    
    Args:
        references_dir: Path to reference audio files
        estimated_dir: Path to estimated sources
        sample_rate: Sample rate for FAD (default 16kHz)
        model_name: Embedding model ("vggish", "pann", or "clap")
    
    Returns:
        DataFrame with FAD scores for each model
    """
    
    print("="*60)
    print("Fréchet Audio Distance (FAD) Calculation")
    print("="*60)
    print(f"Using embedding model: {model_name}")
    print(f"Sample rate: {sample_rate} Hz\n")
    
    # Collect files organized by model
    ref_files, model_files = collect_files_by_model(references_dir, estimated_dir)
    
    if not ref_files:
        print("❌ No reference files found!")
        return pd.DataFrame()
    
    if not model_files:
        print("❌ No estimated files found!")
        return pd.DataFrame()
    
    # Initialize FAD calculator
    print(f"\nInitializing FAD with {model_name} embeddings...")
    frechet = FrechetAudioDistance(
        model_name=model_name,
        sample_rate=sample_rate,
        #use_pca=False,
        #use_activation=False,
        verbose=False
    )
    
    # Calculate FAD for each model
    results = []
    
    print("\nCalculating FAD scores...\n")
    
    with tempfile.TemporaryDirectory() as ref_temp_dir:
        print('Preparing reference files...')
        for ref_file in ref_files:
            shutil.copy2(ref_file, ref_temp_dir)
            
    
        for model, est_files in sorted(model_files.items()):
            print(f"Processing {model}...")
            print(f"  Comparing {len(ref_files)} references vs {len(est_files)} estimates")
            
            try:
                
                with tempfile.TemporaryDirectory() as est_temp_dir:
                    
                    for est_file in est_files:
                        shutil.copy2(est_file, est_temp_dir)
                        
                
                    # Calculate FAD score
                    fad_score = frechet.score(
                        ref_temp_dir,
                        est_temp_dir,
                        dtype="float32"
                    )
                    
                    results.append({
                        'model': model,
                        'FAD': fad_score,
                        'n_references': len(ref_files),
                        'n_estimates': len(est_files)
                    })
                    
                    print(f"  ✅ FAD Score: {fad_score:.4f}\n")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                results.append({
                    'model': model,
                    'FAD': None,
                    'n_references': len(ref_files),
                    'n_estimates': len(est_files)
                })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df

def main():
    """Main execution"""
    
    # Configuration
    references_dir = "references"
    estimated_dir = "estimated_sources"
    output_csv = "results/fad_scores_pann.csv"
    
    # FAD parameters
    sample_rate = 16000  # Standard for FAD
    embedding_model = "pann"  # Options: "vggish", "pann", "clap"
    
    # Calculate FAD scores
    results_df = calculate_fad_scores(
        references_dir=references_dir,
        estimated_dir=estimated_dir,
        sample_rate=sample_rate,
        model_name=embedding_model
    )
    
    if results_df.empty:
        print("No results to save!")
        return
    
    # Create results directory if needed
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to {output_csv}")
    
    # Print summary
    print("\n" + "="*60)
    print("FAD Scores Summary (Lower is Better)")
    print("="*60)
    
    # Sort by FAD score
    sorted_df = results_df.sort_values('FAD')
    
    for _, row in sorted_df.iterrows():
        if row['FAD'] is not None:
            print(f"{row['model']:15s}: {row['FAD']:8.4f}")
        else:
            print(f"{row['model']:15s}: {'ERROR':>8s}")
    
    print("\nInterpretation:")
    print("  FAD < 5    : Excellent quality")
    print("  FAD 5-15   : Good quality")
    print("  FAD 15-30  : Moderate quality")
    print("  FAD > 30   : Poor quality")

if __name__ == "__main__":
    main()