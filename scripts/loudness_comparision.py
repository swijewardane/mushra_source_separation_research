# import librosa

# # For one track, compare loudness
# ref_drums, sr = librosa.load('references/nogravity_drums.wav')
# htd_drums, sr = librosa.load('estimated_sources/nogravity_drums/nogravity_htdemucs_drums.wav')

# ref_rms = librosa.feature.rms(y=ref_drums)[0].mean()
# htd_rms = librosa.feature.rms(y=htd_drums)[0].mean()

# ref_db = 20 * np.log10(ref_rms)
# htd_db = 20 * np.log10(htd_rms)

# print(f"Reference RMS level: {ref_db:.1f} dB")
# print(f"htdemucs RMS level: {htd_db:.1f} dB")
# print(f"Difference: {htd_db - ref_db:.1f} dB")     

import librosa
import numpy as np
from pathlib import Path

print("="*60)
print("LOUDNESS COMPARISON: All Tracks")
print("="*60)

# Define your tracks
tracks = [
    ('nogravity_drums', 'drums'),
    ('monstaclat_drums', 'drums'),
    ('dropnoir_drums', 'drums'),
    ('celebrate_bass', 'bass'),
    ('jackiesgarage_bass', 'bass'),
    ('thisfeeling_bass', 'bass')
]

results = []

for track_name, stem_type in tracks:
    print(f"\nAnalyzing: {track_name}")
    track_name_split = track_name.split('_')
    
    # Construct file paths
    ref_path = f"references/{track_name}.wav"
    htd_path = f"estimated_sources/{track_name}/{track_name_split[0]}_htdemucs_{stem_type}.wav"
    
    # Adjust path if naming is different
    # Check what your actual filenames are and adjust accordingly
    
    try:
        # Load audio
        ref_audio, _ = librosa.load(ref_path, sr=None, mono=True)
        htd_audio, _ = librosa.load(htd_path, sr=None, mono=True)
        
        # Calculate RMS
        ref_rms = np.sqrt(np.mean(ref_audio**2))
        htd_rms = np.sqrt(np.mean(htd_audio**2))
        
        # Convert to dB
        ref_db = 20 * np.log10(ref_rms + 1e-10)
        htd_db = 20 * np.log10(htd_rms + 1e-10)
        
        difference = htd_db - ref_db
        
        results.append({
            'track': track_name,
            'stem_type': stem_type,
            'ref_db': ref_db,
            'htd_db': htd_db,
            'difference_db': difference
        })
        
        print(f"  Reference: {ref_db:.1f} dB")
        print(f"  htdemucs:  {htd_db:.1f} dB")
        print(f"  Difference: {difference:+.1f} dB")
        
    except FileNotFoundError as e:
        print(f"  ❌ File not found: {e}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Summary
if results:
    import pandas as pd
    df_loudness = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nBy Stem Type:")
    for stem_type in ['drums', 'bass']:
        stem_data = df_loudness[df_loudness['stem_type'] == stem_type]
        if len(stem_data) > 0:
            avg_diff = stem_data['difference_db'].mean()
            print(f"\n{stem_type.upper()}:")
            print(f"  Average loudness difference: {avg_diff:+.1f} dB")
            print(f"  Range: {stem_data['difference_db'].min():+.1f} to {stem_data['difference_db'].max():+.1f} dB")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    drums_diff = df_loudness[df_loudness['stem_type'] == 'drums']['difference_db'].mean()
    bass_diff = df_loudness[df_loudness['stem_type'] == 'bass']['difference_db'].mean()
    
    print(f"\nDrums: htdemucs is {drums_diff:+.1f} dB louder on average")
    print(f"Bass:  htdemucs is {bass_diff:+.1f} dB louder on average")
    
    if drums_diff > bass_diff + 1:
        print("\n✅ htdemucs retains more mastering processing on DRUMS than BASS")
        print("   This explains why participants rated htdemucs drums higher!")
    elif abs(drums_diff - bass_diff) < 1:
        print("\n≈ Similar loudness differences for both stem types")
    
    # Save results
    df_loudness.to_csv('results/loudness_comparison.csv', index=False)
    print("\n✅ Results saved to results/loudness_comparison.csv")