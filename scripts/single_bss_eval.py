import numpy as np
import mir_eval
import librosa

target_sr = 44100

reference, sr = librosa.load('references/celebrate_bass.wav', sr=target_sr, mono=True)
estimated, sr = librosa.load('estimated_sources//celebrate_bass/celebrate_dv2_bass.wav', sr=target_sr, mono=True)

print(f"Reference: {len(reference)} samples at {sr} Hz ({len(reference)/sr:.2f} seconds)")
print(f"Estimated: {len(estimated)} samples at {sr} Hz ({len(estimated)/sr:.2f} seconds)")

# Check if the estimated portion matches anywhere in the reference
if len(reference) > len(estimated):
    # Try aligning - check correlation at different offsets
    best_corr = -1
    best_offset = 0
    
    for offset in range(0, len(reference) - len(estimated), sr//10):  # Check every 0.1 sec
        ref_segment = reference[offset:offset+len(estimated)]
        corr = np.corrcoef(ref_segment, estimated)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_offset = offset
    
    print(f"\nBest correlation: {best_corr:.4f} at offset {best_offset/sr:.2f} seconds")
    
    if best_corr > 0.5:
        print("Files appear to be aligned! Using best offset...")
        reference = reference[best_offset:best_offset+len(estimated)]
    else:
        print("WARNING: Low correlation - these might not be the same song/source!")
        reference = reference[:len(estimated)]
else:
    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]

# Calculate BSS Eval metrics
sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
    reference[np.newaxis, :],
    estimated[np.newaxis, :]
)

print(f"\nSDR: {sdr[0]:.2f} dB")
print(f"SIR: {sir[0]:.2f} dB")
print(f"SAR: {sar[0]:.2f} dB")