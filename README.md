# MUSHRA Study: Perceptual Evaluation of Music Source Separation Models

A perceptual evaluation of various commercial models of music source separation, with a focus on model performance against non-traditional source material.

**Author:** Sahan Wijewardane  
**Institution:** University of Miami, Frost School of Music  
**Program:** Master of Science in Music Engineering Technology  
**Expected Completion:** Spring 2027

## Abstract

This repository contains the complete materials, data, and analysis code for a MUSHRA (MUltiple Stimuli with Hidden Reference and Anchor) listening study evaluating the perceptual performance of three commercial music source separation models: **HTDemucs**, **Demucs v2**, and **Spleeter**. The study focuses on evaluating these models' performance on non-traditional source material, specifically electronic music (dubstep/EDM), comparing subjective listener ratings against objective metrics including SDR (Signal-to-Distortion Ratio) and FAD (Fréchet Audio Distance) scores.

## Study Overview

- **Participants:** 15 participants completed the survey; 12 were included in final analysis
- **Models Evaluated:** HTDemucs, Demucs v2, Spleeter
- **Source Material:** 6 dubstep/EDM tracks (non-traditional instrumentation)
- **Target Sources:** Bass and drums
- **Methodology:** MUSHRA protocol implemented via Qualtrics
- **Evaluation Metrics:**
  - Subjective: MUSHRA scores (0-100 scale)
  - Objective: SDR (via BSS Eval), FAD scores (VGGish and PANN embeddings)

## Repository Structure

```
.
├── estimated_sources/          # Separated audio files from each model
│   ├── celebrate_bass/
│   ├── dropnoir_drums/
│   ├── jackiesgarage_bass/
│   ├── monstaclat_drums/
│   ├── nogravity_drums/
│   └── thisfeeling_bass/
│
├── references/                 # Ground truth (reference) audio files
│   ├── celebrate_bass.wav
│   ├── dropnoir_drums.wav
│   ├── jackiesgarage_bass.wav
│   ├── monstaclat_drums.wav
│   ├── nogravity_drums.wav
│   └── thisfeeling_bass.wav
│
├── results/                    # Analysis results and visualizations
│   ├── raw_data/              # Raw BSS Eval, FAD, and Qualtrics data
│   ├── bss_eval_results_processed.csv
│   ├── fad_scores_workbook.xlsx
│   ├── mushra_barchart_se.png
│   └── mushra_interaction_plot.png
│
├── analysis/                   # Processed analysis files
│   ├── correlation_summary_table.csv
│   ├── comparison_mushra_sdr.csv
│   ├── comparison_mushra_fad.csv
│   └── loudness_comparison.csv
│
├── scripts/                    # Python analysis scripts
│   ├── batch_bss_eval.py      # Batch BSS Eval calculation
│   ├── calculate_fad_eval.py  # FAD score calculation
│   ├── correlation_analysis.py # Correlation analysis
│   ├── qualtrics_data_analysis.py
│   └── spearman_correlation.py
│
├── qualtrics_survey/          # Exported Qualtrics survey design/flow
│    ├── MUE_705_Listening_Study.docx
│    ├── MUE_705_Listening_Study.qsf
│
└── environment.yml             # Conda environment specification

```

## Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/swijewardane/mushra_source_separation_research.git
   cd mushra_source_separation_research
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate mushra
   ```

## Usage

### Running BSS Eval Analysis
```bash
python scripts/batch_bss_eval.py
```

### Calculating FAD Scores
```bash
python scripts/calculate_fad_eval.py
```

### Analyzing MUSHRA Results
```bash
python scripts/qualtrics_data_analysis.py
```

### Correlation Analysis
```bash
python scripts/correlation_analysis.py
python scripts/spearman_correlation.py
```

## Key Findings

This study investigates the relationship between subjective perceptual ratings (MUSHRA scores) and objective evaluation metrics (SDR and FAD) for music source separation on non-traditional source material. Correlation analyses examine whether objective metrics reliably predict human perception for electronic music genres.

See the `results/` and `analysis/` directories for detailed findings, statistical analyses, and visualizations.

## Test Material

The study uses 6 dubstep/EDM tracks selected to represent non-traditional instrumentation:
- **celebrate** (bass separation)
- **dropnoir** (drums separation)
- **jackiesgarage** (bass separation)
- **monstaclat** (drums separation)
- **nogravity** (drums separation)
- **thisfeeling** (bass separation)

Each track was processed through three separation models (HTDemucs, Demucs v2, Spleeter), with results compared against isolated ground truth stems (reference) and a low-passed, bitcrushed anchor.

## Methodology

### MUSHRA Protocol
Participants rated separated sources on a 0-100 scale in a MUSHRA test administered via Qualtrics, comparing:
- HTDemucs output
- Demucs v2 output
- Spleeter output
- Hidden reference (ground truth)
- Hidden anchor (degraded version)

### Objective Evaluation
- **BSS Eval:** Computed SDR, SIR, and SAR metrics using `mir_eval`
- **FAD:** Calculated Fréchet Audio Distance using both VGGish and PANN embeddings

## Dependencies

Key packages (see `environment.yml` for complete list):
- Python 3.x
- librosa
- numpy
- scipy
- pandas
- matplotlib
- mir_eval
- torch (for FAD calculation)

## Data Availability

- **Audio stimuli:** All separated sources and references are included in `estimated_sources/` and `references/`
- **Raw data:** BSS Eval metrics, FAD scores, and Qualtrics responses in `results/raw_data/`
- **Processed results:** Aggregated analysis in `results/` and `analysis/`

## Citation

If you use this dataset or methodology in your research, please cite:

```
Wijewardane, S. (2026). A perceptual evaluation of various commercial models 
of music source separation, with a focus on model performance against 
non-traditional source material [Master's research project, University of Miami].
```

## License

This research is conducted as part of a Master's research project at the University of Miami. Please contact the author for information regarding data usage and permissions.

## Contact

**Sahan Wijewardane**  
Master of Science in Music Engineering Technology  
Frost School of Music, University of Miami  
Email: sahan13945@gmail.com  
GitHub: [@swijewardane](https://github.com/swijewardane)

## Acknowledgments

This research was conducted at the University of Miami's Frost School of Music as part of the Music Engineering Technology graduate program.

---

*Repository created: January 2026*  
*Last updated: January 2026*
