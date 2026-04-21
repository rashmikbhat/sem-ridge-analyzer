# SEM Ridge Analyzer

Computer vision tool for automatically measuring ridge dimensions and trench gaps from SEM cross-sectional images. Uses vertical projection to detect ridges and OCR to read the scale bar for unit conversion. Includes robustness testing with 7 different image augmentations.

## Quick Start
```bash
pip install -r requirements.txt
jupyter notebook notebooks/sem_analysis.ipynb
```

**Results**: Detects 5 ridges, measures height (0.288 µm), width (0.271 µm), gap (0.102 µm), substrate (0.038 µm)
