# NLP Final Project - Directory Structure

## 📁 Project Organization

```
nlp-final-project/
├── deliverables/                    # MAIN DELIVERABLES (for submission)
│   ├── SCIENTIFIC_REPORT.docx       # ✅ Final report (ACM format)
│   ├── colab_training_results.json  # ✅ Training metrics (EM, F1)
│   ├── colab_training_stream.log    # ✅ Training log (12.3 min on T4)
│   └── visualizations/              # ✅ 4 publication-quality figures (300 DPI)
│       ├── figure1_performance_comparison.png
│       ├── figure2_training_dynamics.png
│       ├── figure3_cartography_distribution.png
│       └── figure4_statistical_significance.png
│
├── scripts/                         # PYTHON SOURCE CODE
│   ├── create_acm_document.py       # Generate ACM-format Word document
│   ├── create_visualizations.py     # Generate 4 visualizations
│   ├── train_with_cartography.py    # CartographyWeightedTrainer implementation
│   ├── helpers.py                   # QuestionAnsweringTrainer utilities
│   ├── run.py                       # Starter training script
│   └── create_clean_document.py     # Alternative document generator
│
├── notebooks/                       # JUPYTER NOTEBOOKS
│   ├── NLP_Final_Project_Colab.ipynb        # Main Colab notebook
│   └── NLP_Final_Project_Colab_old.ipynb    # Backup version
│
├── documentation/                   # PROJECT DOCUMENTATION
│   ├── README_FINAL_STATUS.md       # Status summary
│   ├── SCIENTIFIC_REPORT.md         # Markdown version of report
│   ├── ACM_FORMAT_READY.txt         # Format verification notes
│   ├── DOCUMENT_FIXED.txt           # Document refinement log
│   ├── VISUALIZATIONS_SUMMARY.md    # Visualization details
│   └── VISUALIZATIONS_COMPLETE.txt  # Visualization status
│
├── archive/                         # OLD/DEBUG FILES (not needed for submission)
│   ├── *.md                         # Debug documentation
│   ├── *.py                         # Debug scripts
│   └── SCIENTIFIC_REPORT_WITH_VISUALIZATIONS.docx
│
├── colab_assist/                    # COLAB GPU TRAINING INFRASTRUCTURE
│   ├── colab_streaming_training.py  # Main GPU training script
│   ├── colab_training*.py           # Variant implementations
│   ├── monitor_training.py          # Training monitor
│   ├── QUICK_START.md               # GPU setup guide
│   ├── README.md                    # Colab documentation
│   └── STREAMING_GUIDE.md           # Streaming output guide
│
├── analysis_scripts/                # DATA ANALYSIS MODULES
│   ├── dataset_cartography.py       # Cartography metrics computation
│   ├── systematic_artifact_analysis.py
│   ├── statistical_artifact_analysis.py
│   ├── model_ablation_analysis.py
│   ├── basic_prediction_analysis.py
│   └── README.md
│
├── results/                         # ANALYSIS RESULTS
│   ├── 01_systematic_artifact_analysis.json
│   ├── 02_model_ablation_analysis.json
│   ├── 03_statistical_significance_tests.json
│   └── README.md
│
├── cartography_model/               # TRAINED CARTOGRAPHY MODEL
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
│
├── datasets/                        # CUSTOM DATASETS
│   ├── question_only_ablation.jsonl
│   └── passage_only_ablation.jsonl
│
├── test_*/                          # TEST/DEBUG DIRECTORIES
│   └── [various trained model checkpoints]
│
├── reports/                         # ANALYSIS REPORTS
│   └── 01_comprehensive_artifact_analysis_report.md
│
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
├── README_v1.1.md                   # Original project specification
└── .vscode/                         # VS Code settings
```

---

## 🎯 For Submission - Use These Files

### Main Deliverable
- **`deliverables/SCIENTIFIC_REPORT.docx`** - The final report
  - Format: ACM conference proceedings (3-8 pages + references)
  - Contains: Abstract, Intro, Related Work, Methodology, Results, Discussion, References
  - Includes: 4 embedded visualizations
  - Status: ✅ Publication-ready

### Supporting Materials
- **`deliverables/colab_training_results.json`** - Quantitative results
  - Baseline: EM 52.2%, F1 61.26%
  - Cartography: EM 57.1%, F1 66.34%
  - Improvement: +4.9% EM, +5.08% F1

- **`deliverables/colab_training_stream.log`** - Training execution log
  - 12.3 minutes on T4 GPU
  - Shows all training epochs and metrics

- **`deliverables/visualizations/`** - 4 publication-quality figures
  - 300 DPI PNG files
  - Embedded in the Word document
  - Standalone available for presentations

---

## 📊 Project Summary

**Task**: Analyze and mitigate dataset artifacts in SQuAD using dataset cartography

**Dataset**: SQuAD 1.1 (10,000 train, 1,000 validation examples)

**Model**: ELECTRA-small (13.5M parameters)

**Method**: Dataset cartography with hard example reweighting (2x multiplier)

**Results**:
- Identified statistically significant artifacts (p < 0.001)
- Achieved +5.08% F1 improvement through cartography-guided reweighting
- Dataset distribution: 7.2% easy, 25.7% hard, 67.1% ambiguous

**Key Findings**:
- Position bias: χ² = 237.21 (p < 0.001)
- Prediction bias: χ² = 1084.87 (p < 0.001)
- Training dynamics show consistent cartography advantage

---

## 🔧 To Run/Reproduce

### GPU Training (Colab)
```bash
# See: colab_assist/QUICK_START.md
# Use: notebooks/NLP_Final_Project_Colab.ipynb
```

### Generate Report (Local)
```bash
python scripts/create_acm_document.py
```

### Generate Visualizations (Local)
```bash
python scripts/create_visualizations.py
```

---

## 📝 Notes

- Main source code: `scripts/`
- Analysis modules: `analysis_scripts/`
- All key results: `deliverables/`
- Old debugging files: `archive/` (can be deleted)
- Project specification: `README_v1.1.md`

---

**Status**: ✅ COMPLETE AND READY FOR SUBMISSION

Last organized: November 5, 2025
