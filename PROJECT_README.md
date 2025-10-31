# SQuAD Dataset Artifact Analysis Project

**CS388 Natural Language Processing - Final Project**  
**Topic**: Analyzing and Mitigating Dataset Artifacts in Question Answering  
**Dataset**: Stanford Question Answering Dataset (SQuAD 1.1)  
**Model**: ELECTRA-small  

## 📁 Project Structure

```
nlp-final-project/
├── 📊 analysis_scripts/           # Analysis and detection scripts
│   ├── basic_prediction_analysis.py      # Initial prediction pattern analysis  
│   ├── systematic_artifact_analysis.py   # Comprehensive artifact detection
│   ├── model_ablation_analysis.py        # Question/passage-only ablations
│   └── statistical_artifact_analysis.py  # Statistical significance testing
├── 📈 results/                   # Analysis results and data
│   ├── 01_systematic_artifact_analysis.json     # Detailed artifact findings
│   ├── 02_model_ablation_analysis.json          # Ablation study results
│   ├── 03_statistical_significance_tests.json   # Statistical validation
│   └── baseline_evaluation/                     # Model evaluation outputs
│       ├── eval_metrics.json                    # Performance metrics
│       └── eval_predictions.jsonl               # Detailed predictions
├── 📋 reports/                   # Documentation and reports
│   └── 01_comprehensive_artifact_analysis_report.md  # Main analysis report
├── 🤖 models/                    # Trained models
│   └── baseline_electra_small/              # ELECTRA-small baseline model
├── 📊 datasets/                  # Custom datasets for analysis
│   ├── question_only_ablation.jsonl        # Question-only test dataset
│   └── passage_only_ablation.jsonl         # Passage-only test dataset
├── 🔧 nlp-fp/                    # Virtual environment
├── 📝 Core Files
│   ├── run.py                    # Main training/evaluation script
│   ├── helpers.py                # Utility functions
│   ├── requirements.txt          # Python dependencies
│   ├── README.md                 # This file
│   └── README_v1.1.md           # Assignment specifications
```

## 🎯 Project Objectives

### Part 1: Analysis ✅ COMPLETED
- [x] **Baseline Model Training**: ELECTRA-small on SQuAD (1000 samples)
- [x] **Artifact Detection**: Systematic identification of spurious correlations
- [x] **Statistical Validation**: Chi-square tests for significance
- [x] **Pattern Analysis**: Question-answer correlations and biases

### Part 2: Mitigation 🔄 IN PROGRESS  
- [ ] **Dataset Cartography**: Focus on hard-to-learn examples
- [ ] **Adversarial Training**: Include challenge examples
- [ ] **Ensemble Debiasing**: Train artifact-aware models

## 🔍 Key Findings Summary

### Major Artifacts Identified:

1. **📍 Position Bias** (χ² = 237.21, p < 0.001)
   - 44% overrepresentation in first 10% of passages
   - Mean answer position: 0.43 (heavily front-loaded)

2. **❓ Question Type Stereotyping** (Multiple significant)
   - "WHEN" questions → dates (76.9%, 10.03x overrep)  
   - "HOW" questions → numbers (38.1%, 2.96x overrep)
   - "WHO" questions → persons (78.7% expected)

3. **📊 Answer Type Prediction Bias** (χ² = 1084.87, p < 0.001)
   - Dates over-predicted by 7.39x frequency
   - Numbers under-predicted by 33x frequency
   - Multi-word answers over-predicted by 2.10x

4. **🔗 Strong Lexical Correlations**
   - "what color" → "gold" (67% correlation)
   - "won super" → "denver broncos" (67% correlation)
   - "much did" → "$1.2 billion" (80% correlation)

### Performance Metrics:
- **Baseline Accuracy**: 8.2% EM, 14.1% F1 (intentionally undertrained)
- **Random Baseline**: 1.2% (maximum artifact exploitation)
- **Overall Artifact Strength**: 0.115/1.0 (detectable but not overwhelming)

## 🚀 Quick Start

### Setup Environment
```bash
# Activate virtual environment
.\nlp-fp\Scripts\Activate.ps1

# Verify installation
python -c "import torch, transformers, datasets; print('Ready!')"
```

### Run Analysis
```bash
# Basic prediction analysis
python analysis_scripts\basic_prediction_analysis.py

# Comprehensive artifact detection  
python analysis_scripts\systematic_artifact_analysis.py

# Statistical significance testing
python analysis_scripts\statistical_artifact_analysis.py

# Model ablation studies
python analysis_scripts\model_ablation_analysis.py
```

### Train Models
```bash
# Train baseline model
python run.py --do_train --task qa --dataset squad --output_dir models\new_baseline

# Evaluate model
python run.py --do_eval --task qa --dataset squad --model models\baseline_electra_small --output_dir results\evaluation
```

## 📊 Results Overview

| Analysis Type | File Location | Key Metrics |
|---------------|---------------|-------------|
| **Systematic Analysis** | `results/01_systematic_artifact_analysis.json` | Position bias, lexical overlap, question patterns |
| **Ablation Studies** | `results/02_model_ablation_analysis.json` | Question/passage-only baselines, correlations |
| **Statistical Tests** | `results/03_statistical_significance_tests.json` | Chi-square values, p-values, significance |
| **Baseline Evaluation** | `results/baseline_evaluation/` | Model performance, detailed predictions |

## 📚 Research Context

This project implements artifact analysis methods from:

- **Gardner et al. (2020)**: Contrast sets for local decision boundaries
- **Swayamdipta et al. (2020)**: Dataset cartography for training dynamics  
- **Poliak et al. (2018)**: Hypothesis-only baselines in NLI
- **Jia & Liang (2017)**: Adversarial examples for reading comprehension
- **Ribeiro et al. (2020)**: CheckList behavioral testing

## 🔄 Next Steps

1. **Implement Dataset Cartography**
   - Analyze training dynamics to identify easy/hard/ambiguous examples
   - Focus learning on challenging subsets

2. **Adversarial Training**
   - Generate contrast examples automatically
   - Include artifact-breaking examples in training

3. **Ensemble Debiasing**
   - Train bias-only model to capture artifacts
   - Train main model on residual after removing bias

## 📝 Citation

```bibtex
@project{squad_artifacts_2025,
  title={Analyzing and Mitigating Dataset Artifacts in SQuAD Question Answering},
  author={[Your Name]},
  year={2025},
  course={CS388 Natural Language Processing},
  institution={University of Texas at Austin}
}
```

## 🛠 Dependencies

See `requirements.txt` for full dependencies:
- `transformers` (4.57.1) - Model framework
- `datasets` (4.3.0) - Data loading
- `torch` (2.9.0) - Deep learning backend  
- `evaluate` (0.4.6) - Metrics computation
- `accelerate` (1.11.0) - Training optimization
- `tqdm` (4.67.1) - Progress bars

---

**Project Status**: ✅ Analysis Phase Complete | 🔄 Mitigation Phase In Progress