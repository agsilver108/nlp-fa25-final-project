# Analysis Scripts Directory

This directory contains all analysis and detection scripts for identifying dataset artifacts in SQuAD.

## Scripts Overview

### 🔍 `basic_prediction_analysis.py`
**Purpose**: Initial exploration of model prediction patterns  
**Features**:
- Top predicted answers analysis
- Question type categorization
- Basic pattern recognition
- Sample error analysis

**Usage**:
```bash
python analysis_scripts\basic_prediction_analysis.py
```

### 📊 `systematic_artifact_analysis.py`  
**Purpose**: Comprehensive artifact detection framework  
**Features**:
- Lexical overlap analysis between questions and answers
- Question pattern analysis by type (what/who/when/where/how)
- Position bias detection in passage answers
- N-gram correlation analysis
- Model prediction vs gold comparison
- Contrast example generation

**Key Methods**:
- `analyze_lexical_overlap()` - Question-answer word overlap
- `analyze_question_patterns()` - Question type distributions  
- `analyze_position_bias()` - Answer location in passages
- `analyze_ngram_correlations()` - Spurious pattern detection
- `generate_contrast_examples()` - Adversarial example creation

**Usage**:
```bash
python analysis_scripts\systematic_artifact_analysis.py
```

### 🧪 `model_ablation_analysis.py`
**Purpose**: Model ablation studies to test spurious correlations  
**Features**:
- Question-only dataset generation (tests memorization)
- Passage-only dataset generation (tests generic patterns)
- Random baseline analysis
- Question-answer correlation strength measurement
- Answer type bias detection

**Key Methods**:
- `create_question_only_dataset()` - Remove passage context
- `create_passage_only_dataset()` - Use generic questions
- `analyze_question_answer_correlations()` - Find spurious patterns
- `analyze_answer_type_biases()` - Type prediction analysis

**Usage**:
```bash
python analysis_scripts\model_ablation_analysis.py
```

### 📈 `statistical_artifact_analysis.py`
**Purpose**: Statistical validation of artifact findings  
**Features**:
- Chi-square tests for bias significance
- Question type bias statistical testing
- Position bias significance testing  
- Model prediction bias validation
- Artifact strength scoring (0-1 scale)

**Key Methods**:
- `chi_square_test()` - Statistical significance testing
- `test_question_type_bias()` - Question pattern significance
- `test_position_bias_significance()` - Position distribution testing
- `test_model_prediction_bias()` - Prediction vs gold testing
- `calculate_artifact_strength_scores()` - Overall artifact quantification

**Usage**:
```bash
python analysis_scripts\statistical_artifact_analysis.py
```

## Dependency Chain

```
basic_prediction_analysis.py
    ↓ (initial findings)
systematic_artifact_analysis.py  
    ↓ (detailed patterns)
model_ablation_analysis.py
    ↓ (ablation datasets)
statistical_artifact_analysis.py
    ↓ (validation)
[Results ready for mitigation phase]
```

## Output Files

Each script generates results in the `../results/` directory:

| Script | Output File | Description |
|--------|-------------|-------------|
| `systematic_artifact_analysis.py` | `01_systematic_artifact_analysis.json` | Comprehensive artifact patterns |
| `model_ablation_analysis.py` | `02_model_ablation_analysis.json` | Ablation study results |
| `statistical_artifact_analysis.py` | `03_statistical_significance_tests.json` | Statistical validation |

## Research Implementation

These scripts implement methods from key papers:

- **Gardner et al. (2021)**: Competency problems framework (n-gram correlations)
- **Gardner et al. (2020)**: Contrast sets generation
- **Poliak et al. (2018)**: Question-only ablations  
- **Kaushik & Lipton (2018)**: Passage-only analysis

## Usage Notes

- All scripts expect predictions in `../results/baseline_evaluation/eval_predictions.jsonl`
- Run scripts from project root directory
- Scripts can be run independently but results build on each other
- Statistical analysis script can use results from previous analyses