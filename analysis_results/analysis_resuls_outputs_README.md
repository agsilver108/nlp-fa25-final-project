# Results Directory

This directory contains all analysis results, evaluation outputs, and generated data from the artifact analysis pipeline.

## File Structure

### 📊 Analysis Results (JSON Files)

#### `01_systematic_artifact_analysis.json`
**Source**: `analysis_scripts/systematic_artifact_analysis.py`  
**Content**:
```json
{
  "lexical_overlap": {
    "mean_overlap": 0.277,           // Avg words shared between Q&A
    "high_overlap_count": 2          // Examples with >3 word overlap
  },
  "question_patterns": {             // Answer type distribution by question type
    "what": {"other": 208, "person": 119, "date": 21, ...},
    "who": {"person": 163, "other": 42, ...},
    "when": {"date": 40, "other": 7, ...}
  },
  "position_bias": {
    "mean_position": 0.425,          // Average answer position (0=start, 1=end)
    "position_distribution": [429, 415, 324, ...], // Counts per decile
    "mean_answer_length": 2.06       // Average words per answer
  },
  "ngram_correlations": {            // Strong question-answer correlations
    "super bowl 50?": {"count": 106, "dominant_pattern": "other"},
    "what was the": {"count": 56, "dominant_pattern": "other"}
  },
  "prediction_analysis": {
    "accuracy": 0.073,               // Overall model accuracy  
    "type_confusion": {...}          // Predicted vs gold type matrix
  }
}
```

#### `02_model_ablation_analysis.json`
**Source**: `analysis_scripts/model_ablation_analysis.py`  
**Content**:
```json
{
  "baseline": {
    "most_common_answer": "three",   // Most frequent answer in dataset
    "baseline_accuracy": 0.012,      // Max artifact exploitation possible
    "answer_distribution": {...}     // Answer frequency distribution
  },
  "correlations": {                  // Strong question-answer patterns
    "what color": {"most_common_answer": "gold", "strength": 0.67, "count": 9},
    "won super": {"most_common_answer": "denver broncos", "strength": 0.67, "count": 6}
  },
  "type_biases": {
    "gold_distribution": {...},      // True answer type distribution
    "predicted_distribution": {...}  // Model's prediction type distribution
  },
  "question_only_file": "datasets/question_only_ablation.jsonl",
  "passage_only_file": "datasets/passage_only_ablation.jsonl"
}
```

#### `03_statistical_significance_tests.json`
**Source**: `analysis_scripts/statistical_artifact_analysis.py`  
**Content**:
```json
{
  "question_type_bias": {            // Chi-square tests per question type
    "when": {"chi_square": 167.64, "p_value": 0.001, "significant": true},
    "how": {"chi_square": 116.31, "p_value": 0.001, "significant": true}
  },
  "position_bias": {
    "chi_square": 237.21,            // Test against uniform distribution
    "p_value": 0.001,
    "significant": true,
    "most_overrepresented_bin": 0,   // First 10% of passage
    "overrepresentation_ratio": 1.44
  },
  "prediction_bias": {
    "chi_square": 1084.87,           // Predicted vs gold type distribution
    "p_value": 0.001,
    "significant": true,
    "bias_ratios": {                 // Type over/under-prediction ratios
      "date": 7.39,                 // 7.39x over-predicted
      "number": 0.03                 // 33x under-predicted
    }
  },
  "artifact_scores": {
    "position_bias": 0.003,          // Position uniformity score (0-1)
    "type_bias": 0.163,              // Type prediction bias score (0-1)  
    "overall_artifact_strength": 0.115 // Overall artifact presence (0-1)
  }
}
```

### 📋 Model Evaluation Results

#### `baseline_evaluation/`
**Source**: Model evaluation runs via `run.py`

##### `eval_metrics.json`
```json
{
  "eval_exact_match": 8.2,          // Exact match accuracy (%)
  "eval_f1": 14.112809073090421     // F1 score
}
```

##### `eval_predictions.jsonl`
Detailed prediction file with 1000 examples:
```jsonl
{
  "id": "56be4db0acb8001400a502ec",
  "title": "Super_Bowl_50", 
  "context": "Super Bowl 50 was an American football game...",
  "question": "Which NFL team represented the AFC at Super Bowl 50?",
  "answers": {"text": ["Denver Broncos", ...], "answer_start": [177, ...]},
  "predicted_answer": "February 7, 2016"
}
```

## Key Metrics Summary

| Metric | Value | Significance |
|--------|-------|--------------|
| **Model Accuracy** | 8.2% EM, 14.1% F1 | Intentionally low (undertrained) |
| **Position Bias** | χ² = 237.21, p < 0.001 | Highly significant front-loading |
| **Date Over-prediction** | 7.39x frequency | Severe type bias |
| **Question Type Bias** | 5/7 types significant | Strong stereotyping |
| **Lexical Overlap** | 0.28 words average | Minimal word matching |
| **Artifact Strength** | 0.115/1.0 | Detectable but moderate |

## Statistical Significance

All major findings are statistically validated:

- **Position Bias**: χ² = 237.21 (p < 0.001) ✅
- **Prediction Type Bias**: χ² = 1084.87 (p < 0.001) ✅ 
- **Question Type Stereotyping**: Multiple χ² > 15 (p < 0.001) ✅

## Usage in Mitigation Phase

These results inform mitigation strategies:

1. **Position Bias** → Focus training on mid/end-passage answers
2. **Type Bias** → Reweight training by answer type frequency  
3. **Question Stereotypes** → Include contrast examples breaking patterns
4. **Strong Correlations** → Add adversarial examples for specific patterns

## File Dependencies

```
baseline_evaluation/eval_predictions.jsonl
    ↓ (input to all analyses)
01_systematic_artifact_analysis.json
    ↓ (patterns identified)
02_model_ablation_analysis.json  
    ↓ (correlations quantified)
03_statistical_significance_tests.json
    ↓ (validation complete)
[Ready for mitigation implementation]
```