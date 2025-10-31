# SQuAD Dataset Artifact Analysis - Comprehensive Report

## Executive Summary

This report presents a systematic analysis of dataset artifacts in the Stanford Question Answering Dataset (SQuAD) using an undertrained ELECTRA-small model. We identified multiple statistically significant artifacts that enable models to achieve performance through spurious correlations rather than genuine reading comprehension.

**Key Findings:**
- **7.3% accuracy** on 1000 SQuAD validation examples (baseline model)
- **3 major artifact categories** identified with statistical significance
- **Strong question-answer correlations** found (e.g., "what color" → "gold" 67% of time)
- **Significant position bias** with 44% higher concentration in first 10% of passages
- **Type prediction bias** with dates over-predicted by 7.39x their frequency

---

## 1. Dataset and Model Setup

### 1.1 Environment
- **Model**: ELECTRA-small (google/electra-small-discriminator)
- **Dataset**: SQuAD 1.1 (1000 validation examples analyzed)
- **Training**: Limited training (1000 samples, 1 epoch) to reveal artifacts
- **Framework**: Hugging Face Transformers 4.57.1

### 1.2 Analysis Methodology
We implemented multiple complementary analysis methods:

1. **Systematic Artifact Analysis** - Pattern detection and correlation analysis
2. **Model Ablation Analysis** - Question-only and passage-only comparisons  
3. **Statistical Significance Testing** - Chi-square tests for bias validation
4. **Contrast Example Generation** - Adversarial example creation

---

## 2. Major Artifact Categories Identified

### 2.1 Question Type Bias ⚠️ STATISTICALLY SIGNIFICANT

**Finding**: Different question types show strong correlations with specific answer types, regardless of passage content.

**Statistical Evidence**:
- **WHEN questions**: χ² = 167.64 (p < 0.001)
  - 76.9% expect date answers (10.03x overrepresentation)
- **HOW questions**: χ² = 116.31 (p < 0.001)  
  - 38.1% expect number answers (2.96x overrepresentation)
- **WHO questions**: χ² = 16.40 (p < 0.001)
  - 78.7% expect person answers

**Examples of Strong Correlations**:
```
"what color" → "gold" (67% correlation, 9 examples)
"won super" → "denver broncos" (67% correlation, 6 examples)  
"what position" → "linebacker" (58% correlation, 12 examples)
"much did" → "$1.2 billion" (80% correlation, 5 examples)
```

**Implication**: Models can achieve performance by memorizing question-answer type mappings without reading comprehension.

### 2.2 Position Bias ⚠️ STATISTICALLY SIGNIFICANT

**Finding**: Answers are not uniformly distributed throughout passages.

**Statistical Evidence**:
- χ² = 237.21 (p < 0.001) against uniform distribution
- **First 10% of passage**: 44% overrepresented (14.3% vs expected 10%)
- **First 30% of passage**: Contains 38.9% of all answers
- **Mean answer position**: 0.43 (0=start, 1=end)

**Distribution**:
```
0-10%:  429 answers (14.3%) ← OVERREPRESENTED
10-20%: 415 answers (13.8%) ← OVERREPRESENTED  
20-30%: 324 answers (10.8%)
30-40%: 365 answers (12.2%)
...declining frequency toward end
```

**Implication**: Models can bias attention toward passage beginnings to improve performance.

### 2.3 Answer Type Prediction Bias ⚠️ STATISTICALLY SIGNIFICANT

**Finding**: Model predictions show severe bias toward certain answer types compared to gold distribution.

**Statistical Evidence**:
- χ² = 1084.87 (p < 0.001) between predicted and gold distributions
- **Dates over-predicted**: 17.0% predicted vs 2.3% actual (7.39x bias)
- **Numbers under-predicted**: 0.5% predicted vs 14.4% actual (0.03x bias)
- **Multi-word answers over-predicted**: 47.7% vs 22.7% actual (2.10x bias)

**Type Confusion Matrix**:
```
Gold Type → Predicted Type (% of gold type)
date      → date: 65.4%, other: 33.3%  ✓ Decent
location  → other: 47.2%, location: 36.1%, date: 16.7%
number    → other: 94.0%, date: 4.8%   ✗ Severe failure
other     → other: 83.4%, date: 14.3%  ✗ Date bias
person    → other: 83.3%, date: 16.7%  ✗ Date bias
```

**Implication**: The model has learned to default to frequent patterns (dates, multi-word phrases) rather than context-appropriate answers.

---

## 3. Specific Artifact Examples

### 3.1 Catastrophic Date Bias
The model predicted **"February 7, 2016"** for completely inappropriate questions:

```
Q: "Which NFL team represented the AFC at Super Bowl 50?"
A: "February 7, 2016" (should be "Denver Broncos")

Q: "Where did Super Bowl 50 take place?"  
A: "February 7, 2016" (should be "Levi's Stadium")

Q: "What color was used to emphasize the 50th anniversary?"
A: "February 7, 2016" (should be "gold")
```

### 3.2 Low Lexical Overlap
- **Average lexical overlap**: 0.28 words between questions and answers
- **High overlap examples**: Only 2 out of 1000 examples had >3 overlapping words
- This suggests the model is NOT using simple word matching strategies

### 3.3 N-gram Pattern Exploitation
Most common question trigrams show clear answer biases:
```
"super bowl 50?" → Appears 106 times (1.2% of all trigrams)
"what was the" → 55 times, predominantly non-specific answers
"who was the" → 33 times, predominantly person names
```

---

## 4. Model Performance Analysis

### 4.1 Overall Performance
- **Exact Match**: 8.2%
- **F1 Score**: 14.1%  
- **Random baseline**: 1.2% (if always predicting most frequent answer "three")

### 4.2 Error Analysis
The model's low performance reveals clear artifact dependencies:

1. **Systematic pattern following** rather than comprehension
2. **Entity extraction** based on frequency, not relevance
3. **Question type stereotyping** instead of semantic understanding

---

## 5. Artifact Mitigation Implications

### 5.1 Contrast Examples Generated
We created 5 contrast examples that should break artifact-dependent models:

```
Original: "What color was used to emphasize the 50th anniversary?"
Contrast: "What was not color was used to emphasize the 50th anniversary?"
Expected: Different model behavior if truly understanding vs. pattern matching
```

### 5.2 Recommended Mitigation Strategies

**1. Dataset Cartography Approach:**
- Focus training on "hard-to-learn" examples
- Down-weight examples that exploit artifacts
- Use training dynamics to identify problematic patterns

**2. Adversarial Training:**
- Include contrast examples in training
- Add distracting information to passages
- Train on modified questions that break spurious correlations

**3. Ensemble Debiasing:**
- Train a "bias model" to capture artifacts
- Train main model to learn residual after removing bias
- Use artifact-aware loss functions

---

## 6. Statistical Validation Summary

Our findings are statistically robust:

| Artifact Type | Chi-Square | P-Value | Significance |
|---------------|------------|---------|--------------|
| Position Bias | 237.21 | < 0.001 | ✅ Highly Significant |
| Question Type Bias | 16.40-167.64 | < 0.001 | ✅ Highly Significant |
| Prediction Type Bias | 1084.87 | < 0.001 | ✅ Highly Significant |

**Overall Artifact Strength Score**: 0.115/1.0 (Low but detectable)

---

## 7. Conclusions and Research Implications

### 7.1 Key Insights
1. **Artifacts are detectable even in undertrained models**, suggesting they are fundamental to the dataset structure
2. **Multiple independent artifact types** combine to enable spurious performance
3. **Statistical significance** validates that these are not random patterns
4. **Position and type biases** are the strongest artifacts identified

### 7.2 Research Contributions
- **Systematic methodology** for artifact detection in QA datasets
- **Statistical validation** of artifact significance  
- **Specific mitigation targets** identified for future work
- **Replicable analysis framework** for other datasets

### 7.3 Limitations
- Analysis based on undertrained model (may miss some artifacts)
- Limited to 1000 examples (subset of full SQuAD)
- Simple statistical tests (more sophisticated methods available)

### 7.4 Future Work
1. **Full-scale baseline training** for comprehensive artifact analysis
2. **Implementation of mitigation strategies** (dataset cartography, adversarial training)
3. **Comparative analysis** with other QA datasets (Natural Questions, MS MARCO)
4. **Human evaluation** of artifact-dependent vs. artifact-resistant models

---

## 8. Files Generated

**Analysis Scripts:**
- `systematic_artifact_analysis.py` - Comprehensive pattern detection
- `model_ablation_analysis.py` - Question/passage-only analysis
- `statistical_artifact_analysis.py` - Statistical significance testing
- `analyze_predictions.py` - Basic prediction analysis

**Results Files:**
- `artifact_analysis_results.json` - Detailed artifact findings
- `ablation_analysis_results.json` - Ablation study results  
- `statistical_analysis_results.json` - Statistical test results
- `question_only_squad.jsonl` - Question-only dataset for testing
- `passage_only_squad.jsonl` - Passage-only dataset for testing

**Model Files:**
- `baseline_model/` - Trained ELECTRA-small checkpoint
- `baseline_eval/` - Evaluation results and predictions

This comprehensive analysis provides a solid foundation for the artifact mitigation phase of the project and demonstrates clear evidence of dataset artifacts that could be exploited by more sophisticated models.