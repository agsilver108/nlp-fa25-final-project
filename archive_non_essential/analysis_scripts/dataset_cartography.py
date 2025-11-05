"""
Dataset Cartography Implementation for SQuAD Artifact Mitigation

This module implements the dataset cartography methodology from Swayamdipta et al. (2020)
to identify and mitigate dataset artifacts in SQuAD question answering.

Key Concepts:
- Confidence: How confident the model is in its predictions across epochs
- Variability: How much the model's predictions change across epochs  
- Correctness: Whether the model predicts correctly across epochs

Artifact Patterns:
- Easy examples: High confidence, low variability, high correctness
- Hard examples: Low confidence, high variability, low correctness
- Ambiguous examples: Low confidence, high variability, mixed correctness

Mitigation Strategy:
- Upweight hard examples (likely contain genuine linguistic complexity)
- Downweight easy examples (likely exploit spurious correlations)
- Remove or reweight ambiguous examples (may be mislabeled)
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass


@dataclass
class ExampleDynamics:
    """Training dynamics for a single example across epochs."""
    example_id: str
    confidences: List[float]  # Model confidence scores per epoch
    predictions: List[str]    # Model predictions per epoch
    gold_answer: str         # True answer
    question_type: str       # Question type (what, who, when, etc.)
    answer_position: float   # Normalized position in passage (0-1)
    
    @property
    def confidence(self) -> float:
        """Mean confidence across epochs."""
        return np.mean(self.confidences) if self.confidences else 0.0
    
    @property 
    def variability(self) -> float:
        """Standard deviation of confidences across epochs."""
        return np.std(self.confidences) if len(self.confidences) > 1 else 0.0
    
    @property
    def correctness(self) -> float:
        """Fraction of epochs where prediction was correct."""
        if not self.predictions:
            return 0.0
        correct_count = sum(1 for pred in self.predictions if pred.strip().lower() == self.gold_answer.strip().lower())
        return correct_count / len(self.predictions)


class DatasetCartographer:
    """
    Implements dataset cartography for identifying and mitigating artifacts.
    """
    
    def __init__(self, output_dir: str = "results/cartography"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dynamics = {}  # example_id -> ExampleDynamics
        
    def add_epoch_results(self, epoch: int, predictions: Dict, confidences: Dict, 
                         gold_data: Dict) -> None:
        """
        Add training results from one epoch.
        
        Args:
            epoch: Epoch number
            predictions: Dict mapping example_id -> predicted_answer
            confidences: Dict mapping example_id -> confidence_score
            gold_data: Dict mapping example_id -> gold_answer_info
        """
        for example_id in predictions:
            if example_id not in self.dynamics:
                # Initialize dynamics for new example
                gold_info = gold_data.get(example_id, {})
                self.dynamics[example_id] = ExampleDynamics(
                    example_id=example_id,
                    confidences=[],
                    predictions=[],
                    gold_answer=gold_info.get('answer', ''),
                    question_type=self._extract_question_type(gold_info.get('question', '')),
                    answer_position=gold_info.get('answer_position', 0.5)
                )
            
            # Add this epoch's results
            self.dynamics[example_id].confidences.append(confidences.get(example_id, 0.0))
            self.dynamics[example_id].predictions.append(predictions.get(example_id, ''))
    
    def _extract_question_type(self, question: str) -> str:
        """Extract question type from question text."""
        question_lower = question.lower().strip()
        
        if question_lower.startswith(('what', 'which')):
            return 'what'
        elif question_lower.startswith('who'):
            return 'who'
        elif question_lower.startswith('when'):
            return 'when'
        elif question_lower.startswith('where'):
            return 'where'
        elif question_lower.startswith('why'):
            return 'why'
        elif question_lower.startswith('how'):
            return 'how'
        else:
            return 'other'
    
    def analyze_training_dynamics(self) -> Dict:
        """
        Analyze training dynamics to identify data categories.
        
        Returns:
            Dictionary with analysis results and category assignments
        """
        if not self.dynamics:
            raise ValueError("No training dynamics recorded. Call add_epoch_results first.")
        
        # Calculate metrics for all examples
        confidences = [d.confidence for d in self.dynamics.values()]
        variabilities = [d.variability for d in self.dynamics.values()]
        correctness_scores = [d.correctness for d in self.dynamics.values()]
        
        # Define thresholds based on data distribution
        conf_thresh = np.percentile(confidences, 66)  # Top 1/3
        var_thresh = np.percentile(variabilities, 66)  # Top 1/3
        corr_thresh = 0.5  # Majority correct
        
        # Categorize examples
        categories = {
            'easy': [],      # High conf, low var, high correctness
            'hard': [],      # Low conf, high var, low correctness  
            'ambiguous': []  # Low conf, high var, mixed correctness
        }
        
        artifact_patterns = defaultdict(list)
        
        for example_id, dynamics in self.dynamics.items():
            conf = dynamics.confidence
            var = dynamics.variability
            corr = dynamics.correctness
            
            # Categorize based on training dynamics
            if conf >= conf_thresh and var < var_thresh and corr >= corr_thresh:
                categories['easy'].append(example_id)
            elif conf < conf_thresh and var >= var_thresh and corr < corr_thresh:
                categories['hard'].append(example_id)
            else:
                categories['ambiguous'].append(example_id)
            
            # Check for artifact patterns
            if dynamics.answer_position < 0.2 and corr >= corr_thresh:
                artifact_patterns['front_position_bias'].append(example_id)
            
            if dynamics.question_type in ['when'] and corr >= corr_thresh:
                artifact_patterns['question_type_bias'].append(example_id)
        
        analysis_results = {
            'total_examples': len(self.dynamics),
            'category_counts': {cat: len(examples) for cat, examples in categories.items()},
            'category_percentages': {
                cat: len(examples) / len(self.dynamics) * 100 
                for cat, examples in categories.items()
            },
            'thresholds': {
                'confidence': conf_thresh,
                'variability': var_thresh, 
                'correctness': corr_thresh
            },
            'categories': categories,
            'artifact_patterns': dict(artifact_patterns),
            'metrics_summary': {
                'confidence': {'mean': np.mean(confidences), 'std': np.std(confidences)},
                'variability': {'mean': np.mean(variabilities), 'std': np.std(variabilities)},
                'correctness': {'mean': np.mean(correctness_scores), 'std': np.std(correctness_scores)}
            }
        }
        
        # Save analysis results
        with open(self.output_dir / 'cartography_analysis.json', 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            json_safe_results = self._make_json_safe(analysis_results)
            json.dump(json_safe_results, f, indent=2)
        
        return analysis_results
    
    def _make_json_safe(self, obj):
        """Convert numpy types to JSON-safe Python types."""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def generate_training_weights(self, strategy: str = 'upweight_hard') -> Dict[str, float]:
        """
        Generate training weights based on cartography analysis.
        
        Args:
            strategy: Weighting strategy ('upweight_hard', 'remove_easy', 'balanced')
        
        Returns:
            Dictionary mapping example_id -> weight
        """
        if not self.dynamics:
            raise ValueError("Must run analyze_training_dynamics first.")
        
        weights = {}
        
        # Load category assignments
        with open(self.output_dir / 'cartography_analysis.json', 'r') as f:
            analysis = json.load(f)
        
        categories = analysis['categories']
        
        if strategy == 'upweight_hard':
            # Upweight hard examples, normal weight others
            for example_id in self.dynamics:
                if example_id in categories['hard']:
                    weights[example_id] = 2.0  # Double weight for hard examples
                elif example_id in categories['easy']:
                    weights[example_id] = 0.5  # Reduce weight for easy examples
                else:
                    weights[example_id] = 1.0  # Normal weight for ambiguous
                    
        elif strategy == 'remove_easy':
            # Remove easy examples entirely
            for example_id in self.dynamics:
                if example_id in categories['easy']:
                    weights[example_id] = 0.0  # Remove easy examples
                else:
                    weights[example_id] = 1.0  # Keep others
                    
        elif strategy == 'balanced':
            # Balance categories by adjusting weights
            total_examples = len(self.dynamics)
            target_per_category = total_examples // 3
            
            for category, example_ids in categories.items():
                category_weight = target_per_category / len(example_ids) if example_ids else 1.0
                for example_id in example_ids:
                    weights[example_id] = category_weight
        
        # Save weights
        weights_file = self.output_dir / f'training_weights_{strategy}.json'
        with open(weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
        
        print(f"Generated {len(weights)} training weights using '{strategy}' strategy")
        print(f"Weights saved to: {weights_file}")
        
        return weights
    
    def create_cartography_plot(self) -> str:
        """
        Create data map visualization showing confidence vs. variability.
        
        Returns:
            Path to saved plot
        """
        if not self.dynamics:
            raise ValueError("No training dynamics to plot.")
        
        # Prepare data for plotting
        confidences = []
        variabilities = []
        correctness_scores = []
        categories = []
        
        # Load analysis results for categories
        try:
            with open(self.output_dir / 'cartography_analysis.json', 'r') as f:
                analysis = json.load(f)
            category_assignments = analysis['categories']
        except FileNotFoundError:
            # Run analysis if not done yet
            analysis = self.analyze_training_dynamics()
            category_assignments = analysis['categories']
        
        for example_id, dynamics in self.dynamics.items():
            confidences.append(dynamics.confidence)
            variabilities.append(dynamics.variability)
            correctness_scores.append(dynamics.correctness)
            
            # Determine category
            if example_id in category_assignments['easy']:
                categories.append('Easy')
            elif example_id in category_assignments['hard']:
                categories.append('Hard')
            else:
                categories.append('Ambiguous')
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot colored by category
        category_colors = {'Easy': 'green', 'Hard': 'red', 'Ambiguous': 'orange'}
        
        for category in ['Easy', 'Hard', 'Ambiguous']:
            mask = [c == category for c in categories]
            if any(mask):
                plt.scatter(
                    [conf for conf, m in zip(confidences, mask) if m],
                    [var for var, m in zip(variabilities, mask) if m],
                    c=category_colors[category],
                    label=f'{category} (n={sum(mask)})',
                    alpha=0.6,
                    s=30
                )
        
        plt.xlabel('Confidence (Mean across epochs)', fontsize=12)
        plt.ylabel('Variability (Std across epochs)', fontsize=12)
        plt.title('Dataset Cartography: Training Dynamics Analysis', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add quadrant lines for visual reference
        conf_thresh = analysis['thresholds']['confidence']
        var_thresh = analysis['thresholds']['variability']
        plt.axvline(x=conf_thresh, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=var_thresh, color='black', linestyle='--', alpha=0.5)
        
        # Save plot
        plot_path = self.output_dir / 'cartography_data_map.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Data map visualization saved to: {plot_path}")
        return str(plot_path)
    
    def generate_report(self) -> str:
        """
        Generate comprehensive cartography analysis report.
        
        Returns:
            Path to generated report
        """
        if not self.dynamics:
            raise ValueError("No training dynamics to analyze.")
        
        # Ensure analysis is complete
        try:
            with open(self.output_dir / 'cartography_analysis.json', 'r') as f:
                analysis = json.load(f)
        except FileNotFoundError:
            analysis = self.analyze_training_dynamics()
        
        # Generate visualization
        plot_path = self.create_cartography_plot()
        
        # Create detailed report
        report_content = f"""# Dataset Cartography Analysis Report

## Overview

Dataset cartography analysis of SQuAD training dynamics to identify and mitigate dataset artifacts.

**Analysis Date**: {Path().cwd()}  
**Total Examples Analyzed**: {analysis['total_examples']}

## Training Dynamics Summary

### Category Distribution

| Category | Count | Percentage | Description |
|----------|-------|------------|-------------|
| **Easy** | {analysis['category_counts']['easy']} | {analysis['category_percentages']['easy']:.1f}% | High confidence, low variability, high correctness |
| **Hard** | {analysis['category_counts']['hard']} | {analysis['category_percentages']['hard']:.1f}% | Low confidence, high variability, low correctness |
| **Ambiguous** | {analysis['category_counts']['ambiguous']} | {analysis['category_percentages']['ambiguous']:.1f}% | Mixed dynamics, possible mislabeling |

### Thresholds Used

- **Confidence Threshold**: {analysis['thresholds']['confidence']:.3f}
- **Variability Threshold**: {analysis['thresholds']['variability']:.3f}  
- **Correctness Threshold**: {analysis['thresholds']['correctness']:.3f}

## Artifact Pattern Detection

### Position Bias Artifacts
- **Front Position Bias Examples**: {len(analysis['artifact_patterns'].get('front_position_bias', []))}
- These examples have answers in the first 20% of the passage and high correctness
- Indicates potential exploitation of position shortcuts

### Question Type Bias Artifacts  
- **Temporal Question Bias**: {len(analysis['artifact_patterns'].get('question_type_bias', []))}
- "When" questions with high correctness rates
- Suggests stereotypical question-answer associations

## Metrics Distribution

### Confidence
- **Mean**: {analysis['metrics_summary']['confidence']['mean']:.3f}
- **Standard Deviation**: {analysis['metrics_summary']['confidence']['std']:.3f}

### Variability  
- **Mean**: {analysis['metrics_summary']['variability']['mean']:.3f}
- **Standard Deviation**: {analysis['metrics_summary']['variability']['std']:.3f}

### Correctness
- **Mean**: {analysis['metrics_summary']['correctness']['mean']:.3f}
- **Standard Deviation**: {analysis['metrics_summary']['correctness']['std']:.3f}

## Data Map Visualization

![Cartography Data Map]({plot_path})

The data map shows the relationship between confidence and variability:
- **Bottom-right quadrant**: Easy examples (high confidence, low variability)
- **Top-left quadrant**: Hard examples (low confidence, high variability)
- **Top-right quadrant**: Ambiguous examples (variable patterns)

## Mitigation Recommendations

### 1. Upweight Hard Examples Strategy
- **Rationale**: Hard examples likely contain genuine linguistic complexity
- **Implementation**: 2x weight for hard examples, 0.5x for easy examples
- **Expected Impact**: Reduced artifact exploitation, improved generalization

### 2. Remove Easy Examples Strategy  
- **Rationale**: Easy examples may exploit spurious correlations
- **Implementation**: Filter out easy examples from training
- **Expected Impact**: Forces model to learn from complex cases

### 3. Balanced Category Strategy
- **Rationale**: Ensure equal representation across difficulty levels
- **Implementation**: Reweight to balance category sizes
- **Expected Impact**: More robust training distribution

## Generated Files

1. `cartography_analysis.json` - Complete analysis results
2. `training_weights_upweight_hard.json` - Weights for hard example upweighting
3. `training_weights_remove_easy.json` - Weights for easy example removal
4. `training_weights_balanced.json` - Weights for balanced training
5. `cartography_data_map.png` - Visualization of training dynamics

## Next Steps

1. **Train Mitigated Model**: Use generated weights with modified training loop
2. **Compare Performance**: Evaluate artifact reduction vs. QA performance
3. **Iterate Strategy**: Adjust weighting based on results

---

*This analysis uses the dataset cartography methodology from Swayamdipta et al. (2020) to identify training dynamics patterns and mitigate dataset artifacts.*
"""
        
        # Save report
        report_path = self.output_dir / 'cartography_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Comprehensive cartography report saved to: {report_path}")
        return str(report_path)


def create_mock_training_dynamics(predictions_file: str, num_epochs: int = 3) -> DatasetCartographer:
    """
    Create mock training dynamics from baseline evaluation results for demonstration.
    
    In a real implementation, this would collect dynamics during actual training.
    
    Args:
        predictions_file: Path to baseline evaluation predictions
        num_epochs: Number of mock epochs to simulate
    
    Returns:
        Configured DatasetCartographer with mock dynamics
    """
    cartographer = DatasetCartographer()
    
    # Load baseline predictions  
    with open(predictions_file, 'r') as f:
        baseline_data = [json.loads(line) for line in f]
    
    print(f"Creating mock training dynamics for {len(baseline_data)} examples across {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_predictions = {}
        epoch_confidences = {}
        gold_data = {}
        
        for example in baseline_data:
            example_id = example['id']
            
            # Mock prediction evolution (more stable over epochs for easy examples)
            if example.get('predicted_answer', '').lower() == example['answers']['text'][0].lower():
                # Correct prediction - simulate easy example  
                confidence = 0.8 + 0.1 * np.random.random()  # High confidence
                prediction = example['answers']['text'][0]  # Stays correct
            else:
                # Incorrect prediction - simulate hard example
                confidence = 0.3 + 0.4 * np.random.random()  # Variable confidence
                prediction = example.get('predicted_answer', 'unknown')  # May change
            
            # Add some epoch-specific noise
            confidence += 0.1 * np.random.normal() * (epoch + 1) / num_epochs
            confidence = max(0.1, min(0.99, confidence))  # Clamp to valid range
            
            epoch_predictions[example_id] = prediction
            epoch_confidences[example_id] = confidence
            
            if epoch == 0:  # Only set gold data once
                # Calculate answer position
                context = example.get('context', '')
                answer_start = example['answers']['answer_start'][0] if example['answers']['answer_start'] else 0
                answer_position = answer_start / len(context) if context else 0.5
                
                gold_data[example_id] = {
                    'answer': example['answers']['text'][0],
                    'question': example.get('question', ''),
                    'answer_position': answer_position
                }
        
        cartographer.add_epoch_results(epoch, epoch_predictions, epoch_confidences, gold_data)
    
    print(f"Mock training dynamics created successfully!")
    return cartographer


if __name__ == "__main__":
    # Demo usage with baseline evaluation results
    predictions_file = "results/baseline_evaluation/eval_predictions.jsonl"
    
    if Path(predictions_file).exists():
        print("Creating dataset cartography analysis from baseline evaluation...")
        
        # Create mock training dynamics
        cartographer = create_mock_training_dynamics(predictions_file, num_epochs=5)
        
        # Run complete analysis
        analysis_results = cartographer.analyze_training_dynamics()
        
        print(f"\n=== Cartography Analysis Results ===")
        print(f"Total examples: {analysis_results['total_examples']}")
        print(f"Easy examples: {analysis_results['category_counts']['easy']} ({analysis_results['category_percentages']['easy']:.1f}%)")
        print(f"Hard examples: {analysis_results['category_counts']['hard']} ({analysis_results['category_percentages']['hard']:.1f}%)")
        print(f"Ambiguous examples: {analysis_results['category_counts']['ambiguous']} ({analysis_results['category_percentages']['ambiguous']:.1f}%)")
        
        # Generate training weights
        print(f"\n=== Generating Training Weights ===")
        upweight_weights = cartographer.generate_training_weights('upweight_hard')
        remove_weights = cartographer.generate_training_weights('remove_easy')
        balanced_weights = cartographer.generate_training_weights('balanced')
        
        # Create visualization and report
        print(f"\n=== Creating Visualization and Report ===")
        plot_path = cartographer.create_cartography_plot()
        report_path = cartographer.generate_report()
        
        print(f"\n✅ Dataset cartography analysis complete!")
        print(f"📊 Results saved to: results/cartography/")
        print(f"📈 Visualization: {plot_path}")
        print(f"📝 Report: {report_path}")
        
    else:
        print(f"❌ Baseline predictions file not found: {predictions_file}")
        print("Please run baseline evaluation first.")