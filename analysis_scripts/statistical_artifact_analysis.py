#!/usr/bin/env python3
"""
Statistical Analysis of Dataset Artifacts
Implements statistical tests to validate artifact findings.
"""

import json
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class StatisticalArtifactAnalyzer:
    """
    Implements statistical tests to validate dataset artifact findings.
    """
    
    def __init__(self, predictions_file: str, results_file: str = None):
        """Load predictions and previous analysis results."""
        with open(predictions_file, 'r', encoding='utf-8') as f:
            self.predictions = [json.loads(line) for line in f]
        
        if results_file:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.previous_results = json.load(f)
        else:
            self.previous_results = {}
        
        print(f"Loaded {len(self.predictions)} predictions for statistical analysis")
    
    def chi_square_test(self, observed: List[int], expected: List[int] = None) -> Tuple[float, float]:
        """
        Simple chi-square test implementation.
        Tests if observed distribution differs significantly from expected.
        """
        if expected is None:
            # Uniform distribution
            expected = [sum(observed) / len(observed)] * len(observed)
        
        if len(observed) != len(expected):
            raise ValueError("Observed and expected must have same length")
        
        chi_square = 0
        for obs, exp in zip(observed, expected):
            if exp > 0:
                chi_square += ((obs - exp) ** 2) / exp
        
        # Degrees of freedom
        df = len(observed) - 1
        
        # Simple p-value approximation (not exact, but indicative)
        # For df=1: critical value ≈ 3.84 (p=0.05), 6.64 (p=0.01)
        # For df=2: critical value ≈ 5.99 (p=0.05), 9.21 (p=0.01)
        critical_values = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07, 
                          6: 12.59, 7: 14.07, 8: 15.51, 9: 16.92}
        
        critical_val = critical_values.get(df, 15.51)  # Default for df > 9
        p_value = 0.01 if chi_square > critical_val else 0.05
        
        return chi_square, p_value
    
    def test_question_type_bias(self) -> Dict:
        """
        Test if question types have significantly biased answer distributions.
        """
        print("\n" + "="*60)
        print("STATISTICAL TEST: Question Type Bias")
        print("="*60)
        
        # Collect question-answer pairs
        question_answer_pairs = defaultdict(list)
        
        for pred in self.predictions:
            question = pred['question'].lower()
            answer_type = self._classify_answer_simple(pred['answers']['text'][0])
            
            # Get question type
            q_type = 'other'
            for qtype in ['what', 'which', 'when', 'where', 'who', 'how', 'why']:
                if question.startswith(qtype):
                    q_type = qtype
                    break
            
            question_answer_pairs[q_type].append(answer_type)
        
        # Test each question type for bias
        results = {}
        overall_distribution = Counter()
        total_answers = 0
        
        # Calculate overall answer type distribution
        for answers in question_answer_pairs.values():
            for answer_type in answers:
                overall_distribution[answer_type] += 1
                total_answers += 1
        
        print("Testing each question type for answer bias:")
        
        for q_type, answers in question_answer_pairs.items():
            if len(answers) < 10:  # Skip if too few examples
                continue
            
            # Count answer types for this question type
            observed_counts = Counter(answers)
            
            # Expected counts based on overall distribution
            expected_counts = {}
            for answer_type in observed_counts:
                expected_prob = overall_distribution[answer_type] / total_answers
                expected_counts[answer_type] = expected_prob * len(answers)
            
            # Prepare for chi-square test
            answer_types = list(observed_counts.keys())
            observed = [observed_counts[at] for at in answer_types]
            expected = [expected_counts[at] for at in answer_types]
            
            try:
                chi_square, p_value = self.chi_square_test(observed, expected)
                
                # Find most biased answer type
                max_bias_ratio = 0
                most_biased_type = None
                for at in answer_types:
                    if expected_counts[at] > 0:
                        bias_ratio = observed_counts[at] / expected_counts[at]
                        if bias_ratio > max_bias_ratio:
                            max_bias_ratio = bias_ratio
                            most_biased_type = at
                
                results[q_type] = {
                    'chi_square': chi_square,
                    'p_value': p_value,
                    'significant': chi_square > 3.84,  # p < 0.05
                    'most_biased_type': most_biased_type,
                    'bias_ratio': max_bias_ratio,
                    'sample_size': len(answers)
                }
                
                significance = "SIGNIFICANT" if chi_square > 3.84 else "not significant"
                print(f"  {q_type.upper()}: χ² = {chi_square:.2f}, {significance}")
                print(f"    Most biased: {most_biased_type} ({max_bias_ratio:.2f}x expected)")
                
            except Exception as e:
                print(f"  {q_type.upper()}: Error in test - {e}")
                results[q_type] = {'error': str(e)}
        
        return results
    
    def test_position_bias_significance(self) -> Dict:
        """
        Test if answer positions are significantly non-uniform.
        """
        print("\n" + "="*60)
        print("STATISTICAL TEST: Position Bias")
        print("="*60)
        
        positions = []
        for pred in self.predictions:
            context = pred['context']
            for answer_start in pred['answers']['answer_start']:
                relative_pos = answer_start / len(context)
                positions.append(relative_pos)
        
        # Create bins (deciles)
        bins = [0] * 10
        for pos in positions:
            bin_idx = min(int(pos * 10), 9)
            bins[bin_idx] += 1
        
        # Test against uniform distribution
        chi_square, p_value = self.chi_square_test(bins)
        
        # Find most overrepresented positions
        expected_per_bin = len(positions) / 10
        max_overrep = 0
        most_overrep_bin = 0
        
        for i, count in enumerate(bins):
            overrep = count / expected_per_bin
            if overrep > max_overrep:
                max_overrep = overrep
                most_overrep_bin = i
        
        significance = "SIGNIFICANT" if chi_square > 3.84 else "not significant"
        print(f"Position distribution test: χ² = {chi_square:.2f}, {significance}")
        print(f"Most overrepresented position: {most_overrep_bin*10}-{(most_overrep_bin+1)*10}% ({max_overrep:.2f}x expected)")
        
        return {
            'chi_square': chi_square,
            'p_value': p_value,
            'significant': chi_square > 3.84,
            'position_distribution': bins,
            'most_overrepresented_bin': most_overrep_bin,
            'overrepresentation_ratio': max_overrep
        }
    
    def test_model_prediction_bias(self) -> Dict:
        """
        Test if model predictions show significant bias compared to gold distribution.
        """
        print("\n" + "="*60)
        print("STATISTICAL TEST: Model Prediction Bias")
        print("="*60)
        
        gold_types = []
        predicted_types = []
        
        for pred in self.predictions:
            gold_answer = pred['answers']['text'][0]
            predicted_answer = pred['predicted_answer']
            
            gold_types.append(self._classify_answer_simple(gold_answer))
            predicted_types.append(self._classify_answer_simple(predicted_answer))
        
        # Count distributions
        gold_dist = Counter(gold_types)
        pred_dist = Counter(predicted_types)
        
        # Get all answer types
        all_types = set(gold_types + predicted_types)
        
        # Prepare for chi-square test
        gold_counts = [gold_dist[at] for at in all_types]
        pred_counts = [pred_dist[at] for at in all_types]
        
        chi_square, p_value = self.chi_square_test(pred_counts, gold_counts)
        
        # Calculate bias ratios
        bias_ratios = {}
        for at in all_types:
            if gold_dist[at] > 0:
                bias_ratios[at] = pred_dist[at] / gold_dist[at]
            else:
                bias_ratios[at] = float('inf') if pred_dist[at] > 0 else 0
        
        # Find most biased types
        most_overrep = max(bias_ratios.items(), key=lambda x: x[1])
        most_underrep = min(bias_ratios.items(), key=lambda x: x[1])
        
        significance = "SIGNIFICANT" if chi_square > 3.84 else "not significant"
        print(f"Prediction bias test: χ² = {chi_square:.2f}, {significance}")
        print(f"Most over-predicted type: {most_overrep[0]} ({most_overrep[1]:.2f}x)")
        print(f"Most under-predicted type: {most_underrep[0]} ({most_underrep[1]:.2f}x)")
        
        return {
            'chi_square': chi_square,
            'p_value': p_value,
            'significant': chi_square > 3.84,
            'bias_ratios': bias_ratios,
            'most_overrepresented': most_overrep,
            'most_underrepresented': most_underrep
        }
    
    def calculate_artifact_strength_scores(self) -> Dict:
        """
        Calculate overall artifact strength scores.
        """
        print("\n" + "="*60)
        print("ARTIFACT STRENGTH SCORING")
        print("="*60)
        
        scores = {}
        
        # 1. Position bias strength (0-1 scale)
        positions = []
        for pred in self.predictions:
            context = pred['context']
            for answer_start in pred['answers']['answer_start']:
                relative_pos = answer_start / len(context)
                positions.append(relative_pos)
        
        # Calculate standard deviation (higher = more uniform, lower = more biased)
        mean_pos = sum(positions) / len(positions)
        variance = sum((p - mean_pos) ** 2 for p in positions) / len(positions)
        std_dev = math.sqrt(variance)
        
        # Convert to bias score (1 - normalized_std_dev)
        max_std = 0.289  # Theoretical max for uniform distribution in [0,1]
        position_bias_score = max(0, 1 - (std_dev / max_std))
        
        scores['position_bias'] = position_bias_score
        
        # 2. Answer type bias strength
        gold_types = [self._classify_answer_simple(pred['answers']['text'][0]) for pred in self.predictions]
        pred_types = [self._classify_answer_simple(pred['predicted_answer']) for pred in self.predictions]
        
        gold_dist = Counter(gold_types)
        pred_dist = Counter(pred_types)
        
        # Calculate KL divergence (information theoretic measure of difference)
        kl_div = 0
        total_pred = sum(pred_dist.values())
        total_gold = sum(gold_dist.values())
        
        for answer_type in set(gold_types + pred_types):
            p_gold = gold_dist[answer_type] / total_gold
            p_pred = pred_dist[answer_type] / total_pred
            
            if p_pred > 0 and p_gold > 0:
                kl_div += p_pred * math.log(p_pred / p_gold)
        
        # Normalize KL divergence to 0-1 scale (approximate)
        type_bias_score = min(1.0, kl_div / 2.0)  # Divide by 2 for rough normalization
        
        scores['type_bias'] = type_bias_score
        
        # 3. Overall artifact score (weighted average)
        overall_score = (position_bias_score * 0.3 + type_bias_score * 0.7)
        scores['overall_artifact_strength'] = overall_score
        
        print(f"Position bias score: {position_bias_score:.3f}")
        print(f"Type bias score: {type_bias_score:.3f}")
        print(f"Overall artifact strength: {overall_score:.3f}")
        
        if overall_score > 0.5:
            print("HIGH artifact presence detected")
        elif overall_score > 0.3:
            print("MODERATE artifact presence detected")
        else:
            print("LOW artifact presence detected")
        
        return scores
    
    def _classify_answer_simple(self, answer: str) -> str:
        """Simple answer classification."""
        answer = answer.lower().strip()
        
        # Date patterns
        if any(month in answer for month in ['january', 'february', 'march', 'april',
                                           'may', 'june', 'july', 'august', 'september',
                                           'october', 'november', 'december']):
            return 'date'
        
        # Number patterns
        if answer.replace('.', '').replace(',', '').replace(' ', '').isdigit():
            return 'number'
        
        # Person (simple heuristic)
        words = answer.split()
        if len(words) == 2 and all(w[0].isupper() for w in words if w.isalpha()):
            return 'person'
        
        # Location indicators
        if any(loc in answer for loc in ['stadium', 'university', 'city', 'street']):
            return 'location'
        
        return 'other'
    
    def generate_statistical_report(self) -> Dict:
        """Generate comprehensive statistical analysis report."""
        print("="*80)
        print("STATISTICAL ARTIFACT ANALYSIS REPORT")
        print("="*80)
        
        results = {}
        
        # Run all statistical tests
        results['question_type_bias'] = self.test_question_type_bias()
        results['position_bias'] = self.test_position_bias_significance()
        results['prediction_bias'] = self.test_model_prediction_bias()
        results['artifact_scores'] = self.calculate_artifact_strength_scores()
        
        # Summary of significant findings
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE SUMMARY")
        print("="*60)
        
        significant_tests = []
        
        if results['position_bias']['significant']:
            significant_tests.append("Position bias")
        
        if results['prediction_bias']['significant']:
            significant_tests.append("Prediction type bias")
        
        significant_qtypes = [qtype for qtype, data in results['question_type_bias'].items()
                            if isinstance(data, dict) and data.get('significant', False)]
        
        if significant_qtypes:
            significant_tests.append(f"Question type bias ({', '.join(significant_qtypes)})")
        
        print(f"Statistically significant artifacts found: {len(significant_tests)}")
        for test in significant_tests:
            print(f"  - {test}")
        
        overall_score = results['artifact_scores']['overall_artifact_strength']
        print(f"\nOverall artifact strength: {overall_score:.3f}/1.0")
        
        return results

if __name__ == "__main__":
    analyzer = StatisticalArtifactAnalyzer(
        "baseline_eval/eval_predictions.jsonl",
        "artifact_analysis_results.json"
    )
    
    results = analyzer.generate_statistical_report()
    
    # Save results
    with open("statistical_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nStatistical analysis results saved to statistical_analysis_results.json")