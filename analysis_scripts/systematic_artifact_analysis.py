#!/usr/bin/env python3
"""
Systematic Dataset Artifact Analysis for SQuAD
Implements multiple artifact analysis methods from the literature.
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

# For numerical operations, we'll use basic Python
def mean(values):
    return sum(values) / len(values) if values else 0

def histogram(values, bins=10):
    """Simple histogram implementation."""
    min_val, max_val = min(values), max(values)
    bin_width = (max_val - min_val) / bins
    bins_count = [0] * bins
    
    for val in values:
        bin_idx = min(int((val - min_val) / bin_width), bins - 1)
        bins_count[bin_idx] += 1
    
    return bins_count

class SQuADArtifactAnalyzer:
    """
    Comprehensive artifact analysis for SQuAD dataset.
    Implements various methods to detect spurious correlations and dataset biases.
    """
    
    def __init__(self, predictions_file: str):
        """Load predictions and initialize analyzer."""
        self.predictions = []
        with open(predictions_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.predictions.append(json.loads(line.strip()))
        
        print(f"Loaded {len(self.predictions)} predictions for analysis")
        
    def analyze_lexical_overlap(self) -> Dict:
        """
        Analyze lexical overlap between questions and correct answers.
        High overlap might indicate models are using shallow matching.
        """
        print("\n" + "="*60)
        print("LEXICAL OVERLAP ANALYSIS")
        print("="*60)
        
        overlaps = []
        for pred in self.predictions:
            question_words = set(pred['question'].lower().split())
            
            # Calculate overlap with each correct answer
            max_overlap = 0
            for answer in pred['answers']['text']:
                answer_words = set(answer.lower().split())
                overlap = len(question_words.intersection(answer_words))
                max_overlap = max(max_overlap, overlap)
            
            overlaps.append(max_overlap)
        
        mean_overlap = mean(overlaps)
        print(f"Average lexical overlap: {mean_overlap:.2f} words")
        print(f"Examples with high overlap (>3 words): {sum(1 for o in overlaps if o > 3)}")
        
        # Show examples of high overlap
        high_overlap_examples = [(pred, overlap) for pred, overlap in zip(self.predictions, overlaps) if overlap > 3]
        
        print(f"\nSample high-overlap examples:")
        for pred, overlap in high_overlap_examples[:3]:
            print(f"  Overlap: {overlap} words")
            print(f"  Q: {pred['question']}")
            print(f"  A: {pred['answers']['text'][0]}")
            print()
        
        return {"mean_overlap": mean_overlap, "high_overlap_count": sum(1 for o in overlaps if o > 3)}
    
    def analyze_question_patterns(self) -> Dict:
        """
        Analyze patterns in question types and their associated answer patterns.
        Based on "competency problems" framework (Gardner et al., 2021).
        """
        print("\n" + "="*60)
        print("QUESTION PATTERN ANALYSIS")
        print("="*60)
        
        # Extract question patterns
        question_patterns = defaultdict(list)
        answer_types = defaultdict(list)
        
        for pred in self.predictions:
            question = pred['question'].lower()
            correct_answer = pred['answers']['text'][0]
            
            # Categorize by question start
            for pattern in ['what', 'which', 'when', 'where', 'who', 'how', 'why']:
                if question.startswith(pattern):
                    question_patterns[pattern].append(correct_answer)
                    
                    # Classify answer type
                    if self._is_date(correct_answer):
                        answer_types[pattern].append('date')
                    elif self._is_number(correct_answer):
                        answer_types[pattern].append('number')
                    elif self._is_person(correct_answer):
                        answer_types[pattern].append('person')
                    elif self._is_location(correct_answer):
                        answer_types[pattern].append('location')
                    else:
                        answer_types[pattern].append('other')
                    break
        
        # Analyze patterns
        results = {}
        for pattern in question_patterns:
            total = len(question_patterns[pattern])
            type_dist = Counter(answer_types[pattern])
            
            print(f"\n{pattern.upper()} questions ({total} total):")
            for answer_type, count in type_dist.most_common():
                percentage = (count / total) * 100
                print(f"  {answer_type}: {count} ({percentage:.1f}%)")
            
            results[pattern] = dict(type_dist)
        
        return results
    
    def analyze_position_bias(self) -> Dict:
        """
        Analyze position bias - do answers tend to appear in specific locations?
        """
        print("\n" + "="*60)
        print("POSITION BIAS ANALYSIS")
        print("="*60)
        
        positions = []
        answer_lengths = []
        
        for pred in self.predictions:
            context = pred['context']
            context_length = len(context.split())
            
            for answer_start in pred['answers']['answer_start']:
                # Calculate relative position (0-1)
                relative_pos = answer_start / len(context)
                positions.append(relative_pos)
                
                # Get answer from context
                answer_text = pred['answers']['text'][0]
                answer_length = len(answer_text.split())
                answer_lengths.append(answer_length)
        
        # Analyze position distribution
        pos_bins = histogram(positions, bins=10)
        
        print("Answer position distribution (by decile):")
        for i, count in enumerate(pos_bins):
            percentage = (count / len(positions)) * 100
            print(f"  {i*10}-{(i+1)*10}%: {count} answers ({percentage:.1f}%)")
        
        print(f"\nAverage answer position: {mean(positions):.2f} (0=start, 1=end)")
        print(f"Average answer length: {mean(answer_lengths):.1f} words")
        
        return {
            "mean_position": mean(positions),
            "position_distribution": pos_bins,
            "mean_answer_length": mean(answer_lengths)
        }
    
    def analyze_ngram_correlations(self) -> Dict:
        """
        Find spurious n-gram correlations between questions and answers.
        Based on "competency problems" framework.
        """
        print("\n" + "="*60)
        print("N-GRAM CORRELATION ANALYSIS")
        print("="*60)
        
        # Extract question-answer pairs
        questions = []
        answers = []
        
        for pred in self.predictions:
            questions.append(pred['question'].lower())
            answers.append(pred['answers']['text'][0].lower())
        
        # Find most common question trigrams
        question_trigrams = []
        for question in questions:
            words = question.split()
            for i in range(len(words) - 2):
                trigram = ' '.join(words[i:i+3])
                question_trigrams.append(trigram)
        
        common_trigrams = Counter(question_trigrams).most_common(10)
        
        print("Most common question trigrams:")
        for trigram, count in common_trigrams:
            percentage = (count / len(question_trigrams)) * 100
            print(f"  '{trigram}': {count} ({percentage:.1f}%)")
        
        # Find correlations between question patterns and answer types
        correlations = {}
        for trigram, _ in common_trigrams[:5]:  # Top 5 trigrams
            matching_indices = [i for i, q in enumerate(questions) if trigram in q]
            if len(matching_indices) > 5:  # Only analyze if enough examples
                matching_answers = [answers[i] for i in matching_indices]
                answer_pattern = self._get_dominant_answer_pattern(matching_answers)
                correlations[trigram] = {
                    'count': len(matching_indices),
                    'dominant_pattern': answer_pattern
                }
        
        print(f"\nQuestion-Answer correlations:")
        for trigram, info in correlations.items():
            print(f"  '{trigram}' -> {info['dominant_pattern']} ({info['count']} examples)")
        
        return correlations
    
    def analyze_model_predictions_vs_gold(self) -> Dict:
        """
        Compare model predictions against gold answers to identify systematic errors.
        """
        print("\n" + "="*60)
        print("MODEL PREDICTION ANALYSIS")
        print("="*60)
        
        prediction_types = []
        gold_types = []
        correct_predictions = 0
        
        for pred in self.predictions:
            predicted = pred['predicted_answer'].lower().strip()
            gold = pred['answers']['text'][0].lower().strip()
            
            if predicted == gold:
                correct_predictions += 1
            
            # Classify prediction and gold types
            pred_type = self._classify_answer_type(predicted)
            gold_type = self._classify_answer_type(gold)
            
            prediction_types.append(pred_type)
            gold_types.append(gold_type)
        
        # Analyze type confusions
        type_confusion = defaultdict(lambda: defaultdict(int))
        for pred_type, gold_type in zip(prediction_types, gold_types):
            type_confusion[gold_type][pred_type] += 1
        
        accuracy = correct_predictions / len(self.predictions)
        print(f"Overall accuracy: {accuracy:.3f}")
        
        print(f"\nType confusion matrix:")
        print("Gold -> Predicted")
        for gold_type in sorted(type_confusion.keys()):
            print(f"\n{gold_type}:")
            total = sum(type_confusion[gold_type].values())
            for pred_type, count in sorted(type_confusion[gold_type].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                print(f"  -> {pred_type}: {count} ({percentage:.1f}%)")
        
        return {
            "accuracy": accuracy,
            "type_confusion": dict(type_confusion)
        }
    
    def generate_contrast_examples(self, num_examples: int = 5) -> List[Dict]:
        """
        Generate contrast examples by modifying questions slightly.
        Based on Gardner et al. (2020) contrast sets approach.
        """
        print("\n" + "="*60)
        print("CONTRAST EXAMPLE GENERATION")
        print("="*60)
        
        contrast_examples = []
        
        # Find examples with clear patterns we can modify
        for pred in self.predictions[:50]:  # Check first 50
            question = pred['question']
            context = pred['context']
            answer = pred['answers']['text'][0]
            
            # Generate contrasts by:
            # 1. Adding negation
            if 'what' in question.lower() and 'not' not in question.lower():
                contrast_q = question.replace('What', 'What was not')
                contrast_examples.append({
                    'original_question': question,
                    'contrast_question': contrast_q,
                    'context': context,
                    'original_answer': answer,
                    'type': 'negation'
                })
            
            # 2. Changing question type
            if question.lower().startswith('when'):
                contrast_q = question.replace('When', 'Where', 1)
                contrast_examples.append({
                    'original_question': question,
                    'contrast_question': contrast_q,
                    'context': context,
                    'original_answer': answer,
                    'type': 'question_type_change'
                })
            
            if len(contrast_examples) >= num_examples:
                break
        
        print(f"Generated {len(contrast_examples)} contrast examples:")
        for i, example in enumerate(contrast_examples):
            print(f"\nExample {i+1} ({example['type']}):")
            print(f"  Original: {example['original_question']}")
            print(f"  Contrast: {example['contrast_question']}")
            print(f"  Answer: {example['original_answer']}")
        
        return contrast_examples
    
    def _is_date(self, text: str) -> bool:
        """Check if text contains a date pattern."""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}\b'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns)
    
    def _is_number(self, text: str) -> bool:
        """Check if text is primarily numeric."""
        return bool(re.search(r'^\d+(\.\d+)?$', text.strip()))
    
    def _is_person(self, text: str) -> bool:
        """Simple heuristic to detect person names."""
        words = text.split()
        if len(words) >= 2:
            return all(word[0].isupper() for word in words if word.isalpha())
        return False
    
    def _is_location(self, text: str) -> bool:
        """Simple heuristic to detect locations."""
        location_indicators = ['stadium', 'city', 'state', 'country', 'street', 'avenue', 'university']
        return any(indicator in text.lower() for indicator in location_indicators)
    
    def _classify_answer_type(self, text: str) -> str:
        """Classify answer into type categories."""
        if self._is_date(text):
            return 'date'
        elif self._is_number(text):
            return 'number'
        elif self._is_person(text):
            return 'person'
        elif self._is_location(text):
            return 'location'
        else:
            return 'other'
    
    def _get_dominant_answer_pattern(self, answers: List[str]) -> str:
        """Find the dominant pattern in a list of answers."""
        types = [self._classify_answer_type(answer) for answer in answers]
        return Counter(types).most_common(1)[0][0]
    
    def generate_full_report(self) -> Dict:
        """Generate comprehensive artifact analysis report."""
        print("="*80)
        print("COMPREHENSIVE SQUAD ARTIFACT ANALYSIS REPORT")
        print("="*80)
        
        results = {}
        
        # Run all analyses
        results['lexical_overlap'] = self.analyze_lexical_overlap()
        results['question_patterns'] = self.analyze_question_patterns()
        results['position_bias'] = self.analyze_position_bias()
        results['ngram_correlations'] = self.analyze_ngram_correlations()
        results['prediction_analysis'] = self.analyze_model_predictions_vs_gold()
        results['contrast_examples'] = self.generate_contrast_examples()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY OF FINDINGS")
        print("="*60)
        print(f"1. Average lexical overlap: {results['lexical_overlap']['mean_overlap']:.2f} words")
        print(f"2. Model accuracy: {results['prediction_analysis']['accuracy']:.3f}")
        print(f"3. Average answer position: {results['position_bias']['mean_position']:.2f}")
        print(f"4. High lexical overlap examples: {results['lexical_overlap']['high_overlap_count']}")
        print(f"5. Generated contrast examples: {len(results['contrast_examples'])}")
        
        return results

if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = SQuADArtifactAnalyzer("baseline_eval/eval_predictions.jsonl")
    results = analyzer.generate_full_report()
    
    # Save results
    with open("artifact_analysis_results.json", "w") as f:
        # Convert any non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if key != 'contrast_examples':  # Skip complex objects
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to artifact_analysis_results.json")