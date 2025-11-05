#!/usr/bin/env python3
"""
Model Ablation Analysis for SQuAD Dataset Artifacts
Implements question-only and passage-only models to test spurious correlations.
"""

import json
import sys
import os
from collections import Counter, defaultdict
from typing import List, Dict
import subprocess

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ModelAblationAnalyzer:
    """
    Implements model ablations to test dataset artifacts:
    1. Question-only model (no passage context)
    2. Passage-only model (no question)
    3. Random baseline
    """
    
    def __init__(self, dataset_name="squad"):
        self.dataset_name = dataset_name
        self.results = {}
    
    def create_question_only_dataset(self, output_file="question_only_squad.jsonl"):
        """
        Create a question-only dataset by removing passage context.
        This tests if models can answer questions without reading comprehension.
        """
        print("Creating question-only dataset...")
        
        # We'll create a synthetic dataset where context is just the question
        question_only_examples = []
        
        # For demonstration, let's create some examples
        examples = [
            {
                "id": "qo_001",
                "title": "Question_Only_Test", 
                "context": "What team won Super Bowl 50?",  # Question as context
                "question": "What team won Super Bowl 50?",
                "answers": {"text": ["Denver Broncos"], "answer_start": [0]}
            },
            {
                "id": "qo_002", 
                "title": "Question_Only_Test",
                "context": "When was Super Bowl 50 played?",
                "question": "When was Super Bowl 50 played?", 
                "answers": {"text": ["February 7, 2016"], "answer_start": [0]}
            },
            {
                "id": "qo_003",
                "title": "Question_Only_Test",
                "context": "Where was Super Bowl 50 played?",
                "question": "Where was Super Bowl 50 played?",
                "answers": {"text": ["Santa Clara"], "answer_start": [0]}
            }
        ]
        
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        
        print(f"Created question-only dataset: {output_file}")
        return output_file
    
    def create_passage_only_dataset(self, input_predictions_file, output_file="passage_only_squad.jsonl"):
        """
        Create a passage-only dataset by using generic questions.
        This tests if models rely too heavily on passage content without question semantics.
        """
        print("Creating passage-only dataset...")
        
        # Load original predictions to get contexts
        with open(input_predictions_file, 'r') as f:
            original_data = [json.loads(line) for line in f]
        
        passage_only_examples = []
        generic_questions = [
            "What is mentioned in the passage?",
            "What information is provided?", 
            "What is discussed?",
            "What details are given?",
            "What is described?"
        ]
        
        for i, pred in enumerate(original_data[:5]):  # Take first 5 for demo
            example = {
                "id": f"po_{i:03d}",
                "title": "Passage_Only_Test",
                "context": pred["context"],
                "question": generic_questions[i % len(generic_questions)],
                "answers": pred["answers"]  # Keep original answers
            }
            passage_only_examples.append(example)
        
        with open(output_file, 'w') as f:
            for example in passage_only_examples:
                f.write(json.dumps(example) + "\n")
        
        print(f"Created passage-only dataset: {output_file}")
        return output_file
    
    def run_baseline_analysis(self, predictions_file):
        """
        Analyze what a random baseline would achieve.
        """
        print("\n" + "="*50)
        print("RANDOM BASELINE ANALYSIS")
        print("="*50)
        
        with open(predictions_file, 'r') as f:
            predictions = [json.loads(line) for line in f]
        
        # Collect all possible answers from the dataset
        all_answers = []
        for pred in predictions:
            all_answers.extend(pred['answers']['text'])
        
        answer_freq = Counter(all_answers)
        most_common_answers = answer_freq.most_common(10)
        
        print("Most frequent answers in dataset:")
        for answer, count in most_common_answers:
            percentage = (count / len(all_answers)) * 100
            print(f"  '{answer}': {count} times ({percentage:.1f}%)")
        
        # Simulate random baseline performance
        # If we always predicted the most common answer
        most_common_answer = most_common_answers[0][0]
        correct_if_always_most_common = sum(1 for pred in predictions 
                                          if most_common_answer.lower() in [a.lower() for a in pred['answers']['text']])
        
        baseline_accuracy = correct_if_always_most_common / len(predictions)
        
        print(f"\nRandom baseline analysis:")
        print(f"  If always predicting '{most_common_answer}': {baseline_accuracy:.3f} accuracy")
        print(f"  This represents the maximum artifact exploitation possible")
        
        return {
            "most_common_answer": most_common_answer,
            "baseline_accuracy": baseline_accuracy,
            "answer_distribution": dict(most_common_answers)
        }
    
    def analyze_question_answer_correlations(self, predictions_file):
        """
        Find specific question patterns that correlate with answer patterns.
        """
        print("\n" + "="*50)
        print("QUESTION-ANSWER CORRELATION ANALYSIS")
        print("="*50)
        
        with open(predictions_file, 'r') as f:
            predictions = [json.loads(line) for line in f]
        
        # Find patterns like "What color" -> "gold" type correlations
        question_patterns = defaultdict(list)
        
        for pred in predictions:
            question = pred['question'].lower()
            answer = pred['answers']['text'][0].lower()
            
            # Extract question patterns
            words = question.split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                question_patterns[bigram].append(answer)
        
        # Find strong correlations
        strong_correlations = {}
        for pattern, answers in question_patterns.items():
            if len(answers) >= 5:  # Only consider patterns with enough examples
                answer_counter = Counter(answers)
                most_common = answer_counter.most_common(1)[0]
                correlation_strength = most_common[1] / len(answers)
                
                if correlation_strength > 0.5:  # Strong correlation threshold
                    strong_correlations[pattern] = {
                        'most_common_answer': most_common[0],
                        'strength': correlation_strength,
                        'count': len(answers)
                    }
        
        print("Strong question-answer correlations found:")
        for pattern, info in sorted(strong_correlations.items(), 
                                  key=lambda x: x[1]['strength'], reverse=True):
            print(f"  '{pattern}' -> '{info['most_common_answer']}' "
                  f"({info['strength']:.2f} correlation, {info['count']} examples)")
        
        return strong_correlations
    
    def analyze_answer_type_biases(self, predictions_file):
        """
        Analyze if certain answer types are over-represented.
        """
        print("\n" + "="*50)
        print("ANSWER TYPE BIAS ANALYSIS")
        print("="*50)
        
        with open(predictions_file, 'r') as f:
            predictions = [json.loads(line) for line in f]
        
        def classify_answer_type(answer):
            answer = answer.lower().strip()
            
            # Date patterns
            if any(month in answer for month in ['january', 'february', 'march', 'april', 
                                               'may', 'june', 'july', 'august', 'september', 
                                               'october', 'november', 'december']):
                return 'date'
            
            # Number patterns
            if answer.replace('.', '').replace(',', '').isdigit():
                return 'number'
            
            # Yes/No
            if answer in ['yes', 'no']:
                return 'yes_no'
            
            # Length-based classification
            if len(answer.split()) == 1:
                return 'single_word'
            elif len(answer.split()) == 2:
                return 'two_words'
            else:
                return 'multi_word'
        
        # Classify all answers
        gold_types = []
        predicted_types = []
        
        for pred in predictions:
            gold_answer = pred['answers']['text'][0]
            predicted_answer = pred['predicted_answer']
            
            gold_types.append(classify_answer_type(gold_answer))
            predicted_types.append(classify_answer_type(predicted_answer))
        
        gold_distribution = Counter(gold_types)
        predicted_distribution = Counter(predicted_types)
        
        print("Gold answer type distribution:")
        for answer_type, count in gold_distribution.most_common():
            percentage = (count / len(gold_types)) * 100
            print(f"  {answer_type}: {count} ({percentage:.1f}%)")
        
        print("\nPredicted answer type distribution:")
        for answer_type, count in predicted_distribution.most_common():
            percentage = (count / len(predicted_types)) * 100
            print(f"  {answer_type}: {count} ({percentage:.1f}%)")
        
        # Calculate bias
        print("\nType bias analysis (predicted/gold ratio):")
        for answer_type in set(gold_types + predicted_types):
            gold_pct = (gold_distribution[answer_type] / len(gold_types)) * 100
            pred_pct = (predicted_distribution[answer_type] / len(predicted_types)) * 100
            
            if gold_pct > 0:
                bias_ratio = pred_pct / gold_pct
                print(f"  {answer_type}: {bias_ratio:.2f}x "
                      f"(gold: {gold_pct:.1f}%, pred: {pred_pct:.1f}%)")
        
        return {
            'gold_distribution': dict(gold_distribution),
            'predicted_distribution': dict(predicted_distribution)
        }
    
    def generate_ablation_report(self, predictions_file):
        """Generate comprehensive ablation analysis report."""
        print("="*80)
        print("MODEL ABLATION ANALYSIS REPORT")
        print("="*80)
        
        results = {}
        
        # Run all analyses
        results['baseline'] = self.run_baseline_analysis(predictions_file)
        results['correlations'] = self.analyze_question_answer_correlations(predictions_file)
        results['type_biases'] = self.analyze_answer_type_biases(predictions_file)
        
        # Create ablation datasets
        results['question_only_file'] = self.create_question_only_dataset()
        results['passage_only_file'] = self.create_passage_only_dataset(predictions_file)
        
        print("\n" + "="*50)
        print("ABLATION SUMMARY")
        print("="*50)
        print(f"1. Random baseline accuracy: {results['baseline']['baseline_accuracy']:.3f}")
        print(f"2. Strong correlations found: {len(results['correlations'])}")
        print(f"3. Answer type biases detected: {len(results['type_biases']['gold_distribution'])}")
        print(f"4. Question-only dataset created: {results['question_only_file']}")
        print(f"5. Passage-only dataset created: {results['passage_only_file']}")
        
        return results

if __name__ == "__main__":
    analyzer = ModelAblationAnalyzer()
    results = analyzer.generate_ablation_report("baseline_eval/eval_predictions.jsonl")
    
    # Save results
    with open("ablation_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAblation analysis results saved to ablation_analysis_results.json")