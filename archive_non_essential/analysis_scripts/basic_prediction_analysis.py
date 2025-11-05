#!/usr/bin/env python3
"""
Quick analysis script to examine prediction patterns and identify potential artifacts.
"""

import json
from collections import Counter, defaultdict
import re

def analyze_predictions(predictions_file):
    """Analyze patterns in model predictions to identify artifacts."""
    
    predictions = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line.strip()))
    
    print(f"Analyzing {len(predictions)} predictions...")
    print("=" * 50)
    
    # 1. Most common predicted answers
    predicted_answers = [pred['predicted_answer'] for pred in predictions]
    answer_counts = Counter(predicted_answers)
    
    print("TOP 10 MOST COMMON PREDICTED ANSWERS:")
    for answer, count in answer_counts.most_common(10):
        percentage = (count / len(predictions)) * 100
        print(f"  '{answer}': {count} times ({percentage:.1f}%)")
    
    print("\n" + "=" * 50)
    
    # 2. Question type analysis
    question_patterns = defaultdict(list)
    for pred in predictions:
        question = pred['question'].lower()
        predicted = pred['predicted_answer']
        
        # Categorize by question word
        if question.startswith('what'):
            question_patterns['what'].append(predicted)
        elif question.startswith('which'):
            question_patterns['which'].append(predicted)
        elif question.startswith('when'):
            question_patterns['when'].append(predicted)
        elif question.startswith('where'):
            question_patterns['where'].append(predicted)
        elif question.startswith('who'):
            question_patterns['who'].append(predicted)
        elif question.startswith('how'):
            question_patterns['how'].append(predicted)
    
    print("PREDICTIONS BY QUESTION TYPE:")
    for q_type, answers in question_patterns.items():
        top_answer = Counter(answers).most_common(1)[0] if answers else ("None", 0)
        print(f"  {q_type.upper()} questions: Most common answer = '{top_answer[0]}' ({top_answer[1]}/{len(answers)})")
    
    print("\n" + "=" * 50)
    
    # 3. Pattern analysis
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    number_pattern = r'\b\d+\b'
    
    date_predictions = [pred for pred in predicted_answers if re.search(date_pattern, pred)]
    number_predictions = [pred for pred in predicted_answers if re.search(number_pattern, pred)]
    
    print("PATTERN ANALYSIS:")
    print(f"  Date format predictions: {len(date_predictions)} ({len(date_predictions)/len(predictions)*100:.1f}%)")
    print(f"  Contains numbers: {len(number_predictions)} ({len(number_predictions)/len(predictions)*100:.1f}%)")
    
    # 4. Sample error analysis
    print("\n" + "=" * 50)
    print("SAMPLE ERROR ANALYSIS (first 5 wrong predictions):")
    
    errors = 0
    for pred in predictions[:20]:  # Check first 20
        correct_answers = [ans.lower().strip() for ans in pred['answers']['text']]
        predicted = pred['predicted_answer'].lower().strip()
        
        if predicted not in correct_answers:
            errors += 1
            if errors <= 5:
                print(f"\nERROR {errors}:")
                print(f"  Question: {pred['question']}")
                print(f"  Correct: {pred['answers']['text']}")
                print(f"  Predicted: '{pred['predicted_answer']}'")
    
    return predictions

if __name__ == "__main__":
    predictions = analyze_predictions("baseline_eval/eval_predictions.jsonl")