#!/usr/bin/env python3
"""
Debug script to understand why metrics are returning 0.
Tests compute_metrics locally without full training.
"""

import json
from transformers import EvalPrediction
from evaluate import load

# Load the SQuAD metric
metric = load("squad")

# Create example predictions in the exact format that trainer provides
formatted_predictions = [
    {"id": "56be4db0ace52140001992a7", "prediction_text": "California"},
    {"id": "56be4db0ace52140001992a8", "prediction_text": "Nevada"},
]

# Create example references in the exact format that trainer provides
references = [
    {"id": "56be4db0ace52140001992a7", "answers": {"text": ["California", "CA"], "answer_start": [123, 456]}},
    {"id": "56be4db0ace52140001992a8", "answers": {"text": ["Nevada", "NV"], "answer_start": [789, 101]}},
]

print("=" * 60)
print("TEST 1: Direct metric computation (how trainer calls it)")
print("=" * 60)

# This is exactly how the trainer calls compute_metrics
eval_pred = EvalPrediction(predictions=formatted_predictions, label_ids=references)

print(f"\nEvalPrediction.predictions type: {type(eval_pred.predictions)}")
print(f"EvalPrediction.predictions[0]: {eval_pred.predictions[0]}")

print(f"\nEvalPrediction.label_ids type: {type(eval_pred.label_ids)}")
print(f"EvalPrediction.label_ids[0]: {eval_pred.label_ids[0]}")

try:
    result = metric.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids)
    print(f"\n✅ Metric computation successful!")
    print(f"Result keys: {list(result.keys())}")
    print(f"Results: {json.dumps(result, indent=2)}")
except Exception as e:
    print(f"\n❌ Metric computation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 2: Test compute_metrics function from colab_streaming_training.py")
print("=" * 60)

def compute_metrics(eval_preds):
    """Compute SQuAD metrics for evaluation."""
    # eval_preds is an EvalPrediction with:
    # - predictions: list of {"id": ..., "prediction_text": ...}
    # - label_ids: list of {"id": ..., "answers": ...}
    predictions = eval_preds.predictions
    references = eval_preds.label_ids
    
    metric = load("squad")
    result = metric.compute(predictions=predictions, references=references)
    return result

try:
    result = compute_metrics(eval_pred)
    print(f"✅ compute_metrics function works!")
    print(f"Result: {json.dumps(result, indent=2)}")
except Exception as e:
    print(f"❌ compute_metrics function failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 3: Check what keys are returned with prefix")
print("=" * 60)

try:
    result = compute_metrics(eval_pred)
    # Add prefix like trainer does
    metric_key_prefix = "eval"
    prefixed_results = {}
    for key in list(result.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            prefixed_results[f"{metric_key_prefix}_{key}"] = result[key]
        else:
            prefixed_results[key] = result[key]
    
    print(f"Prefixed results keys: {list(prefixed_results.keys())}")
    print(f"Prefixed results: {json.dumps(prefixed_results, indent=2)}")
    
    # Check what we'd extract
    em = prefixed_results.get('eval_exact_match', 0)
    f1 = prefixed_results.get('eval_f1', 0)
    print(f"\nExtracted EM: {em}")
    print(f"Extracted F1: {f1}")
    
except Exception as e:
    print(f"❌ Prefix test failed: {e}")
    import traceback
    traceback.print_exc()
