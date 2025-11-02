# Debugging Zero Metrics Issue - Complete Analysis

## Problem
Training is returning EM=0 and F1=0 for both baseline and cartography models.

## Root Cause Analysis

### What We've Fixed

1. **Added `data_collator`** - The trainer now receives a `DataCollatorWithPadding` object
   - This ensures proper batching and padding of tokenized examples
   - May have been causing malformed predictions

2. **Added comprehensive debug logging** - The `compute_metrics` function now logs:
   - Number of predictions received
   - First prediction and reference examples
   - All metrics returned before prefixing
   - Result keys and values

## Why Metrics Could Be Zero

### Scenario 1: postprocess_qa_predictions returns empty dict
- If all examples are filtered out during postprocessing
- Would result in empty `formatted_predictions` list
- Metric computation would fail or return 0

### Scenario 2: SQuAD metric gets wrong format
- Predictions format: `[{"id": str, "prediction_text": str}, ...]` ✓
- References format: `[{"id": str, "answers": {"text": [...], "answer_start": [...]}}, ...]` ✓
- If this format is wrong, metric returns 0

### Scenario 3: evaluate() not being called
- If `self.compute_metrics` is None during evaluate()
- No metrics would be computed
- However, code checks: `if self.compute_metrics is not None:`

## Next Steps

1. **Run Training in Colab with NEW code**:
   ```python
   !git pull origin main
   exec(open('colab_assist/colab_streaming_training.py').read())
   ```

2. **Check Log Output**:
   - Download `colab_training_stream.log`
   - Look for lines starting with `[HH:MM:SS] [DEBUG] compute_metrics called`
   - This shows:
     - If compute_metrics is being called
     - How many predictions are being passed
     - What the first prediction/reference looks like
     - What keys the metric returns

3. **Common Issues to Look For**:

   a) `compute_metrics called with 0 predictions`
      → postprocess_qa_predictions is returning empty dict
      → check output.predictions for issues

   b) `compute_metrics result keys: []`
      → SQuAD metric is returning empty dict
      → check predictions/references format

   c) `compute_metrics result: {'exact_match': 0.0, 'f1': 0.0}`
      → Format is correct but actual performance is 0
      → This means the model is giving wrong answers

   d) `compute_metrics` not called at all
      → compute_metrics function not passed to trainer
      → Check trainer initialization

## Technical Details

### How QuestionAnsweringTrainer.evaluate() works:

```python
def evaluate(self, ...):
    # 1. Temporarily disable compute_metrics
    compute_metrics = self.compute_metrics
    self.compute_metrics = None
    
    # 2. Run evaluation to get raw start/end logits
    output = self.evaluation_loop(...)
    
    # 3. Restore compute_metrics
    self.compute_metrics = compute_metrics
    
    # 4. If compute_metrics is available:
    if self.compute_metrics is not None:
        # 5. Post-process logits to predictions
        eval_preds = postprocess_qa_predictions(eval_examples, eval_dataset, output.predictions)
        # Returns: {"example_id_1": "answer_text", "example_id_2": "answer_text", ...}
        
        # 6. Format predictions
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in eval_preds.items()]
        
        # 7. Format references
        references = [{"id": ex["id"], "answers": ex['answers']} for ex in eval_examples]
        
        # 8. Compute metrics
        metrics = self.compute_metrics(
            EvalPrediction(predictions=formatted_predictions, label_ids=references)
        )
        
        # 9. Add prefix to keys
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics["eval_" + key] = metrics.pop(key)
        
        return metrics
    else:
        return {}
```

### What compute_metrics receives:

```python
EvalPrediction(
    predictions=[
        {"id": "1", "prediction_text": "The answer"},
        {"id": "2", "prediction_text": "Another answer"},
        ...
    ],
    label_ids=[
        {"id": "1", "answers": {"text": ["The answer", "Answer"], "answer_start": [0, 4]}},
        {"id": "2", "answers": {"text": ["Another answer"], "answer_start": [50]}},
        ...
    ]
)
```

## Expected Output

When working correctly:
```
[HH:MM:SS] [DEBUG] compute_metrics called with 1000 predictions
[HH:MM:SS] [DEBUG]   First prediction: {'id': '56be4db0ace52140001992a7', 'prediction_text': 'California'}
[HH:MM:SS] [DEBUG]   First reference: {'id': '56be4db0ace52140001992a7', 'answers': {'text': ['California'], 'answer_start': [123]}}
[HH:MM:SS] [DEBUG] compute_metrics result keys: ['exact_match', 'f1']
[HH:MM:SS] [DEBUG] compute_metrics result: {'exact_match': 0.45, 'f1': 0.62}
[HH:MM:SS] [METRIC] Baseline EM: 0.4500
[HH:MM:SS] [METRIC] Baseline F1: 0.6200
```

If you see this, training is working! If not, the debug output will tell us exactly what went wrong.
