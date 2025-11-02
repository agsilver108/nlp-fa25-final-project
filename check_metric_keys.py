"""
Quick check of what the SQuAD metric returns
"""
print("Checking SQuAD metric output keys...")
print("The evaluate library's 'squad' metric should return:")
print("  - 'exact_match': The exact match score")
print("  - 'f1': The F1 score")
print("")
print("If you have the evaluate library installed, run this to verify:")
print("")
print("from evaluate import load")
print("metric = load('squad')")
print("# Test with dummy data")
print("predictions = [{'id': '1', 'prediction_text': 'test'}]")
print("references = [{'id': '1', 'answers': {'text': ['test'], 'answer_start': [0]}}]")
print("result = metric.compute(predictions=predictions, references=references)")
print("print('Keys returned:', list(result.keys()))")
print("print('Result:', result)")
