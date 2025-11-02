#!/usr/bin/env python3
"""
FIXED Colab Training Script for NLP Final Project
Optimized for GPU training with COMPLETE error fixing.
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset
import time

# Import our custom modules with error handling
try:
    from helpers import QuestionAnsweringTrainer, prepare_train_dataset_qa, prepare_validation_dataset_qa
    print("✅ Successfully imported helpers")
except ImportError as e:
    print(f"❌ Failed to import helpers: {e}")
    raise

try:
    from train_with_cartography import CartographyWeightedTrainer, load_cartography_weights
    print("✅ Successfully imported cartography modules")
except ImportError as e:
    print(f"❌ Failed to import cartography modules: {e}")
    # Continue without cartography if needed

def run_colab_training():
    """Run fast GPU training in Colab environment with complete error checking."""
    
    print("🚀 Starting FIXED Colab GPU Training...")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    model_name = "google/electra-small-discriminator"
    
    # Load model and tokenizer
    print("📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    print("📊 Loading SQuAD dataset...")
    dataset = load_dataset("squad")
    
    # Prepare datasets with reasonable subset for fast training
    train_dataset = dataset['train'].select(range(10000))  # 10K for speed
    eval_dataset = dataset['validation'].select(range(1000))   # 1K for eval
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Preprocessing functions
    def prepare_train_dataset(examples):
        return prepare_train_dataset_qa(examples, tokenizer)
    
    def prepare_eval_dataset(examples):
        return prepare_validation_dataset_qa(examples, tokenizer)
    
    # Preprocess datasets
    print("🔄 Preprocessing datasets...")
    start_time = time.time()
    
    train_dataset_processed = train_dataset.map(
        prepare_train_dataset,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset_processed = eval_dataset.map(
        prepare_eval_dataset,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    print(f"Preprocessing completed in {time.time() - start_time:.1f}s")
    
    # CRITICAL: Use ONLY the correct argument names for newer transformers
    print("🔧 Creating TrainingArguments with FIXED parameters...")
    base_training_args = TrainingArguments(
        output_dir="/content/baseline_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="epoch",  # CORRECT: eval_strategy (not evaluation_strategy)
        save_strategy="epoch",
        load_best_model_at_end=False,  # Disabled to avoid metric issues
        fp16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        save_total_limit=2,
        report_to=[],
        seed=42,
    )
    
    print("✅ TrainingArguments created successfully")
    
    # Define compute_metrics function
    def compute_metrics(eval_preds):
        """Compute SQuAD metrics for evaluation."""
        from evaluate import load
        
        predictions, references = eval_preds
        
        # Load SQuAD metric
        metric = load("squad")
        
        # Compute metrics
        result = metric.compute(predictions=predictions, references=references)
        
        return result
    
    # 1. Train baseline model
    print("\n🎯 Training Baseline Model...")
    baseline_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    baseline_trainer = QuestionAnsweringTrainer(
        model=baseline_model,
        args=base_training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=eval_dataset_processed,
        eval_examples=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # CRITICAL: Add compute_metrics function
    )
    
    print("✅ Baseline trainer created, starting training...")
    baseline_start = time.time()
    baseline_trainer.train()
    baseline_time = time.time() - baseline_start
    
    # Evaluate baseline
    print("📊 Evaluating baseline model...")
    baseline_results = baseline_trainer.evaluate()
    print(f"✅ Baseline training completed in {baseline_time:.1f}s")
    print(f"Baseline Results: EM={baseline_results.get('eval_exact_match', 0):.3f}, F1={baseline_results.get('eval_f1', 0):.3f}")
    
    # Debug: Print all available metrics
    print("🔍 Available metrics in baseline_results:")
    for key, value in baseline_results.items():
        print(f"  {key}: {value}")
    
    # 2. Train cartography-mitigated model (if weights available)
    print("\n🗺️ Training Cartography-Mitigated Model...")
    
    # Try multiple possible paths for cartography weights
    possible_paths = [
        "/content/nlp-fa25-final-project/results/cartography/training_weights_upweight_hard.json",
        "/content/nlp-final-project/results/cartography/training_weights_upweight_hard.json",
        "./results/cartography/training_weights_upweight_hard.json",
        "results/cartography/training_weights_upweight_hard.json"
    ]
    
    weights_path = None
    for path in possible_paths:
        if os.path.exists(path):
            weights_path = path
            print(f"✅ Found cartography weights at: {path}")
            break
    
    if weights_path:
        try:
            cartography_weights = load_cartography_weights(weights_path)
            
            cartography_training_args = TrainingArguments(
                output_dir="/content/cartography_model",
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                learning_rate=3e-5,
                warmup_steps=500,
                logging_steps=100,
                eval_strategy="epoch",  # CORRECT: eval_strategy (not evaluation_strategy)
                save_strategy="epoch",
                load_best_model_at_end=False,  # Disabled to avoid metric issues
                fp16=True,
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                save_total_limit=2,
                report_to=[],
                seed=42,
            )
            
            cartography_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            cartography_trainer = CartographyWeightedTrainer(
                model=cartography_model,
                args=cartography_training_args,
                train_dataset=train_dataset_processed,
                eval_dataset=eval_dataset_processed,
                eval_examples=eval_dataset,
                tokenizer=tokenizer,
                cartography_weights=cartography_weights,
                compute_metrics=compute_metrics,  # CRITICAL: Add compute_metrics function
            )
            
            cartography_start = time.time()
            cartography_trainer.train()
            cartography_time = time.time() - cartography_start
            
            # Evaluate cartography model
            cartography_results = cartography_trainer.evaluate()
            print(f"✅ Cartography training completed in {cartography_time:.1f}s")
            print(f"Cartography Results: EM={cartography_results.get('eval_exact_match', 0):.3f}, F1={cartography_results.get('eval_f1', 0):.3f}")
            
            # Compare results
            print("\n📊 Comparison:")
            print(f"Baseline:    EM={baseline_results.get('eval_exact_match', 0):.3f}, F1={baseline_results.get('eval_f1', 0):.3f}")
            print(f"Cartography: EM={cartography_results.get('eval_exact_match', 0):.3f}, F1={cartography_results.get('eval_f1', 0):.3f}")
            
            # Save results
            results = {
                "baseline": {
                    "exact_match": baseline_results.get('eval_exact_match', 0),
                    "f1": baseline_results.get('eval_f1', 0),
                    "training_time": baseline_time
                },
                "cartography": {
                    "exact_match": cartography_results.get('eval_exact_match', 0),
                    "f1": cartography_results.get('eval_f1', 0),
                    "training_time": cartography_time
                },
                "improvement": {
                    "em_diff": cartography_results.get('eval_exact_match', 0) - baseline_results.get('eval_exact_match', 0),
                    "f1_diff": cartography_results.get('eval_f1', 0) - baseline_results.get('eval_f1', 0)
                }
            }
            
        except Exception as e:
            print(f"❌ Error in cartography training: {e}")
            # Save baseline-only results
            results = {
                "baseline": {
                    "exact_match": baseline_results.get('eval_exact_match', 0),
                    "f1": baseline_results.get('eval_f1', 0),
                    "training_time": baseline_time
                },
                "cartography": None,
                "improvement": None,
                "error": str(e)
            }
            
    else:
        print("⚠️  Cartography weights not found!")
        print("🔍 Debugging information:")
        print(f"Current working directory: {os.getcwd()}")
        print("Contents of current directory:")
        for item in os.listdir('.'):
            print(f"  {item}")
        if os.path.exists('results'):
            print("Contents of results directory:")
            for item in os.listdir('results'):
                print(f"  results/{item}")
            if os.path.exists('results/cartography'):
                print("Contents of results/cartography directory:")
                for item in os.listdir('results/cartography'):
                    print(f"  results/cartography/{item}")
        
        # Save baseline-only results
        results = {
            "baseline": {
                "exact_match": baseline_results.get('eval_exact_match', 0),
                "f1": baseline_results.get('eval_f1', 0),
                "training_time": baseline_time
            },
            "cartography": None,
            "improvement": None
        }
    
    # Save results to file
    with open("/content/colab_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Training complete! Results saved to /content/colab_training_results.json")
    
    # Final summary
    baseline_f1 = baseline_results.get('eval_f1', 0)
    if baseline_f1 > 0.5:
        print(f"🎉 SUCCESS: Baseline F1 = {baseline_f1:.3f} - Model learned successfully!")
    elif baseline_f1 > 0.1:
        print(f"⚠️  PARTIAL: Baseline F1 = {baseline_f1:.3f} - Some learning occurred")
    else:
        print(f"❌ FAILURE: Baseline F1 = {baseline_f1:.3f} - Model did not learn properly")
        print("🔍 This suggests an issue with:")
        print("  1. Evaluation function")
        print("  2. Model training")
        print("  3. Data preprocessing")

if __name__ == "__main__":
    run_colab_training()