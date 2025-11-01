#!/usr/bin/env python3
"""
Colab Training Script for NLP Final Project
Optimized for GPU training with baseline and cartography mitigation.
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

# Import our custom modules
from helpers import QuestionAnsweringTrainer, prepare_train_dataset_qa, prepare_validation_dataset_qa
from train_with_cartography import CartographyWeightedTrainer, load_cartography_weights

def run_colab_training():
    """Run fast GPU training in Colab environment."""
    
    print("🚀 Starting Colab GPU Training...")
    
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
    
    # Training configurations
    base_training_args = TrainingArguments(
        output_dir="/content/baseline_model",
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        warmup_steps=200,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,  # Enable mixed precision
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        save_total_limit=2,
        report_to=[],
        seed=42,
    )
    
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
    )
    
    baseline_start = time.time()
    baseline_trainer.train()
    baseline_time = time.time() - baseline_start
    
    # Evaluate baseline
    baseline_results = baseline_trainer.evaluate()
    print(f"✅ Baseline training completed in {baseline_time:.1f}s")
    print(f"Baseline Results: EM={baseline_results.get('eval_exact_match', 0):.3f}, F1={baseline_results.get('eval_f1', 0):.3f}")
    
    # 2. Train cartography-mitigated model
    print("\n🗺️ Training Cartography-Mitigated Model...")
    
    # Load cartography weights
    weights_path = "/content/nlp-final-project/results/cartography/training_weights_upweight_hard.json"
    if os.path.exists(weights_path):
        cartography_weights = load_cartography_weights(weights_path)
        
        cartography_training_args = TrainingArguments(
            output_dir="/content/cartography_model",
            num_train_epochs=2,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=3e-5,
            warmup_steps=200,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
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
        
        with open("/content/colab_training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Training complete! Results saved to /content/colab_training_results.json")
        
    else:
        print("⚠️  Cartography weights not found, skipping mitigated training")

if __name__ == "__main__":
    run_colab_training()