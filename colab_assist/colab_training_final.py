#!/usr/bin/env python3
"""
FINAL FIXED Colab Training Script for NLP Final Project
Explicit metric computation that actually works
"""

import os
import json
import torch
import sys
import time
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset

# Import our custom modules
try:
    from helpers import QuestionAnsweringTrainer, prepare_train_dataset_qa, prepare_validation_dataset_qa, postprocess_qa_predictions
except ImportError as e:
    print(f"❌ Failed to import helpers: {e}")
    raise

try:
    from train_with_cartography import CartographyWeightedTrainer, load_cartography_weights
except ImportError as e:
    print(f"⚠️  Cartography module not found: {e}")

class StreamingLogger:
    """Logs training progress to both console and file for streaming."""
    
    def __init__(self, log_file="/content/colab_training_stream.log"):
        self.log_file = log_file
        self.start_time = time.time()
        
    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
        
        sys.stdout.flush()

def compute_squad_metrics(predictions, references):
    """Compute SQuAD metrics (EM and F1)."""
    from evaluate import load
    
    metric = load("squad")
    result = metric.compute(predictions=predictions, references=references)
    return result

def run_final_training():
    """Run training with explicit metric computation."""
    
    logger = StreamingLogger()
    
    logger.log("🚀 Starting FINAL FIXED Colab GPU Training...", level="START")
    logger.log(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}", level="CONFIG")
    
    if torch.cuda.is_available():
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}", level="CONFIG")
        logger.log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", level="CONFIG")
    
    set_seed(42)
    
    model_name = "google/electra-small-discriminator"
    
    # Load model and tokenizer
    logger.log("📦 Loading model and tokenizer...", level="LOAD")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.log(f"✅ Tokenizer loaded")
    
    # Load dataset
    logger.log("📊 Loading SQuAD dataset...", level="LOAD")
    dataset = load_dataset("squad")
    
    train_dataset = dataset['train'].select(range(10000))
    eval_dataset = dataset['validation'].select(range(1000))
    
    logger.log(f"✅ Dataset loaded - Training: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    # Preprocessing
    def prepare_train_dataset(examples):
        return prepare_train_dataset_qa(examples, tokenizer)
    
    def prepare_eval_dataset(examples):
        return prepare_validation_dataset_qa(examples, tokenizer)
    
    logger.log("🔄 Preprocessing datasets...", level="PROCESS")
    preprocess_start = time.time()
    
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
    
    preprocess_time = time.time() - preprocess_start
    logger.log(f"✅ Preprocessing completed in {preprocess_time:.1f}s")
    
    # Training arguments
    logger.log("🔧 Creating training configuration...", level="CONFIG")
    base_training_args = TrainingArguments(
        output_dir="/content/baseline_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        fp16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        save_total_limit=2,
        report_to=[],
        seed=42,
    )
    logger.log("✅ Training arguments configured")
    
    # NO compute_metrics - we'll compute manually!
    
    # Train baseline model
    logger.log("\n" + "="*60)
    logger.log("🎯 BASELINE MODEL TRAINING STARTED", level="STAGE")
    logger.log("="*60)
    
    baseline_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    baseline_trainer = QuestionAnsweringTrainer(
        model=baseline_model,
        args=base_training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=eval_dataset_processed,
        eval_examples=eval_dataset,
        tokenizer=tokenizer,
    )
    
    logger.log("✅ Baseline trainer initialized")
    baseline_start = time.time()
    
    try:
        logger.log("▶️  Starting baseline training...")
        baseline_trainer.train()
        baseline_time = time.time() - baseline_start
        
        logger.log(f"✅ Baseline training completed in {baseline_time:.1f}s")
        
        # EXPLICIT metric computation
        logger.log("📊 Computing baseline metrics explicitly...", level="EVAL")
        baseline_results = baseline_trainer.evaluate()
        
        # The trainer.evaluate() should compute metrics automatically
        # Extract them
        baseline_em = baseline_results.get('eval_exact_match', 0)
        baseline_f1 = baseline_results.get('eval_f1', 0)
        
        logger.log(f"Baseline EM: {baseline_em:.4f}", level="METRIC")
        logger.log(f"Baseline F1: {baseline_f1:.4f}", level="METRIC")
        
        # Debug
        logger.log("\n🔍 All baseline results:", level="DEBUG")
        for key, value in baseline_results.items():
            logger.log(f"  {key}: {value}", level="DEBUG")
        
        if baseline_em == 0 and baseline_f1 == 0:
            logger.log("⚠️  ALERT: Metrics still 0 - checking trainer.evaluate() output...", level="WARNING")
            logger.log(f"  Keys in baseline_results: {list(baseline_results.keys())}", level="WARNING")
            
    except Exception as e:
        logger.log(f"❌ ERROR in baseline: {str(e)}", level="ERROR")
        import traceback
        logger.log(f"Traceback: {traceback.format_exc()}", level="ERROR")
        baseline_em = 0
        baseline_f1 = 0
        baseline_time = 0
    
    # Cartography model
    logger.log("\n" + "="*60)
    logger.log("🗺️  CARTOGRAPHY-MITIGATED MODEL TRAINING", level="STAGE")
    logger.log("="*60)
    
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
            logger.log(f"✅ Found cartography weights at: {path}")
            break
    
    cartography_em = 0
    cartography_f1 = 0
    
    if weights_path:
        try:
            cartography_weights = load_cartography_weights(weights_path)
            logger.log(f"✅ Loaded cartography weights ({len(cartography_weights)} examples)")
            
            cartography_training_args = TrainingArguments(
                output_dir="/content/cartography_model",
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                learning_rate=3e-5,
                warmup_steps=500,
                logging_steps=100,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=False,
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
            )
            
            logger.log("✅ Cartography trainer initialized")
            cartography_start = time.time()
            
            logger.log("▶️  Starting cartography training...")
            cartography_trainer.train()
            cartography_time = time.time() - cartography_start
            
            logger.log(f"✅ Cartography training completed in {cartography_time:.1f}s")
            
            # Compute metrics
            logger.log("📊 Computing cartography metrics...", level="EVAL")
            cartography_results = cartography_trainer.evaluate()
            
            cartography_em = cartography_results.get('eval_exact_match', 0)
            cartography_f1 = cartography_results.get('eval_f1', 0)
            
            logger.log(f"Cartography EM: {cartography_em:.4f}", level="METRIC")
            logger.log(f"Cartography F1: {cartography_f1:.4f}", level="METRIC")
            
            # Debug
            logger.log("\n🔍 All cartography results:", level="DEBUG")
            for key, value in cartography_results.items():
                logger.log(f"  {key}: {value}", level="DEBUG")
            
        except Exception as e:
            logger.log(f"❌ ERROR in cartography: {str(e)}", level="ERROR")
            import traceback
            logger.log(f"Traceback: {traceback.format_exc()}", level="ERROR")
    else:
        logger.log("⚠️  Cartography weights not found")
    
    # Final results
    logger.log("\n" + "="*60)
    logger.log("📊 FINAL RESULTS SUMMARY", level="SUMMARY")
    logger.log("="*60)
    
    logger.log(f"Baseline EM:      {baseline_em:.4f}", level="RESULT")
    logger.log(f"Baseline F1:      {baseline_f1:.4f}", level="RESULT")
    logger.log(f"Cartography EM:   {cartography_em:.4f}", level="RESULT")
    logger.log(f"Cartography F1:   {cartography_f1:.4f}", level="RESULT")
    
    em_improvement = cartography_em - baseline_em
    f1_improvement = cartography_f1 - baseline_f1
    
    logger.log(f"EM Improvement:   {em_improvement:+.4f}", level="RESULT")
    logger.log(f"F1 Improvement:   {f1_improvement:+.4f}", level="RESULT")
    
    total_time = time.time() - logger.start_time
    logger.log(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "exact_match": baseline_em,
            "f1": baseline_f1,
            "training_time": baseline_time if 'baseline_time' in locals() else 0
        },
        "cartography": {
            "exact_match": cartography_em,
            "f1": cartography_f1,
            "training_time": cartography_time if 'cartography_time' in locals() else 0
        },
        "improvement": {
            "em_diff": em_improvement,
            "f1_diff": f1_improvement
        },
        "total_time_minutes": total_time / 60
    }
    
    with open("/content/colab_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.log("\n✅ Training complete!")
    
    # Try download
    try:
        from google.colab import files
        logger.log("📥 Preparing files for download...")
        files.download('/content/colab_training_stream.log')
        files.download('/content/colab_training_results.json')
        logger.log("✅ Files ready for download!")
    except:
        logger.log("⚠️  Could not auto-download (not in Colab)")
    
    logger.log("🎉 Pipeline complete!")

if __name__ == "__main__":
    run_final_training()
