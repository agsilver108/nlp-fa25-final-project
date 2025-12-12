#!/usr/bin/env python3
"""
Baseline Training Script (adapted from colab_training_final.py)
Trains a model WITHOUT cartography weighting for comparison.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset
from helpers import QuestionAnsweringTrainer, prepare_train_dataset_qa, prepare_validation_dataset_qa
from evaluate import load

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def compute_metrics(eval_preds):
    """Compute SQuAD metrics (EM and F1)."""
    metric = load("squad")
    predictions, references = eval_preds.predictions, eval_preds.label_ids
    result = metric.compute(predictions=predictions, references=references)
    return {
        "exact_match": result.get("exact_match", 0),
        "f1": result.get("f1", 0)
    }

def main():
    """Run baseline training without cartography weighting."""
    
    logger.info("="*80)
    logger.info("BASELINE MODEL TRAINING (No Cartography)")
    logger.info("="*80)
    
    # Configuration
    model_name = "google/electra-small-discriminator"
    output_dir = "models/baseline_full"
    num_epochs = 8
    batch_size = 8
    learning_rate = 3e-5
    
    set_seed(42)
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Load dataset
    logger.info("Loading SQuAD dataset...")
    dataset = load_dataset("squad")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    logger.info(f"Dataset loaded - Training: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    # Preprocessing functions
    def prepare_train_dataset(examples):
        return prepare_train_dataset_qa(examples, tokenizer)
    
    def prepare_eval_dataset(examples):
        return prepare_validation_dataset_qa(examples, tokenizer)
    
    logger.info("Preprocessing datasets...")
    train_dataset_processed = train_dataset.map(
        prepare_train_dataset,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training dataset"
    )
    
    eval_dataset_processed = eval_dataset.map(
        prepare_eval_dataset,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation dataset"
    )
    
    logger.info("Preprocessing completed")
    
    # Training arguments
    logger.info("Creating training configuration...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,  # Disabled since we don't have compute_metrics
        save_total_limit=3,
        report_to=[],
        seed=42,
    )
    
    # Initialize trainer (standard QuestionAnsweringTrainer, NO cartography weights)
    logger.info("Initializing baseline trainer (no cartography)...")
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=eval_dataset_processed,
        eval_examples=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Add metrics computation
    )
    
    logger.info("="*80)
    logger.info("Starting baseline training...")
    logger.info("="*80)
    
    # Train
    train_result = trainer.train()
    
    logger.info("Training completed!")
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    
    # Save training results
    train_results_file = os.path.join(output_dir, "train_results.json")
    with open(train_results_file, "w") as f:
        json.dump({
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "train_loss": train_result.metrics.get("train_loss", 0),
            "epoch": train_result.metrics.get("epoch", 0),
        }, f, indent=2)
    
    logger.info(f"Training results saved to {train_results_file}")
    
    # Evaluate
    logger.info("="*80)
    logger.info("Evaluating baseline model...")
    logger.info("="*80)
    
    eval_results = trainer.evaluate()
    
    # Save evaluation results
    eval_results_file = os.path.join(output_dir, "eval_results.json")
    with open(eval_results_file, "w") as f:
        json.dump({
            "eval_exact_match": eval_results.get("eval_exact_match", 0),
            "eval_f1": eval_results.get("eval_f1", 0),
            "eval_loss": eval_results.get("eval_loss", 0),
            "epoch": num_epochs,
        }, f, indent=2)
    
    logger.info(f"Evaluation results saved to {eval_results_file}")
    
    # Print summary
    logger.info("="*80)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Exact Match: {eval_results.get('eval_exact_match', 0):.2f}%")
    logger.info(f"F1 Score: {eval_results.get('eval_f1', 0):.2f}%")
    logger.info(f"Training Time: {train_result.metrics.get('train_runtime', 0)/60:.2f} minutes")
    logger.info("="*80)

if __name__ == "__main__":
    main()
