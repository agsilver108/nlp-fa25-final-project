"""
Modified Training Script with Dataset Cartography Mitigation

This script extends the base run.py to incorporate dataset cartography weights
for artifact mitigation during training.

Key Features:
- Loads cartography-generated training weights
- Implements weighted sampling during training
- Maintains compatibility with existing evaluation pipeline
- Supports multiple mitigation strategies (upweight_hard, remove_easy, balanced)
"""

import argparse
import json
import logging
import math
import os
import random
import warnings
from pathlib import Path
from typing import Optional, Dict

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import evaluate
from helpers import QuestionAnsweringTrainer, prepare_train_dataset_qa, prepare_validation_dataset_qa

# Setup logging
logger = logging.getLogger(__name__)

# Check dependencies
check_min_version("4.17.0")
require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")


class CartographyWeightedTrainer(QuestionAnsweringTrainer):
    """
    Extended trainer that incorporates dataset cartography weights.
    """
    
    def __init__(self, cartography_weights: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(**kwargs)
        self.cartography_weights = cartography_weights or {}
        
    def get_train_dataloader(self) -> DataLoader:
        """
        Override to create weighted sampler based on cartography analysis.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        # If we have cartography weights, use weighted sampling
        if self.cartography_weights:
            # Extract weights for training examples
            example_weights = []
            
            for i, example in enumerate(self.train_dataset):
                example_id = example.get('id', str(i))
                weight = self.cartography_weights.get(example_id, 1.0)
                example_weights.append(weight)
            
            # Filter out zero weights (effectively removing examples)
            valid_indices = [i for i, w in enumerate(example_weights) if w > 0]
            valid_weights = [example_weights[i] for i in valid_indices]
            
            if len(valid_indices) < len(example_weights):
                logger.info(f"Filtered out {len(example_weights) - len(valid_indices)} examples with zero weight")
                # Create subset dataset
                self.train_dataset = self.train_dataset.select(valid_indices)
                example_weights = valid_weights
            
            # Create weighted sampler
            sampler = WeightedRandomSampler(
                weights=example_weights,
                num_samples=len(example_weights),
                replacement=True
            )
            
            logger.info(f"Using weighted sampling with {len(example_weights)} examples")
            logger.info(f"Weight statistics: min={min(example_weights):.2f}, max={max(example_weights):.2f}, mean={np.mean(example_weights):.2f}")
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=False,  # Disable pin_memory to avoid warning
            )
        else:
            # Use default behavior if no weights provided
            return super().get_train_dataloader()


def load_cartography_weights(weights_file: str) -> Dict[str, float]:
    """Load cartography training weights from JSON file."""
    with open(weights_file, 'r') as f:
        weights = json.load(f)
    
    logger.info(f"Loaded {len(weights)} cartography weights from {weights_file}")
    
    # Print weight distribution statistics
    weight_values = list(weights.values())
    logger.info(f"Weight distribution: min={min(weight_values):.2f}, max={max(weight_values):.2f}, mean={np.mean(weight_values):.2f}")
    
    # Count examples by weight
    weight_counts = {}
    for weight in weight_values:
        weight_counts[weight] = weight_counts.get(weight, 0) + 1
    
    logger.info("Weight distribution:")
    for weight, count in sorted(weight_counts.items()):
        percentage = (count / len(weight_values)) * 100
        if weight == 0.0:
            logger.info(f"  Removed (weight=0.0): {count} examples ({percentage:.1f}%)")
        elif weight == 0.5:
            logger.info(f"  Downweighted (weight=0.5): {count} examples ({percentage:.1f}%)")
        elif weight == 1.0:
            logger.info(f"  Normal weight (weight=1.0): {count} examples ({percentage:.1f}%)")
        elif weight == 2.0:
            logger.info(f"  Upweighted (weight=2.0): {count} examples ({percentage:.1f}%)")
        else:
            logger.info(f"  Custom weight ({weight}): {count} examples ({percentage:.1f}%)")
    
    return weights


def preprocess_function(examples, tokenizer, max_seq_length, doc_stride):
    """Wrapper function for dataset preprocessing."""
    return prepare_train_dataset_qa(examples, tokenizer, max_seq_length)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train QA model with dataset cartography mitigation")
    
    # Base arguments from run.py
    parser.add_argument("--task", default="qa", help="Task type")
    parser.add_argument("--dataset", default="squad", help="Dataset name")
    parser.add_argument("--model", default="google/electra-small-discriminator", help="Model name or path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_train_samples", type=int, help="Maximum training samples")
    parser.add_argument("--max_eval_samples", type=int, help="Maximum evaluation samples")
    parser.add_argument("--max_seq_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--doc_stride", type=int, default=128, help="Document stride for long sequences")
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation")
    parser.add_argument("--per_device_train_batch_size", type=int, default=12, help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=2.0, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=500, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Save frequency")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--eval_strategy", default="steps", help="Evaluation strategy")
    parser.add_argument("--save_strategy", default="steps", help="Save strategy")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end")
    parser.add_argument("--metric_for_best_model", default="eval_f1", help="Metric for best model")
    
    # Dataset cartography arguments
    parser.add_argument("--cartography_weights", type=str, help="Path to cartography weights JSON file")
    parser.add_argument("--cartography_strategy", default="upweight_hard", 
                       choices=["upweight_hard", "remove_easy", "balanced"],
                       help="Cartography mitigation strategy")
    parser.add_argument("--no_cartography", action="store_true", 
                       help="Disable cartography weighting (baseline training)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    if args.dataset == "squad":
        raw_datasets = load_dataset("squad")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer: {args.model}")
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model, config=config)
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    column_names = raw_datasets["train"].column_names
    
    # Preprocess training data
    train_tokenized = raw_datasets["train"].map(
        lambda examples: prepare_train_dataset_qa(examples, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on training dataset",
    )
    
    # Preprocess validation data  
    eval_tokenized = raw_datasets["validation"].map(
        lambda examples: prepare_validation_dataset_qa(examples, tokenizer),
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on validation dataset",
    )
    
    # Prepare training dataset
    train_dataset = train_tokenized
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    
    # Prepare evaluation dataset
    eval_dataset = eval_tokenized
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    
    # Load cartography weights if provided
    cartography_weights = None
    if args.no_cartography:
        logger.info("Cartography disabled - running baseline training")
    elif args.cartography_weights:
        logger.info(f"Loading cartography weights: {args.cartography_weights}")
        cartography_weights = load_cartography_weights(args.cartography_weights)
    elif args.do_train:
        # Try to auto-detect weights file based on strategy
        weights_file = f"results/cartography/training_weights_{args.cartography_strategy}.json"
        if os.path.exists(weights_file):
            logger.info(f"Auto-detected cartography weights: {weights_file}")
            cartography_weights = load_cartography_weights(weights_file)
        else:
            logger.warning(f"No cartography weights found at {weights_file}")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        seed=args.seed,
        report_to=None,  # Disable wandb/tensorboard
        dataloader_pin_memory=False,  # Disable pin_memory to avoid warning
    )
    
    # Data collator
    data_collator = default_data_collator
    
    # Initialize metrics
    metric = evaluate.load("squad")
    
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    # Initialize trainer with cartography weights
    eval_examples = raw_datasets["validation"] if args.do_eval else None
    if args.max_eval_samples and eval_examples:
        eval_examples = eval_examples.select(range(args.max_eval_samples))
    
    trainer = CartographyWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        eval_examples=eval_examples,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        cartography_weights=cartography_weights,
    )
    
    # Training
    if args.do_train:
        logger.info("Starting training with cartography mitigation...")
        if cartography_weights:
            logger.info(f"Using cartography strategy: {args.cartography_strategy}")
        else:
            logger.info("Training without cartography weights (baseline)")
        
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        
        # Log training results
        logger.info("Training completed!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        # Save training metrics
        with open(os.path.join(args.output_dir, "train_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)
    
    # Evaluation
    if args.do_eval:
        logger.info("Starting evaluation...")
        eval_result = trainer.evaluate()
        
        # Log evaluation results
        logger.info("Evaluation completed!")
        logger.info(f"Exact Match: {eval_result.get('eval_exact_match', 0):.4f}")
        logger.info(f"F1 Score: {eval_result.get('eval_f1', 0):.4f}")
        
        # Save evaluation metrics
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_result, f, indent=2)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()