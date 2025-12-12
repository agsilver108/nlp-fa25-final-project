#!/usr/bin/env python3
"""
Complete Training, Evaluation, Comparison, and Visualization Pipeline
Combines all training and analysis steps in proper order.

This script:
1. Trains baseline model (no cartography)
2. Trains cartography-mitigated model
3. Evaluates both models
4. Compares results
5. Visualizes results with dashboard
"""

import os
import sys
import json
import logging
import time
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
from train_with_cartography import CartographyWeightedTrainer, load_cartography_weights
from evaluate import load

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def compute_metrics(eval_preds):
    """Compute SQuAD metrics (EM and F1)."""
    metric = load("squad")
    predictions, references = eval_preds.predictions, eval_preds.label_ids
    result = metric.compute(predictions=predictions, references=references)
    return {
        "exact_match": result.get("exact_match", 0),
        "f1": result.get("f1", 0)
    }


def train_model(model_name, output_dir, train_dataset, eval_dataset, eval_examples, 
                tokenizer, cartography_weights=None, model_type="baseline"):
    """Train a single model and return results."""
    
    logger.info("="*80)
    logger.info(f"Training {model_type.upper()} model")
    logger.info("="*80)
    
    # Create fresh model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Training arguments (matching Colab)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=16,  # Match Colab
        per_device_eval_batch_size=32,   # Match Colab
        learning_rate=3e-5,
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=2,
        report_to=[],
        seed=42,
    )
    
    # Choose trainer type based on cartography weights
    if cartography_weights:
        logger.info(f"Using CartographyWeightedTrainer with {len(cartography_weights)} weights")
        trainer = CartographyWeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            cartography_weights=cartography_weights,
            compute_metrics=compute_metrics,
        )
    else:
        logger.info("Using standard QuestionAnsweringTrainer (no cartography)")
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    
    # Train
    logger.info(f"Starting {model_type} training...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    logger.info(f"{model_type} training completed in {training_time:.1f}s")
    
    # Save model
    trainer.save_model()
    
    # Evaluate
    logger.info(f"Evaluating {model_type} model...")
    eval_results = trainer.evaluate()
    
    # Save results
    train_results_file = os.path.join(output_dir, "train_results.json")
    with open(train_results_file, "w") as f:
        json.dump({
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "train_loss": train_result.metrics.get("train_loss", 0),
            "epoch": train_result.metrics.get("epoch", 0),
        }, f, indent=2)
    
    eval_results_file = os.path.join(output_dir, "eval_results.json")
    with open(eval_results_file, "w") as f:
        json.dump({
            "eval_exact_match": eval_results.get("eval_exact_match", 0),
            "eval_f1": eval_results.get("eval_f1", 0),
            "eval_loss": eval_results.get("eval_loss", 0),
            "epoch": 8,
        }, f, indent=2)
    
    results = {
        "exact_match": eval_results.get("eval_exact_match", 0),
        "f1": eval_results.get("eval_f1", 0),
        "training_time": training_time,
        "train_loss": train_result.metrics.get("train_loss", 0),
    }
    
    logger.info(f"{model_type} Results:")
    logger.info(f"  Exact Match: {results['exact_match']:.2f}%")
    logger.info(f"  F1 Score: {results['f1']:.2f}%")
    logger.info(f"  Training Time: {training_time:.1f}s")
    
    return results


def show_epoch_results(results_dir, model_name):
    """Show epoch-by-epoch results for a model."""
    checkpoints = sorted(
        [d for d in os.listdir(results_dir) if d.startswith('checkpoint-')],
        key=lambda x: int(x.split('-')[1])
    )
    
    if not checkpoints:
        logger.warning(f"No checkpoints found in {results_dir}")
        return
    
    print(f'\n{"="*100}')
    print(f'EPOCH-BY-EPOCH RESULTS - {model_name}')
    print('='*100)
    print(f"{'Epoch':<8} {'Steps':<10} {'Exact Match %':<18} {'F1 Score %':<18}")
    print('-'*100)
    
    epoch = 1
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(results_dir, checkpoint)
        trainer_state_file = os.path.join(checkpoint_path, 'trainer_state.json')
        
        if os.path.exists(trainer_state_file):
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
            
            exact_match = None
            f1_score = None
            
            for entry in trainer_state.get('log_history', []):
                if entry.get('epoch') == float(epoch):
                    if 'eval_exact_match' in entry:
                        exact_match = entry['eval_exact_match']
                    if 'eval_f1' in entry:
                        f1_score = entry['eval_f1']
            
            step = checkpoint.split('-')[1]
            em_str = f'{exact_match:.2f}%' if exact_match is not None else 'N/A'
            f1_str = f'{f1_score:.2f}%' if f1_score is not None else 'N/A'
            
            print(f'{epoch:<8} {step:<10} {em_str:<18} {f1_str:<18}')
            epoch += 1
    
    print('='*100)


def compare_results(baseline_dir, cartography_dir):
    """Compare baseline and cartography model results."""
    
    # Load results
    baseline_train = json.load(open(Path(baseline_dir) / 'train_results.json'))
    baseline_eval = json.load(open(Path(baseline_dir) / 'eval_results.json'))
    
    cartography_train = json.load(open(Path(cartography_dir) / 'train_results.json'))
    cartography_eval = json.load(open(Path(cartography_dir) / 'eval_results.json'))
    
    print('\n' + '='*110)
    print('MODEL COMPARISON: BASELINE vs CARTOGRAPHY (upweight_hard)')
    print('='*110)
    
    print('\n{:<20} {:<20} {:<20}'.format('Metric', 'Baseline', 'Cartography'))
    print('-'*110)
    
    print('{:<20} {:<20.2f} {:<20.2f}'.format(
        'Exact Match (%)',
        baseline_eval['eval_exact_match'],
        cartography_eval['eval_exact_match']
    ))
    
    print('{:<20} {:<20.2f} {:<20.2f}'.format(
        'F1 Score (%)',
        baseline_eval['eval_f1'],
        cartography_eval['eval_f1']
    ))
    
    print('{:<20} {:<20.4f} {:<20.4f}'.format(
        'Train Loss',
        baseline_train['train_loss'],
        cartography_train['train_loss']
    ))
    
    print('{:<20} {:<20.2f} {:<20.2f}'.format(
        'Training Time (min)',
        baseline_train['train_runtime'] / 60,
        cartography_train['train_runtime'] / 60
    ))
    
    print('{:<20} {:<20.2f} {:<20.2f}'.format(
        'Samples/sec',
        baseline_train['train_samples_per_second'],
        cartography_train['train_samples_per_second']
    ))
    
    print('='*110)
    
    # Calculate differences
    em_diff = cartography_eval['eval_exact_match'] - baseline_eval['eval_exact_match']
    f1_diff = cartography_eval['eval_f1'] - baseline_eval['eval_f1']
    
    print('\nPERFORMANCE DELTA (Cartography - Baseline):')
    print('-'*110)
    print(f'  Exact Match: {em_diff:+.2f}% {"(IMPROVEMENT)" if em_diff > 0 else "(WORSE)" if em_diff < 0 else "(NO CHANGE)"}')
    print(f'  F1 Score:    {f1_diff:+.2f}% {"(IMPROVEMENT)" if f1_diff > 0 else "(WORSE)" if f1_diff < 0 else "(NO CHANGE)"}')
    print('='*110)
    
    return {
        "baseline": baseline_eval,
        "cartography": cartography_eval,
        "improvement": {"em_diff": em_diff, "f1_diff": f1_diff}
    }


def visualize_results(baseline_results, cartography_results, output_file='results/training_results_dashboard.png'):
    """Create comprehensive visualization dashboard."""
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Create a comprehensive visualization dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 NLP Final Project: Dataset Cartography Results Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Performance Comparison Bar Chart
    models = ['Baseline', 'Cartography']
    em_scores = [
        baseline_results['exact_match'] / 100 if baseline_results['exact_match'] > 1 else baseline_results['exact_match'],
        cartography_results['exact_match'] / 100 if cartography_results['exact_match'] > 1 else cartography_results['exact_match']
    ]
    f1_scores = [
        baseline_results['f1'] / 100 if baseline_results['f1'] > 1 else baseline_results['f1'],
        cartography_results['f1'] / 100 if cartography_results['f1'] > 1 else cartography_results['f1']
    ]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, em_scores, width, label='Exact Match', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1 Score', color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('📊 Performance Comparison: EM vs F1', fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training Time Comparison
    times = [
        baseline_results['training_time'],
        cartography_results['training_time']
    ]
    colors = ['#FFD93D', '#6BCF7F']
    
    wedges, texts, autotexts = ax2.pie(times, labels=models, autopct='%1.1f%%',
                                      colors=colors, startangle=90, explode=(0.05, 0.05))
    ax2.set_title('⏱️ Training Time Distribution', fontweight='bold', pad=20)
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # 3. Improvement Metrics
    em_diff = cartography_results['exact_match'] - baseline_results['exact_match']
    f1_diff = cartography_results['f1'] - baseline_results['f1']
    
    improvements = [em_diff, f1_diff]
    metrics = ['EM Improvement', 'F1 Improvement']
    colors_imp = ['#FF9FF3' if x >= 0 else '#FF6B6B' for x in improvements]
    
    bars3 = ax3.bar(metrics, improvements, color=colors_imp, alpha=0.8)
    ax3.set_ylabel('Improvement Score', fontweight='bold')
    ax3.set_title('📈 Cartography vs Baseline Improvement', fontweight='bold', pad=20)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.,
                height + (0.5 if height >= 0 else -1.5),
                f'{val:+.2f}%', ha='center',
                va='bottom' if height >= 0 else 'top',
                fontweight='bold')
    
    # 4. Performance Quality Assessment
    baseline_f1 = baseline_results['f1'] / 100 if baseline_results['f1'] > 1 else baseline_results['f1']
    cartography_f1 = cartography_results['f1'] / 100 if cartography_results['f1'] > 1 else cartography_results['f1']
    
    # Create quality categories
    quality_levels = ['Poor\n(0-20%)', 'Okay\n(20-50%)', 'Good\n(50-70%)', 'Excellent\n(70%+)']
    
    # Determine quality for each model
    def get_quality_level(f1_score):
        if f1_score <= 0.2:
            return 0
        elif f1_score <= 0.5:
            return 1
        elif f1_score <= 0.7:
            return 2
        else:
            return 3
    
    baseline_quality = get_quality_level(baseline_f1)
    cartography_quality = get_quality_level(cartography_f1)
    
    # Create stacked bar chart for quality assessment
    quality_colors = ['#FF4444', '#FFA500', '#90EE90', '#32CD32']
    
    # Show current performance levels
    ax4.barh(['Baseline'], [1], color=quality_colors[baseline_quality], alpha=0.8, height=0.4)
    ax4.barh(['Cartography'], [1], color=quality_colors[cartography_quality], alpha=0.8, height=0.4)
    
    ax4.set_xlim(0, 1)
    ax4.set_xlabel('Performance Quality', fontweight='bold')
    ax4.set_title('🎯 Model Quality Assessment', fontweight='bold', pad=20)
    
    # Add performance level labels
    ax4.text(0.5, 0, f'F1: {baseline_f1:.3f}\n{quality_levels[baseline_quality]}',
            ha='center', va='center', fontweight='bold', color='white')
    ax4.text(0.5, 1, f'F1: {cartography_f1:.3f}\n{quality_levels[cartography_quality]}',
            ha='center', va='center', fontweight='bold', color='white')
    
    # Add legend for quality levels
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8)
                      for color in quality_colors]
    ax4.legend(legend_elements, quality_levels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Visualization saved as '{output_file}'")
    plt.show()
    
    # Print detailed summary
    print("\n" + "="*80)
    print("📊 DETAILED RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n🎯 BASELINE MODEL:")
    print(f"   • Exact Match: {baseline_results['exact_match']:.2f}%")
    print(f"   • F1 Score: {baseline_results['f1']:.2f}%")
    print(f"   • Training Time: {baseline_results['training_time']:.1f} seconds")
    print(f"   • Quality Level: {quality_levels[baseline_quality]}")
    
    print(f"\n🗺️ CARTOGRAPHY MODEL:")
    print(f"   • Exact Match: {cartography_results['exact_match']:.2f}%")
    print(f"   • F1 Score: {cartography_results['f1']:.2f}%")
    print(f"   • Training Time: {cartography_results['training_time']:.1f} seconds")
    print(f"   • Quality Level: {quality_levels[cartography_quality]}")
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   • EM Change: {em_diff:+.2f}% ({'✅ Better' if em_diff > 0 else '❌ Worse' if em_diff < 0 else '➖ Same'})")
    print(f"   • F1 Change: {f1_diff:+.2f}% ({'✅ Better' if f1_diff > 0 else '❌ Worse' if f1_diff < 0 else '➖ Same'})")
    
    if em_diff > 0 or f1_diff > 0:
        print(f"   🎉 Dataset cartography shows positive impact on artifact mitigation!")
    elif em_diff == 0 and f1_diff == 0:
        print(f"   ➖ Dataset cartography shows no significant change.")
    else:
        print(f"   ⚠️ Dataset cartography may need tuning - negative impact observed.")
    
    print("\n" + "="*80)


def main():
    """Run complete training, evaluation, comparison, and visualization pipeline."""
    
    logger.info("="*80)
    logger.info("COMPLETE TRAINING & EVALUATION PIPELINE")
    logger.info("LOCAL TRAINING - MATCHING COLAB CONFIGURATION")
    logger.info("="*80)
    
    set_seed(42)
    
    model_name = "google/electra-small-discriminator"
    baseline_dir = "models/baseline_10k"
    cartography_dir = "models/cartography_10k"
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset - MATCH COLAB: 10K training, 1K validation
    logger.info("Loading SQuAD dataset (10K training, 1K validation - matching Colab)...")
    dataset = load_dataset("squad")
    
    train_dataset = dataset['train'].select(range(10000))  # Match Colab
    eval_dataset = dataset['validation'].select(range(1000))  # Match Colab
    
    logger.info(f"Dataset loaded - Training: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    # Preprocessing
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
    
    # STEP 1: Train baseline model
    logger.info("\n" + "="*80)
    logger.info("STEP 1/5: TRAINING BASELINE MODEL")
    logger.info("="*80)
    
    baseline_results = train_model(
        model_name=model_name,
        output_dir=baseline_dir,
        train_dataset=train_dataset_processed,
        eval_dataset=eval_dataset_processed,
        eval_examples=eval_dataset,
        tokenizer=tokenizer,
        cartography_weights=None,
        model_type="baseline"
    )
    
    # STEP 2: Train cartography model
    logger.info("\n" + "="*80)
    logger.info("STEP 2/5: TRAINING CARTOGRAPHY MODEL")
    logger.info("="*80)
    
    weights_file = "results/cartography/training_weights_upweight_hard.json"
    if os.path.exists(weights_file):
        logger.info(f"Loading cartography weights from {weights_file}")
        cartography_weights = load_cartography_weights(weights_file)
        
        cartography_results = train_model(
            model_name=model_name,
            output_dir=cartography_dir,
            train_dataset=train_dataset_processed,
            eval_dataset=eval_dataset_processed,
            eval_examples=eval_dataset,
            tokenizer=tokenizer,
            cartography_weights=cartography_weights,
            model_type="cartography"
        )
    else:
        logger.error(f"Cartography weights not found at {weights_file}")
        logger.error("Cannot train cartography model. Exiting.")
        return
    
    # STEP 3: Show epoch-by-epoch results
    logger.info("\n" + "="*80)
    logger.info("STEP 3/5: EPOCH-BY-EPOCH RESULTS")
    logger.info("="*80)
    
    show_epoch_results(baseline_dir, "BASELINE MODEL")
    show_epoch_results(cartography_dir, "CARTOGRAPHY MODEL")
    
    # STEP 4: Compare results
    logger.info("\n" + "="*80)
    logger.info("STEP 4/5: COMPARING MODELS")
    logger.info("="*80)
    
    comparison = compare_results(baseline_dir, cartography_dir)
    
    # Save combined results
    combined_results = {
        "baseline": baseline_results,
        "cartography": cartography_results,
        "improvement": comparison["improvement"]
    }
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    results_file = "results/local_training_results.json"
    with open(results_file, "w") as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"\n✅ Combined results saved to {results_file}")
    
    # STEP 5: Visualize results
    logger.info("\n" + "="*80)
    logger.info("STEP 5/5: VISUALIZING RESULTS")
    logger.info("="*80)
    
    visualize_results(baseline_results, cartography_results, 'results/training_results_dashboard.png')
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"✅ Baseline model trained and saved to: {baseline_dir}")
    logger.info(f"✅ Cartography model trained and saved to: {cartography_dir}")
    logger.info(f"✅ Results saved to: {results_file}")
    logger.info(f"✅ Visualization saved to: results/training_results_dashboard.png")
    logger.info("="*80)


if __name__ == "__main__":
    main()
