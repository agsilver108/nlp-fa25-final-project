#!/usr/bin/env python3
"""
Google Colab Setup Script for NLP Final Project
Handles environment setup, dataset loading, and GPU configuration for fast training.
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_colab_environment():
    """Setup the Colab environment with required packages and configurations."""
    
    print("🚀 Setting up NLP Final Project in Google Colab...")
    
    # Check if we're in Colab
    try:
        import google.colab  # type: ignore
        IN_COLAB = True
        print("✅ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("❌ Not running in Colab - this script is optimized for Colab")
        return False
    
    # Mount Google Drive (optional, for saving results)
    print("\n📁 Mounting Google Drive...")
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
    
    # Install additional packages if needed
    print("\n📦 Installing packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "transformers", "torch", "evaluate", "matplotlib", "seaborn", "scipy"], check=True)
    
    # Clone the repository if not already present
    if not os.path.exists('/content/nlp-final-project'):
        print("\n📥 Cloning repository...")
        subprocess.run(["git", "clone", "https://github.com/agsilver108/nlp-fa25-final-project.git", "/content/nlp-final-project"], check=True)
        os.chdir('/content/nlp-final-project')
    else:
        print("\n📂 Repository already exists, updating...")
        os.chdir('/content/nlp-final-project')
        subprocess.run(["git", "pull"], check=True)
    
    # Check GPU availability
    print("\n🔧 Checking GPU availability...")
    import torch  # type: ignore
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU available - training will be slow")
    
    print("\n✅ Setup complete! Ready for fast training.")
    return True

def create_colab_training_config():
    """Create optimized training configuration for Colab GPU."""
    
    config = {
        "model_name": "google/electra-small-discriminator",
        "output_dir": "/content/nlp-final-project/models/colab_baseline",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,  # Larger batch size for GPU
        "per_device_eval_batch_size": 64,
        "learning_rate": 3e-5,
        "warmup_steps": 500,
        "logging_steps": 100,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "fp16": True,  # Enable mixed precision for speed
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 2,
        "save_total_limit": 2,
        "report_to": [],  # Disable wandb logging
    }
    
    return config

if __name__ == "__main__":
    setup_colab_environment()