# Colab Assistance Files

This folder contains all Google Colab Pro setup and training scripts for the NLP Final Project. These files are optimized for GPU-accelerated training in the cloud.

## Contents

### Training Scripts
- **colab_training.py** - Main training script with baseline and cartography-mitigated model training
- **colab_training_fixed.py** - Enhanced version with complete error handling and debugging output
- **colab_setup.py** - Environment setup script for Colab (mounts drive, installs packages, checks GPU)

### Remote Development Setup
- **COLAB_REMOTE_SETUP.md** - Comprehensive guide for Method 1, 2, and 3 remote setups
- **colab_ssh_setup.py** - Full SSH tunnel setup with ngrok for VS Code integration
- **colab_ssh_simple.py** - Simplified SSH setup cell ready to copy-paste into Colab

## Quick Start

### Option 1: Direct Upload + Manual Sync (Simplest)
1. Upload `NLP_Final_Project_Colab.ipynb` to Colab
2. Change runtime to GPU (T4/A100)
3. Run cells for training

### Option 2: GitHub Sync (Recommended)
1. Edit files locally in VS Code
2. Commit and push to GitHub
3. In Colab, run: `!git pull origin main`
4. Execute training scripts

### Option 3: Remote SSH (Advanced)
1. Run `colab_ssh_simple.py` in a Colab cell with your ngrok token
2. Connect via VS Code Remote-SSH extension
3. Edit files directly in Colab environment

## Usage

### Training in Colab with GitHub Sync:
```bash
# In VS Code terminal:
git add .
git commit -m "Update configuration"
git push

# In Colab:
!git pull origin main
exec(open('colab_assist/colab_training_fixed.py').read())
```

### Expected Training Time:
- **GPU (T4)**: ~30-45 minutes for full training
- **GPU (A100)**: ~15-20 minutes for full training
- **CPU**: 10+ hours (not recommended)

## Output Files

Training produces:
- **colab_training_results.json** - Performance metrics (EM, F1, training time)
- **baseline_model/** - Trained baseline model checkpoint
- **cartography_model/** - Cartography-mitigated model checkpoint
- Visualization plots and analysis results

## Troubleshooting

### Issue: Can't find cartography weights
**Solution**: Run artifact analysis scripts first to generate weights:
```bash
python analysis_scripts/dataset_cartography.py
```

### Issue: Out of memory errors
**Solution**: Reduce batch size or training samples in the training scripts:
```python
train_dataset = dataset['train'].select(range(5000))  # Reduce from 10K
per_device_train_batch_size = 8  # Reduce batch size
```

### Issue: Evaluation metrics show 0
**Solution**: Ensure `compute_metrics` function is included and evaluation is enabled

## SSH Connection Troubleshooting

### ngrok tunnel disconnects:
- Keep the Colab cell running (don't stop it)
- Reconnect with same credentials
- Check ngrok authtoken validity

### SSH connection refused:
- Verify host and port from ngrok output
- Check credentials (default: root / colab123)
- Ensure openssh-server is running in Colab

## Project Structure

```
nlp-final-project/
├── colab_assist/              # This folder
│   ├── colab_setup.py
│   ├── colab_training.py
│   ├── colab_training_fixed.py
│   ├── colab_ssh_setup.py
│   ├── colab_ssh_simple.py
│   ├── COLAB_REMOTE_SETUP.md
│   └── README.md
├── colab_notebook/            # Jupyter notebooks
│   └── NLP_Final_Project_Colab.ipynb
├── analysis_scripts/          # Artifact analysis
├── models/                    # Model checkpoints
├── results/                   # Analysis results
├── helpers.py
├── run.py
└── requirements.txt
```

## Performance Tips

1. **Use mixed precision (fp16)**: Enabled by default, ~2x faster
2. **Increase batch size on A100**: Change `per_device_train_batch_size = 32`
3. **Monitor GPU utilization**: Run `!nvidia-smi` in Colab
4. **Download results regularly**: Models can be large, save locally
5. **Use Google Drive**: Mount and save to `/content/drive/My Drive/`

## Recommended Settings for GPU

**T4 GPU:**
- batch_size: 16
- epochs: 3
- Expected F1: 60-70%

**A100 GPU:**
- batch_size: 32
- epochs: 3-4
- Expected F1: 65-75%

---

*Last Updated*: November 2, 2025
*Project*: NLP Final Project - Dataset Cartography for Artifact Mitigation
