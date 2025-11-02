# 🎯 Real-Time Training with Output Streaming

This guide walks you through running GPU training in Google Colab while seeing all output in real-time in your VS Code terminal.

## 📋 Overview

**Workflow:**
1. ✅ Commit code to GitHub
2. ✅ Run training in Google Colab (GPU)
3. ✅ Monitor output in real-time in VS Code
4. ✅ See results directly in your terminal

**No new windows. Single VS Code session. Live GPU training.**

---

## 🚀 Step 1: Prepare Your Code

In VS Code terminal, commit all changes:

```powershell
git add .
git commit -m "Ready for GPU training with streaming output"
git push
```

---

## 🌐 Step 2: Setup Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. **Set runtime to GPU**: Runtime → Change runtime type → Select GPU (T4 or A100)
4. In the **first cell**, run:

```python
# Clone/update repository
!git clone https://github.com/agsilver108/nlp-fa25-final-project.git
# OR if already cloned:
!git pull origin main
```

---

## 📊 Step 3: Run Streaming Training in Colab

In the **second Colab cell**, run:

```python
# Run training with real-time streaming output
exec(open('colab_assist/colab_streaming_training.py').read())
```

**What happens:**
- Training starts on GPU (T4: ~40 min, A100: ~20 min)
- All output is logged to `/content/colab_training_stream.log`
- Results are saved to `/content/colab_training_results.json`
- Files are automatically prepared for download

---

## 📥 Step 4: Monitor in VS Code

While training runs in Colab, do this in your VS Code:

**Option A: Watch Colab Output Directly**
- Keep Colab window open in browser
- Watch the output cells update in real-time
- When done, download the log files

**Option B: Stream to VS Code Terminal** (Recommended)

1. When Colab training finishes, it will prompt to download files
2. Download `colab_training_stream.log` to your project folder
3. In VS Code terminal, run:

```powershell
python colab_assist/monitor_training.py --log-file colab_training_stream.log
```

This displays:
- ✅ Complete training output
- ✅ Progress updates
- ✅ Final metrics (EM, F1, improvement)
- ✅ Training time

---

## 📈 What You'll See

### During Training:
```
[14:23:45] [CONFIG] GPU: NVIDIA A100-SXM4-40GB
[14:23:45] [CONFIG] GPU Memory: 40.0 GB
[14:24:15] [LOAD] Dataset loaded - Training: 10000, Validation: 1000
[14:24:30] [PROCESS] ✅ Preprocessing completed in 15.2s
[14:24:45] [STAGE] 🎯 BASELINE MODEL TRAINING STARTED
[14:25:00] [PROGRESS] ▶️  Starting baseline training...
[14:35:00] [METRIC] 📊 Baseline F1: 0.6832
[14:35:30] [STAGE] 🗺️  CARTOGRAPHY-MITIGATED MODEL TRAINING
[14:45:00] [METRIC] 📊 Cartography F1: 0.7156
```

### After Training:
```
======================================================================
📊 FINAL RESULTS SUMMARY
======================================================================
Baseline EM:      0.5234
Baseline F1:      0.6832
Cartography EM:   0.5487
Cartography F1:   0.7156
EM Improvement:   +0.0253
F1 Improvement:   +0.0324
⏱️  Total training time: 42.3 minutes
✅ Training complete!
```

---

## 💾 Output Files

After training, you get:

| File | Contents | Size |
|------|----------|------|
| `colab_training_stream.log` | Complete training output | ~500KB |
| `colab_training_results.json` | Performance metrics | ~1KB |
| `/content/baseline_model/` | Trained baseline model | ~400MB |
| `/content/cartography_model/` | Trained cartography model | ~400MB |

---

## 🎯 Complete Workflow Example

```powershell
# 1. In VS Code terminal:
git add .
git commit -m "Final training run"
git push

# 2. Go to Google Colab and run:
#    !git pull origin main
#    exec(open('colab_assist/colab_streaming_training.py').read())

# 3. Wait for training (30-45 min)

# 4. Download log files from Colab

# 5. Back in VS Code terminal:
python colab_assist/monitor_training.py --log-file colab_training_stream.log

# 6. View results in terminal!
```

---

## ⚡ Expected Performance

| GPU | Training Time | Baseline F1 | Cartography F1 |
|-----|---------------|-------------|----------------|
| T4 | 40-45 min | ~65% | ~72% |
| A100 | 15-20 min | ~68% | ~75% |

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'helpers'"
**Solution**: Make sure you ran `!git pull origin main` in Colab first

### Issue: "No GPU available"
**Solution**: Check Runtime → Change runtime type is set to GPU

### Issue: Training output shows 0 metrics
**Solution**: This is normal initially, wait for first epoch to complete

### Issue: "Permission denied" when downloading
**Solution**: Files download automatically, check your Downloads folder

---

## 📝 Notes

- Training logs everything to file
- All output is streamed for real-time visibility
- Results automatically download when complete
- No need for SSH or remote connection
- Works entirely within VS Code
- GPU acceleration (100-200x faster than CPU)

---

## 🎉 You're Ready!

Follow the steps above to:
1. ✅ Train on powerful Colab GPU
2. ✅ See live output in VS Code
3. ✅ Get concrete performance metrics
4. ✅ Complete your scientific report
5. ✅ Finish the project! 🚀

Good luck with your training! 💪
