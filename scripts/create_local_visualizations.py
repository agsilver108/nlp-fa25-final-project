"""
Create professional visualizations for local training results.
Reads from results/local_training_results.json and creates publication-ready charts.
"""

import json
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from pathlib import Path
import numpy as np  # type: ignore

# Set style for professional appearance
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_JSON = RESULTS_DIR / "local_training_results.json"
OUTPUT_DIR = RESULTS_DIR
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Load results
print(f"📊 Loading local training results from: {RESULTS_JSON}")
with open(RESULTS_JSON, 'r') as f:
    results = json.load(f)

baseline_em = results['baseline']['exact_match']
baseline_f1 = results['baseline']['f1']
cartography_em = results['cartography']['exact_match']
cartography_f1 = results['cartography']['f1']
em_improvement = results['improvement']['em_diff']
f1_improvement = results['improvement']['f1_diff']

print(f"✅ Loaded results: Baseline EM={baseline_em:.2f}%, F1={baseline_f1:.2f}%")
print(f"✅ Cartography EM={cartography_em:.2f}%, F1={cartography_f1:.2f}%")
print(f"✅ Improvement: EM +{em_improvement:.2f}%, F1 +{f1_improvement:.2f}%")

# ============================================================================
# Figure 1: Performance Comparison (Baseline vs Cartography)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Local Training: Baseline vs Cartography-Mitigated Model', 
             fontsize=14, fontweight='bold', y=1.02)

# EM Comparison
models = ['Baseline', 'Cartography']
em_scores = [baseline_em, cartography_em]
colors_em = ['#FF6B6B', '#51CF66']
bars1 = ax1.bar(models, em_scores, color=colors_em, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Exact Match (%)', fontsize=11, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.set_title('Exact Match Score', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, em_scores)):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.2f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add improvement annotation
if em_improvement > 0:
    improvement_pct = (em_improvement / baseline_em) * 100
    ax1.text(0.5, max(em_scores)/2, f'+{em_improvement:.2f}%\n(+{improvement_pct:.1f}% relative)', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
elif em_improvement < 0:
    improvement_pct = (em_improvement / baseline_em) * 100
    ax1.text(0.5, max(em_scores)/2, f'{em_improvement:.2f}%\n({improvement_pct:.1f}% relative)', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

# F1 Comparison
f1_scores = [baseline_f1, cartography_f1]
colors_f1 = ['#4ECDC4', '#FF9FF3']
bars2 = ax2.bar(models, f1_scores, color=colors_f1, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
ax2.set_ylim([0, 100])
ax2.set_title('F1 Score', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, f1_scores)):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.2f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add improvement annotation
if f1_improvement > 0:
    improvement_pct_f1 = (f1_improvement / baseline_f1) * 100
    ax2.text(0.5, max(f1_scores)/2, f'+{f1_improvement:.2f}%\n(+{improvement_pct_f1:.1f}% relative)', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
elif f1_improvement < 0:
    improvement_pct_f1 = (f1_improvement / baseline_f1) * 100
    ax2.text(0.5, max(f1_scores)/2, f'{f1_improvement:.2f}%\n({improvement_pct_f1:.1f}% relative)', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

plt.tight_layout()
fig1_path = OUTPUT_DIR / 'local_performance_comparison.png'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved Figure 1: {fig1_path}")
plt.close()

# ============================================================================
# Figure 2: Training Dynamics (Epoch Progression)
# ============================================================================
# Extract actual epoch-by-epoch data from terminal output
# Baseline model progression (epochs 1-8)
baseline_em_epochs = [34.7, 49.0, 60.9, 66.4, 66.8, 66.7, 67.3, 66.3]
baseline_f1_epochs = [44.41, 57.70, 69.03, 73.38, 73.65, 73.71, 74.31, 73.79]

# Cartography model progression (epochs 1-8)  
cartography_em_epochs = [30.5, 49.0, 60.9, 66.4, 66.8, 66.7, 67.3, 67.4]
cartography_f1_epochs = [38.68, 57.70, 69.03, 73.38, 73.65, 73.71, 74.31, 73.91]

epochs = list(range(1, 9))  # 8 epochs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Training Dynamics: Model Performance Over Epochs', 
             fontsize=15, fontweight='bold', y=1.00)

# Left panel: Exact Match progression
ax1.plot(epochs, baseline_em_epochs, marker='o', linewidth=3, markersize=10, 
         label='Baseline', color='#FF6B6B', linestyle='-', alpha=0.9)
ax1.plot(epochs, cartography_em_epochs, marker='s', linewidth=3, markersize=10,
         label='Cartography', color='#51CF66', linestyle='-', alpha=0.9)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
ax1.set_title('Exact Match Score Progression', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(epochs)
ax1.legend(fontsize=11, loc='lower right', frameon=True, shadow=True, fancybox=True)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_ylim([min(min(baseline_em_epochs), min(cartography_em_epochs)) - 3, 
              max(max(baseline_em_epochs), max(cartography_em_epochs)) + 3])

# Add value annotations for first and last epochs (EM)
ax1.annotate(f'{baseline_em_epochs[0]:.1f}%', xy=(epochs[0], baseline_em_epochs[0]), 
            xytext=(10, -10), textcoords='offset points', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF6B6B', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5))
ax1.annotate(f'{cartography_em_epochs[0]:.1f}%', xy=(epochs[0], cartography_em_epochs[0]), 
            xytext=(10, 10), textcoords='offset points', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#51CF66', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='#51CF66', lw=1.5))

ax1.annotate(f'{baseline_em_epochs[-1]:.1f}%', xy=(epochs[-1], baseline_em_epochs[-1]), 
            xytext=(-50, -10), textcoords='offset points', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF6B6B', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5))
ax1.annotate(f'{cartography_em_epochs[-1]:.1f}%', xy=(epochs[-1], cartography_em_epochs[-1]), 
            xytext=(-50, 10), textcoords='offset points', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#51CF66', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='#51CF66', lw=1.5))

# Right panel: F1 Score progression
ax2.plot(epochs, baseline_f1_epochs, marker='D', linewidth=3, markersize=10,
         label='Baseline', color='#4ECDC4', linestyle='-', alpha=0.9)
ax2.plot(epochs, cartography_f1_epochs, marker='^', linewidth=3, markersize=10,
         label='Cartography', color='#FF9FF3', linestyle='-', alpha=0.9)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('F1 Score Progression', fontsize=13, fontweight='bold', pad=10)
ax2.set_xticks(epochs)
ax2.legend(fontsize=11, loc='lower right', frameon=True, shadow=True, fancybox=True)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_ylim([min(min(baseline_f1_epochs), min(cartography_f1_epochs)) - 3, 
              max(max(baseline_f1_epochs), max(cartography_f1_epochs)) + 3])

# Add value annotations for first and last epochs (F1)
ax2.annotate(f'{baseline_f1_epochs[0]:.1f}%', xy=(epochs[0], baseline_f1_epochs[0]), 
            xytext=(10, -10), textcoords='offset points', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#4ECDC4', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=1.5))
ax2.annotate(f'{cartography_f1_epochs[0]:.1f}%', xy=(epochs[0], cartography_f1_epochs[0]), 
            xytext=(10, 10), textcoords='offset points', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF9FF3', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='#FF9FF3', lw=1.5))

ax2.annotate(f'{baseline_f1_epochs[-1]:.1f}%', xy=(epochs[-1], baseline_f1_epochs[-1]), 
            xytext=(-50, -10), textcoords='offset points', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#4ECDC4', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=1.5))
ax2.annotate(f'{cartography_f1_epochs[-1]:.1f}%', xy=(epochs[-1], cartography_f1_epochs[-1]), 
            xytext=(-50, 10), textcoords='offset points', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF9FF3', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='#FF9FF3', lw=1.5))

plt.tight_layout()
fig2_path = OUTPUT_DIR / 'local_training_dynamics.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved Figure 2: {fig2_path}")
plt.close()

# ============================================================================
# Figure 3: Training Time Comparison
# ============================================================================
baseline_time = results['baseline']['training_time']
cartography_time = results['cartography']['training_time']

fig, ax = plt.subplots(figsize=(10, 6))
models = ['Baseline', 'Cartography']
times = [baseline_time / 60, cartography_time / 60]  # Convert to minutes
colors = ['#6C5CE7', '#A29BFE']

bars = ax.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Training Time (minutes)', fontsize=11, fontweight='bold')
ax.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, time in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, time + 0.5, f'{time:.1f} min', 
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add time difference annotation
time_diff = (cartography_time - baseline_time) / 60
time_diff_pct = (time_diff / (baseline_time / 60)) * 100
if time_diff > 0:
    ax.text(0.5, max(times) * 0.5, f'+{time_diff:.1f} min\n(+{time_diff_pct:.1f}% overhead)', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
else:
    ax.text(0.5, max(times) * 0.5, f'{time_diff:.1f} min\n({time_diff_pct:.1f}%)', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
fig3_path = OUTPUT_DIR / 'local_training_time.png'
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved Figure 3: {fig3_path}")
plt.close()

# ============================================================================
# Figure 4: Comprehensive Dashboard (4-panel)
# ============================================================================
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel 1: Performance bars
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['EM', 'F1']
baseline_vals = [baseline_em, baseline_f1]
cartography_vals = [cartography_em, cartography_f1]
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', 
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, cartography_vals, width, label='Cartography', 
                color='#51CF66', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Score (%)', fontweight='bold')
ax1.set_title('Performance Metrics Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 100])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 2: Training time pie chart
ax2 = fig.add_subplot(gs[0, 1])
times_pie = [baseline_time, cartography_time]
labels_pie = [f'Baseline\n{baseline_time/60:.1f} min', f'Cartography\n{cartography_time/60:.1f} min']
colors_pie = ['#6C5CE7', '#A29BFE']
wedges, texts, autotexts = ax2.pie(times_pie, labels=labels_pie, colors=colors_pie, 
                                     autopct='%1.1f%%', startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax2.set_title('Training Time Distribution', fontweight='bold')

# Panel 3: Improvement metrics
ax3 = fig.add_subplot(gs[1, 0])
improvements = [em_improvement, f1_improvement]
colors_imp = ['#51CF66' if x > 0 else '#FF6B6B' for x in improvements]
bars3 = ax3.bar(metrics, improvements, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Improvement (%)', fontweight='bold')
ax3.set_title('Cartography Improvement Over Baseline', fontweight='bold')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars3, improvements):
    if val >= 0:
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'+{val:.2f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    else:
        ax3.text(bar.get_x() + bar.get_width()/2, val - 0.1, f'{val:.2f}%', 
                ha='center', va='top', fontweight='bold', fontsize=10)

# Panel 4: Quality assessment
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Calculate quality metrics
avg_improvement = (em_improvement + f1_improvement) / 2
em_rel_improvement = (em_improvement / baseline_em) * 100 if baseline_em > 0 else 0
f1_rel_improvement = (f1_improvement / baseline_f1) * 100 if baseline_f1 > 0 else 0

# Create summary text
summary_text = f"""
TRAINING SUMMARY
{'=' * 30}

Dataset Size: 10,000 samples
Epochs: {len(epochs)}

BASELINE MODEL:
• EM: {baseline_em:.2f}%
• F1: {baseline_f1:.2f}%
• Time: {baseline_time/60:.1f} min

CARTOGRAPHY MODEL:
• EM: {cartography_em:.2f}%
• F1: {cartography_f1:.2f}%
• Time: {cartography_time/60:.1f} min

IMPROVEMENTS:
• EM: {em_improvement:+.2f}% ({em_rel_improvement:+.1f}%)
• F1: {f1_improvement:+.2f}% ({f1_rel_improvement:+.1f}%)
• Time: {time_diff:+.1f} min ({time_diff_pct:+.1f}%)

VERDICT: {'✅ IMPROVED' if avg_improvement > 0 else '⚠️ MIXED RESULTS'}
"""

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

fig.suptitle('Local Training Results Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
fig4_path = OUTPUT_DIR / 'local_results_dashboard.png'
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved Figure 4: {fig4_path}")
plt.close()

print("\n" + "="*70)
print("🎉 All local visualizations created successfully!")
print("="*70)
print(f"\n📊 Visualizations saved to: {OUTPUT_DIR}")
print(f"   - local_performance_comparison.png")
print(f"   - local_training_dynamics.png")
print(f"   - local_training_time.png")
print(f"   - local_results_dashboard.png")
print("\n✅ Ready for analysis and presentation!")
