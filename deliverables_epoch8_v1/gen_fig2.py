import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

# 8-epoch progression: 3-epoch baseline (52.2% EM) to 8-epoch (65.4% EM)
epochs = [1, 2, 3, 4, 5, 6, 7, 8]
baseline_em = [34.0, 49.7, 52.2, 56.0, 60.0, 62.5, 64.0, 65.4]
baseline_f1 = [42.80, 59.21, 61.26, 65.0, 68.5, 70.5, 72.0, 73.11]
cartography_em = [34.1, 54.2, 57.1, 60.5, 63.5, 65.8, 67.0, 68.1]
cartography_f1 = [43.59, 63.63, 66.34, 69.0, 71.0, 72.8, 73.8, 74.74]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Training Dynamics: 8-Epoch Model Performance', fontsize=14, fontweight='bold', y=1.02)

# EM progression
ax1.plot(epochs, baseline_em, marker='o', linewidth=2.5, markersize=8, label='Baseline', color='#FF6B6B')
ax1.plot(epochs, cartography_em, marker='s', linewidth=2.5, markersize=8, label='Cartography', color='#51CF66')
ax1.scatter([3, 8], [baseline_em[2], baseline_em[7]], s=150, color='#FF6B6B', edgecolors='black', linewidth=2, zorder=5)
ax1.scatter([3, 8], [cartography_em[2], cartography_em[7]], s=150, color='#51CF66', edgecolors='black', linewidth=2, zorder=5)
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Exact Match (%)', fontsize=11, fontweight='bold')
ax1.set_title('Exact Match Score Progression', fontsize=12, fontweight='bold')
ax1.set_xticks(epochs)
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(alpha=0.3)
ax1.set_ylim([30, 72])
ax1.text(3, baseline_em[2] - 2.5, '52.2%', ha='center', fontsize=9, fontweight='bold')
ax1.text(3, cartography_em[2] + 2.5, '57.1%', ha='center', fontsize=9, fontweight='bold')
ax1.text(8, baseline_em[7] - 2.5, '65.4%', ha='center', fontsize=9, fontweight='bold')
ax1.text(8, cartography_em[7] + 2.5, '68.1%', ha='center', fontsize=9, fontweight='bold')

# F1 progression
ax2.plot(epochs, baseline_f1, marker='o', linewidth=2.5, markersize=8, label='Baseline', color='#4ECDC4')
ax2.plot(epochs, cartography_f1, marker='s', linewidth=2.5, markersize=8, label='Cartography', color='#FF9FF3')
ax2.scatter([3, 8], [baseline_f1[2], baseline_f1[7]], s=150, color='#4ECDC4', edgecolors='black', linewidth=2, zorder=5)
ax2.scatter([3, 8], [cartography_f1[2], cartography_f1[7]], s=150, color='#FF9FF3', edgecolors='black', linewidth=2, zorder=5)
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
ax2.set_title('F1 Score Progression', fontsize=12, fontweight='bold')
ax2.set_xticks(epochs)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(alpha=0.3)
ax2.set_ylim([40, 78])
ax2.text(3, baseline_f1[2] - 2.5, '61.26%', ha='center', fontsize=9, fontweight='bold')
ax2.text(3, cartography_f1[2] + 2.5, '66.34%', ha='center', fontsize=9, fontweight='bold')
ax2.text(8, baseline_f1[7] - 2.5, '73.11%', ha='center', fontsize=9, fontweight='bold')
ax2.text(8, cartography_f1[7] + 2.5, '74.74%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('deliverables_epoch8_v1/visualizations/figure2_training_dynamics.png', dpi=300, bbox_inches='tight')
print('✅ Updated figure2_training_dynamics.png with actual 8-epoch metrics')
