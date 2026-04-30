import matplotlib.pyplot as plt
import numpy as np

x = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
adversarial = [78.61, 84.67, 86.33, 86.22, 82.44, 82, 82.06, 81.89, 81.67, 81.39, 80.61]
random      = [77.07, 78.78, 80.28, 79.11,78.06,77.56,77.61,77.61,77.83,77.72,77.22]
popular = [
    77.81,
    81.35,
    82.61,
    82.09,
    79.69,
    79.04,
    79.15,
    79.09,
    79.11,
    78.94,
    78.35,
]
avg = (np.array(adversarial) + np.array(random) + np.array(popular)) / 3

# ==================== 正刊级设置（Adversarial 改为紫色） ====================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})

fig, ax = plt.subplots(figsize=(6.8, 5.5))

# 新配色：Adversarial 改为高级深紫（NeurIPS/ICLR 2024–2025 爆款色）
colors = {
    'random':      '#1f77b4',   # 经典蓝
    'adversarial':    '#9467bd',   # 优雅深紫 ← 改这里
    'popular':     '#2ca02c',   # 经典绿
    'average':     '#4c4c4c',   # 更深的灰（比 #7f7f7f 更稳重）
}

ax.plot(x, random,      color=colors['random'],      marker='o', markeredgecolor='white', markeredgewidth=1, label='Random')
ax.plot(x, adversarial, color=colors['adversarial'], marker='s', markeredgecolor='white', markeredgewidth=1, label='Adversarial')
ax.plot(x, popular,     color=colors['popular'],     marker='^', markeredgecolor='white', markeredgewidth=1, label='Popular')
ax.plot(x, avg,         color=colors['average'],     marker='D', linewidth=2.8, markeredgecolor='white', markeredgewidth=1.2, label='Average')

ax.set_xlabel("Visual Token Pruning Ratio (%)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim(-2, 52)
ax.set_ylim(72, 90)
ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)

ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)

plt.tight_layout(pad=0.3)
plt.savefig("/Users/yangmorunliu/Desktop/Yangmrl/ALD^2/token_pruning_ratio.pdf", dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.savefig("/Users/yangmorunliu/Desktop/Yangmrl/ALD^2/token_pruning_ratio.png", dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.show()
