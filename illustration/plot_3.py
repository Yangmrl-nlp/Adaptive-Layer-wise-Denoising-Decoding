import matplotlib.pyplot as plt
import numpy as np

# ================== 数据 ==================
methods = ['Greedy', 'Dola', 'VCD', 'ICD', 'ALD$^2$(Ours)']
values_ms = [123.80, 124.73, 258.30, 166.90, 238.98]   # 单位：ms
labels = ['Greedy', 'Dola', 'VCD', 'ICD', 'ALD$^2$']  # 实际显示的标签（可改）

# ================== 绿色渐变配色 ==================
# 从浅绿到深绿（数值越大越深）
cmap = plt.get_cmap('Greens')
norm = plt.Normalize(min(values_ms), max(values_ms))
colors = cmap(norm(values_ms))

# ================== 绘图 ==================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(9, 7))  # 适合论文单栏/双栏

# 横向柱状图
bars = ax.barh(labels, values_ms, color=colors, height=0.65, edgecolor='black', linewidth=0.8)

# 在条形末端显示数值（白色粗体更清晰）
for i, (bar, val) in enumerate(zip(bars, values_ms)):
    ax.text(val + 4, bar.get_y() + bar.get_height()/2,
            f'{val}', va='center', ha='left',
            fontsize=25, fontweight='bold', color='black', family='Times New Roman')

# 美化
ax.set_xlabel('Token Latency (ms/token)', fontsize=25, family='Times New Roman', labelpad=8)
ax.tick_params(axis='x', labelsize=25, width=1.5, length=6)
ax.set_xlim(0, max(values_ms) * 1.15)


# 去掉多余边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticklabels(labels, fontsize=22, family='Times New Roman')

# Y轴标签去掉背景，X轴网格
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', linestyle='--', alpha=0.7, linewidth=0.7)
ax.set_axisbelow(True)

# 可选：在图外加个小标题（论文中常用）
# plt.title('5. latency(done)', fontsize=14, family='Times New Roman', pad=20, loc='left')

plt.tight_layout()

# ================== 保存为矢量 PDF ==================
plt.savefig('/Users/yangmorunliu/Desktop/Yangmrl/ALW/latency.pdf', format='pdf', dpi=300, bbox_inches='tight', transparent=True)


plt.show()