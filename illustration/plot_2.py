import matplotlib.pyplot as plt
import numpy as np

# 全局字体设为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 数据（只保留原来的四个模型）
models = ["LLaVA-1.5-7B", "InstructBlip-Vicuna-7B", "InternVL-3.5-8B", "LLaVA-Next-8B"]

random = [86.67, 88.33, 83.33, 27.00]
popular = [85.83, 82.83, 85.00, 30.16]
adversarial = [84.67, 81.00, 84.50, 27.83]

# 颜色（保持四个）
colors = ["#6699CC", "#88BB99", "#EE9999"]

bar_width = 0.24  # 柱子再细一点（原 0.27 → 0.24）
x = np.arange(len(models))

plt.figure(figsize=(9, 6.5))

# 画柱子（位置微调，让组更匀称）
bars1 = plt.bar(
    x - bar_width * 1.1,
    random,
    width=bar_width,
    label="Random",
    color=colors[0],
    alpha=0.95,
    edgecolor="white",
    linewidth=1.2,
)
bars2 = plt.bar(
    x,
    popular,
    width=bar_width,
    label="Popular",
    color=colors[1],
    alpha=0.95,
    edgecolor="white",
    linewidth=1.2,
)
bars3 = plt.bar(
    x + bar_width * 1.1,
    adversarial,
    width=bar_width,
    label="Adversarial",
    color=colors[2],
    alpha=0.95,
    edgecolor="white",
    linewidth=1.2,
)

# 柱子上数值（高度稍降低，避免太高）
for bars, values in zip([bars1, bars2, bars3], [random, popular, adversarial]):
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.6,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            family="Times New Roman",
        )

# 坐标轴标签
plt.ylabel("Accuracy (%)", fontsize=20, family="Times New Roman", labelpad=10)


# X轴标签
plt.xticks(x, models, fontsize=18, family="Times New Roman", rotation=10, ha="right")

# Y轴
plt.ylim(0, 100)
plt.yticks(np.arange(0, 101, 10), fontsize=17, family="Times New Roman")
plt.grid(axis="y", linestyle="--", alpha=0.6, linewidth=0.8)

# 图例：右上角，直角边框
legend = plt.legend(
    prop={"family": "Times New Roman", "size": 17},
    loc="upper right",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    framealpha=1,
)
legend.get_frame().set_linewidth(0.8)

# 去掉顶部和右边框
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.tight_layout()

# 保存为矢量 PDF
plt.savefig(
    "/Users/yangmorunliu/Desktop/Yangmrl/多模态幻觉:推理/ALD^2/illustration/pre_acc.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)

plt.show()
