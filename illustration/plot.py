import matplotlib.pyplot as plt
import numpy as np

# ================== LaTeX 论文风格 ==================
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "text.latex.preamble": r"\usepackage{times}",
    }
)

# ================== 数据 ==================
methods = ["Greedy", "DoLa", "ICD", "VCD", "ALD$^2$(Ours)"]
llava_15 = np.array([1432.65, 1419.15, 1594.98, 1687.03, 1647.5])
instructblip = np.array([1598.94, 1486.86, 1534.36, 1598.19, 1610.19])
llavanext = np.array([1769.88, 1432.89, 1545.47, 1650.90, 1858.44])
internvl = np.array([2342.72, 2182.60, 2239.32, 2288.41, 2347.75])

# 平均值（如果想算四个模型的平均）
mean_scores = (llava_15 + instructblip + llavanext + internvl) / 4

fig, ax = plt.subplots(figsize=(11, 8))  # 宽度稍微加大一点
bar_width = 0.2  # 四个柱子稍微窄一点
x = np.arange(len(methods))

# ================== 柱状图 ==================
ax.bar(
    x - 1.5 * bar_width,
    llava_15,
    width=bar_width,
    color="#8ebcb7",
    label="LLaVA-1.5-7B",
)
ax.bar(
    x - 0.5 * bar_width,
    instructblip,
    width=bar_width,
    color="#708ab9",
    label="InstructBLIP-Vicuna-7B",
)
ax.bar(
    x + 0.5 * bar_width,
    llavanext,
    width=bar_width,
    color="#efdfbb",
    label="LLaVA-Next-8B",
)
ax.bar(
    x + 1.5 * bar_width, internvl, width=bar_width, color="#eba985", label="InternVL-3.5-8B"
)

# ================== 平均值折线 ==================
line_color = "#999A9D"
font_color = "#141415"
ax.plot(
    x,
    mean_scores,
    color=line_color,
    marker="o",
    markersize=6,
    linewidth=2,
    label="Average MME score",
)

# 标注平均值
for i, val in enumerate(mean_scores):
    if i == 4:
        ax.text(
            i,
            val + 20,
            r"\textbf{" + f"{val:.1f}" + "}",
            ha="center",
            va="bottom",
            fontsize=20,
            color=font_color,
        )
    else:
        ax.text(
            x[i],
            val + 15,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=20,
            color=font_color,
        )

# ================== 样式设置 ==================
ax.set_ylabel("MME score")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(1300, 2650)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.legend(loc="upper left", frameon=False)

plt.tight_layout(pad=0.5)
plt.savefig(
    "/Users/yangmorunliu/Desktop/Yangmrl/多模态幻觉:推理/ALD^2/illustration/mme_results.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "/Users/yangmorunliu/Desktop/Yangmrl/多模态幻觉:推理/ALD^2/illustration/mme_results.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
