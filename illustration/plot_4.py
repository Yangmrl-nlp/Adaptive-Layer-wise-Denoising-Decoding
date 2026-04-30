import matplotlib.pyplot as plt
import numpy as np

# 全局字体设为 Comic Sans MS（系统没有就自动回退）
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

# 数据（32层）
raw_data = [
    [0.0300, 0.2340, 0.2549, 0.2452, 0.2358],
    [0.0344, 0.2313, 0.2769, 0.2368, 0.2207],
    [0.0371, 0.2346, 0.2786, 0.2274, 0.2222],
    [0.0423, 0.2284, 0.2776, 0.2302, 0.2214],
    [0.0498, 0.2305, 0.2781, 0.2415, 0.2002],
    [0.0530, 0.2286, 0.2756, 0.2377, 0.2050],
    [0.0728, 0.2482, 0.2925, 0.2173, 0.1693],
    [0.0975, 0.2607, 0.2507, 0.2375, 0.1533],
    [0.1079, 0.2671, 0.2163, 0.2267, 0.1821],
    [0.2063, 0.3025, 0.1698, 0.2000, 0.1213],
    [0.2440, 0.3157, 0.1562, 0.1885, 0.0955],
    [0.3474, 0.3315, 0.0825, 0.1761, 0.0623],
    [0.7222, 0.1207, 0.0349, 0.0869, 0.0357],
    [0.8569, 0.0839, 0.0090, 0.0361, 0.0140],
    [0.99756, 0.0016928, 0.000075221, 0.00066566, 0.000089347],
    [0.99951, 0.00035787, 0.000023425, 0.00016904, 0.000057101],
    [1.0000, 0.00003922, 0.0000028014, 0.000016451, 0.0000044107],
    [1.0000, 0.000012994, 0.00000011921, 0.0000011325, 0.00000041723],
    [1.0000, 0.00000077486, 0.0, 0.0, 0.0],
    [1.0000, 0.0000061989, 0.0, 0.00000035763, 0.00000017881],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0000, 0.00000083447, 0.0, 0.0, 0.0],
    [1.0000, 0.00000023842, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0000, 0.000000059605, 0.0, 0.0, 0.0],
    [1.0000, 0.00016403, 0.0, 0.0, 0.0],
    [0.99854, 0.0016022, 0.00000077486, 0.0, 0.0],
    [0.92627, 0.071472, 0.0017338, 0.00018275, 0.00015628],
]

data = np.array(raw_data)
layers = np.arange(1, 33)

plt.figure(figsize=(14, 7), facecolor="white")

# 稳重高级配色（ColorBrewer Set2 + 调整）
colors = [
    "#e15759",  # Yes  - 暖红
    "#4e79a7",  # No   - 深蓝
    "#59a14f",  # There- 森林绿
    "#edc949",  # The  - 柔和金黄
    "#af7aa1",
]  # I    - 紫灰

tokens = ["Yes", "No", "There", "the", "I"]

for i in range(5):
    plt.plot(
        layers,
        data[:, i],
        label=tokens[i],
        color=colors[i],
        linewidth=2.2,
        marker="o",
        markersize=5,
        markevery=1,
        alpha=0.95,
    )

plt.xlabel("Layer", fontsize=16, fontweight="bold")
plt.ylabel("Probability", fontsize=16, fontweight="bold")
plt.title("Token Probability Across 32 Decoder Layers", fontsize=19, pad=20)

plt.xticks(layers, fontsize=10, rotation=0)
plt.yticks(np.linspace(0, 1, 11), fontsize=12)
plt.ylim(0, 1.02)
plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, color="gray")
plt.legend(
    fontsize=14, loc="upper left", frameon=True, fancybox=False, edgecolor="black"
)

plt.savefig("find.pdf", dpi=300)  
plt.tight_layout()
plt.show()
