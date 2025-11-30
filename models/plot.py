import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import numpy as np

# def plot_token_prob_bar(logits, name):
#     sns.set_style("white")
#     sns.set_context("paper", font_scale=1.4)
    
#     probs = F.softmax(logits, dim=-1).squeeze(0).cpu().detach().numpy()
    
#     fig, ax = plt.subplots(figsize=(12, 4))
    
#     ax.bar(np.arange(len(probs)), probs,
#            width=1.0,               # 完全无缝衔接
#            color='#002E63',          # Nature 经典深蓝
#            edgecolor='none',         # 干净无边框
#            linewidth=0)
    
#     ax.set_xlabel("Token ID", fontweight='bold')
#     ax.set_ylabel("Probability", fontweight='bold')
#     ax.set_ylim(0, None)           # 从 0 开始，不留任何悬空
    
#     sns.despine(left=False, bottom=False)
#     plt.tight_layout(pad=0.3)
    
#     plt.savefig(f"/mnt/data1/yangmrl/ALW_debug/picture/{name}.pdf",
#                 dpi=400, bbox_inches='tight', pad_inches=0.01)
#     plt.close()
def plot_token_prob_bar(logits,name):

       # 转概率分布
       probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
       # 画图
       plt.figure(figsize=(6, 4))
       plt.plot(range(len(probs)), probs,label = 'Probability Distribution', color = "#0072B2")
       # plt.title(f"Layer {i+1} - Last Token Probability Distribution")
       plt.grid(True,              # 显示网格
             which='major',     # 主网格线（major ticks）
             axis='both',       # x 和 y 轴都加网格
             linestyle='-',    # 网格线样式，比如虚线
             color='gray',      # 网格线颜色
             linewidth=0.5      # 网格线宽度
       )
       plt.xlabel("Token ID")
       plt.ylabel("Probability")
       plt.legend(loc='upper right')
       plt.savefig(f"/mnt/data1/yangmrl/ALW_debug/picture/{name}.pdf")
       plt.close()