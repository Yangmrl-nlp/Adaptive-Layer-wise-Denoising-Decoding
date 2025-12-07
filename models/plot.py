import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import numpy as np

def plot_token_prob_bar(logits,name):

       
       probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
       
       plt.figure(figsize=(6, 4))
       plt.plot(range(len(probs)), probs,label = 'Probability Distribution', color = "#0072B2")
       
       plt.grid(True,              
             which='major',    
             axis='both',      
             linestyle='-',    
             color='gray',      
             linewidth=0.5      
       )
       plt.xlabel("Token ID")
       plt.ylabel("Probability")
       plt.legend(loc='upper right')
       plt.savefig(f"/mnt/data1/yangmrl/ALD^2/picture/{name}.pdf")
       plt.close()
