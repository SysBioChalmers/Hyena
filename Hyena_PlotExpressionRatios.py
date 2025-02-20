# -*- coding: utf-8 -*-
"""
Script to plot the gene expression ratios
Part of the Hyena Toolbox (see https://github.com/SysBioChalmers/Hyena)
@author: Christoph S. Börlin; Chalmers University of Technology, Gothenburg Sweden
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load gene expression data
exp_data=pd.read_csv('Data/RNAseqData.csv',index_col=0)
exp_data=exp_data[['Glu','Eth']]
#select only genes where both are expressed
exp_data = exp_data[exp_data.min(axis=1) >= 1]
exp_data['Ratio'] = np.log(exp_data['Eth'] / exp_data['Glu'])

#create overview
fig=plt.figure(figsize=(5.55 / 2.54, 7.6 / 2.54))
plt.rcParams['font.sans-serif'] = 'Arial'
ax = {}
ax[1] = plt.subplot(2,1,2)
ax[1].axvline(x = 0, ymin = 0, ymax = 0.95, linewidth = 1.5, color = 'grey', linestyle = '--')
ax[1].hist(exp_data.Ratio, bins = np.arange(round(exp_data.Ratio.min(),1), round(exp_data.Ratio.max(),1), 0.1), color = 'green', density = True)
ax[1].text(0.65, 1.9, 'Mean ' + str(round(exp_data.Ratio.mean(), 2)), family = 'monospace', fontsize = 11)
ax[1].text(0.65, 1.4, 'Mean ' + str(round(exp_data.Ratio.abs().mean(), 2)), family = 'monospace', fontsize = 11)
ax[1].text(0.55, 1.0, '(Abs)', family = 'monospace', fontsize = 12)
ax[1].xaxis.set_tick_params(labelsize = 11)
ax[1].yaxis.set_tick_params(labelsize = 11)
ax[1].set_ylabel('Density',fontSize=11)
ax[1].set_xlabel('Log2 Ratio (Eth / Glu)', fontSize = 11)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylim([0, 2.5])

#plot for specific genes
gene_1 = 'FBA1'
gene_2 = 'ADH2'
gene_3 = 'HHT2'
ax[2] = plt.subplot(2,1,1)
#draw glucose expresssion
ax[2].bar([1, 4, 7], [exp_data.loc[gene_1, 'Glu'], exp_data.loc[gene_2, 'Glu'], exp_data.loc[gene_3, 'Glu']], color = [0, 0, 1], label = 'Glu')
#draw ethanol expression
ax[2].bar([2, 5, 8], [exp_data.loc[gene_1, 'Eth'], exp_data.loc[gene_2, 'Eth'], exp_data.loc[gene_3, 'Eth']], color = [220/255,0,0], label = 'Eth')
ax[2].xaxis.set_tick_params(labelsize = 11)
ax[2].yaxis.set_tick_params(labelsize = 11)
ax[2].set_xticks([1.5, 4.5, 7.5])
ax[2].set_xticklabels([gene_1, gene_2, gene_3])
ax[2].set_yticklabels([])
ax[2].set_ylabel('Expression', fontSize = 11)
ax[2].legend(fontsize = 10, loc = 1, bbox_to_anchor=(1.13, 1))
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)

#plt.tight_layout()
plt.show
plt.savefig('Results/ExpressionRatios.png', dpi=600, bbox_inches="tight")