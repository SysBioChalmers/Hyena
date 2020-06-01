# -*- coding: utf-8 -*-
"""
Script to analyse the conditional changes in transcription factor binding
Part of the Hyena Toolbox (see https://github.com/SysBioChalmers/Hyena)
@author: Christoph S. Börlin; Chalmers University of Technology, Gothenburg Sweden
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Step 0: Define available TFs and their group
tf_list = ['Cat8', 'Cbf1', 'Cst6', 'Ert1', 'Gcn4', 'Gcr1', 'Gcr2', 'Hap1', 'Ino2', 'Ino4', 'Leu3', 'Oaf1', 'Pip2', 'Rds2', 'Rgt1', 'Rtg1', 'Rtg3', 'Sip4', 'Stb5', 'Sut1', 'Tye7']

tf_groups_tmp = {'Zinc cluster' : ['Cat8', 'Ert1', 'Hap1', 'Leu3', 'Sip4', 'Oaf1', 'Pip2', 'Rds2', 'Sut1', 'Rgt1', 'Stb5'],
          'Zipper' : ['Cbf1', 'Cst6', 'Ino2', 'Ino4', 'Tye7', 'Rtg1', 'Rtg3', 'Gcn4'],
          'Other' : ['Gcr1', 'Gcr2']}
tf_groups_color = {'Zinc cluster': 'blue', 'Zipper': 'red', 'Other': 'grey'}
#reorder groups
tf_groups = {}
for group_name, group_list in tf_groups_tmp.items():
    for tf in group_list:
        tf_groups[tf] = {'group': group_name, 'color' : tf_groups_color[group_name]}
del tf_groups_tmp, group_name, group_list, tf


#Step 1: Load data
tf_data = pd.read_csv('Data/CombinedTFdata_intervals.csv', index_col = 0)

#take only data for Glu_sum and Eth_Sum and single tfs
tf_data = tf_data.loc[tf_data.Type.isin(['Glu_sum', 'Eth_sum']) & tf_data.TF.isin(tf_list)]

#combine tf and type name
tf_data.loc[:,'TF_type'] = tf_data.apply(lambda x: x.TF + '_' + x['Type'], axis = 1)
tf_data.drop(columns = ['TF', 'Type'], inplace = True)

#sum up intervals
tf_data = tf_data.groupby('TF_type').sum()
tf_data.drop(columns = 'Interval', inplace = True)

#Step 2: Analyse conditional dependency of tf binding

#count how many genes are bound in one or in both conditions
tf_data_overview = pd.DataFrame(index = tf_list, columns = ['Bound in Glu', 'Bound in only Glu', 'Bound in Eth', 'Bound in only Eth', 'Bound in min one', 'Bound in only one', 'Bound in both', 'Percent both bound', 'Log2 ratio list', 'Log2 ratio mean', 'TF group', 'TF color', 'Plot position'], dtype = np.float64)
#set type of tf group, color and log2 ratios as object
for x in ['Log2 ratio list', 'TF group', 'TF color']:
    tf_data_overview.loc[:, x] = tf_data_overview.loc[:, x].astype('object')
for tf in tf_list:
    tf_data_overview.loc[tf, 'Bound in Glu'] = sum(tf_data.loc[tf + '_Glu_sum'] > 0)
    tf_data_overview.loc[tf, 'Bound in only Glu'] = sum((tf_data.loc[tf + '_Glu_sum'] > 0) & (tf_data.loc[tf + '_Eth_sum'] == 0)) 
    
    tf_data_overview.loc[tf, 'Bound in Eth'] = sum(tf_data.loc[tf + '_Eth_sum'] > 0) 
    tf_data_overview.loc[tf, 'Bound in only Eth'] = sum((tf_data.loc[tf + '_Eth_sum'] > 0) & (tf_data.loc[tf + '_Glu_sum'] == 0))
    
    tf_data_overview.loc[tf, 'Bound in min one'] = sum((tf_data.loc[tf + '_Glu_sum'] > 0) | (tf_data.loc[tf + '_Eth_sum'] > 0)) 
    tf_data_overview.loc[tf, 'Bound in only one'] = tf_data_overview.loc[tf, 'Bound in only Glu'] + tf_data_overview.loc[tf, 'Bound in only Eth'] 
    
    tf_data_overview.loc[tf, 'Bound in both'] = sum((tf_data.loc[tf + '_Glu_sum'] > 0) & (tf_data.loc[tf + '_Eth_sum'] > 0)) 
    tf_data_overview.loc[tf, 'Percent both bound'] = tf_data_overview.loc[tf, 'Bound in both'] / tf_data_overview.loc[tf, 'Bound in min one'] * 100

    #save list of log2 ratios
    index_tmp = (tf_data.loc[tf + '_Glu_sum'] > 0) & (tf_data.loc[tf + '_Eth_sum'] > 0)
    tf_data_overview.at[tf, 'Log2 ratio list'] = np.log2(tf_data.loc[tf + '_Eth_sum', index_tmp] / tf_data.loc[tf + '_Glu_sum', index_tmp])
    tf_data_overview.loc[tf, 'Log2 ratio mean'] = tf_data_overview.loc[tf, 'Log2 ratio list'].mean()
    #add tf group and color
    tf_data_overview.loc[tf, 'TF group'] = tf_groups[tf]['group']
    tf_data_overview.loc[tf, 'TF color'] = tf_groups[tf]['color']
    
#Make barplot for percentage of genes bound in both conditions
#sort by ratio
tf_data_overview.sort_values(by = 'Percent both bound', inplace = True, ascending = True)
#give them an plot index
tf_data_overview.loc[:,'Plot position'] = range(len(tf_list))

plt.figure(figsize = (6,10))
ax = plt.gca()
#plot groups
for group in tf_groups_color.keys():
    tf_data_overview_tmp = tf_data_overview.loc[tf_data_overview.loc[:, 'TF group'] == group]
    plt.barh(tf_data_overview_tmp.loc[:,'Plot position'], tf_data_overview_tmp.loc[:, 'Percent both bound'], color = tf_groups_color[group], label = group)
plt.yticks(range(len(tf_data_overview.index)), tf_data_overview.index)
plt.xlabel('% of genes with tf reads in both conditions', fontSize = 16)
ax.xaxis.set_tick_params(labelsize = 16)
ax.yaxis.set_tick_params(labelsize = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(-0.5,20.5)
plt.legend(fontsize = 16)
plt.tight_layout()
plt.savefig('Results/ConditonalTFBinding_PercentOverview.png', dpi = 600, tight = True)
plt.show()

#Make boxplot for log2 ratios for genes bound in both conditions
#sort by log2 ratio mean
tf_data_overview.sort_values(by = 'Log2 ratio mean', inplace = True, ascending = True)
#give them an plot index
tf_data_overview.loc[:,'Plot position'] = range(len(tf_list))
plt.figure(figsize = (8,10))
ax = plt.gca()
plt.axvline(x = 0, ymin = 0, ymax = 1, linewidth = 2, color = 'grey', linestyle = '--')
#plot groups
boxplot_groups = {}
for group in tf_groups_color.keys():
    tf_data_overview_tmp = tf_data_overview.loc[tf_data_overview.loc[:, 'TF group'] == group]
    boxplot_groups[group] = plt.boxplot(tf_data_overview_tmp.loc[:, 'Log2 ratio list'], positions = tf_data_overview_tmp.loc[:,'Plot position'], patch_artist=True, boxprops = dict(facecolor = tf_groups_color[group]), medianprops=dict(color='white', linewidth = 2), vert=False)

plt.yticks(range(len(tf_data_overview.index)), tf_data_overview.index)
plt.xlabel('Log2 binding ratio Ethanol / Glucose', fontSize = 16)
ax.xaxis.set_tick_params(labelsize = 16)
ax.yaxis.set_tick_params(labelsize = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim()
[xmin, xmax] = plt.xlim()
plt.xlim([xmin, xmax + 3])
plt.legend([x["boxes"][0] for x in boxplot_groups.values()], boxplot_groups.keys(), fontsize = 16)
plt.tight_layout()
plt.savefig('Results/ConditonalTFBinding_Log2Ratios.png', dpi = 600, tight = True)
plt.show()