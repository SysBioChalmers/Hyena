# -*- coding: utf-8 -*-
"""
Script to creat plots for feature selection process
Part of the Hyena Toolbox (see https://github.com/SysBioChalmers/Hyena)
@author: Christoph S. Börlin; Chalmers University of Technology, Gothenburg Sweden
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

#load data
scores_sfs = pd.read_csv('Results/FeatureSelection_Scores.csv', index_col = 0)
predictions = pd.read_csv('Results/FeatureSelection_Predictions.csv', index_col = 0)
with open('Results/FeatureSelection_Features.csv', 'r') as myfile:
    reader = csv.reader(myfile)
    selected_features = [line[0] for line in reader]

#Setup figure
plt.figure(figsize = (16.8 / 2.54, 12.5 / 2.54))
plt.rcParams['font.sans-serif'] = 'Arial'
ax = {}
ax[0] = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax[1] = plt.subplot2grid((2, 2), (1, 0))
ax[2] = plt.subplot2grid((2, 2), (1, 1))

#create figure for feature selection process
num_features = scores_sfs['Num_features'].max()
ax[0].axvline(x = len(selected_features), ymin = 0, ymax = 1, linewidth = 1.5, color = 'red')
ax[0].plot(scores_sfs['Num_features'], scores_sfs['Avg_score'], linewidth = 1.5, marker = 'o', markersize = 3)
ax[0].fill_between(scores_sfs['Num_features'], scores_sfs['Avg_score'] - scores_sfs['Std_dev'], scores_sfs['Avg_score'] + scores_sfs['Std_dev'], alpha=0.2)
ax[0].text(len(selected_features) + 5, 0.2, str(len(selected_features)) + ' features\n selected', fontsize = 12)
ax[0].set_ylabel('Crossvalidated R$^2$ score', fontSize = 11)
ax[0].set_xlabel('Number of features selected', fontSize = 11)
ax[0].xaxis.set_tick_params(labelsize = 11)
ax[0].yaxis.set_tick_params(labelsize = 11)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

scatter_info = {'All Features': 1, 'Selected Features' : 2}

for name, i in scatter_info.items():

    min_value = math.floor(min([min(predictions.loc[:,'exp_data']), min(predictions.iloc[:, i])]))
    max_value = math.ceil(max([max(predictions.loc[:,'exp_data']), max(predictions.iloc[:, i])]))
    ax[i].plot([min_value, max_value],[min_value, max_value],'r--', linewidth = 1.5, zorder = 0)
    ax[i].scatter(predictions.loc[:,'exp_data'], predictions.iloc[:, i], s = 30, zorder = 1, alpha = 0.6, linewidth = 1)
    ax[i].set_xlabel('Experimental Log2 Expression Ratio', fontSize = 11)
    ax[i].set_ylabel('Predicted Log2 Expression Ratio', fontSize = 11)
    ax[i].set_title(name + ' - R$^2$: ' + predictions.columns[i].split('_')[-1], fontSize = 11)
    ax[i].xaxis.set_tick_params(labelsize = 11)
    ax[i].yaxis.set_tick_params(labelsize = 11)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)


plt.tight_layout()
plt.savefig('Results/FeatureSelection.png', dpi = 600, tight = True)

