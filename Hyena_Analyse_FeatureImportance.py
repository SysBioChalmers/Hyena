# -*- coding: utf-8 -*-
"""
Script to analyse the feature importance of the final model
Part of the Hyena Toolbox (see https://github.com/SysBioChalmers/Hyena)
@author: Christoph S. Börlin; Chalmers University of Technology, Gothenburg Sweden
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
import pickle

#load model
model_name = 'FeatureSelection_Model'
model = pickle.load(open('Results/' + model_name + '.pkl', 'rb'))

#get feature importance and convert into list
feature_importance = [[key, value] for key, value in model.get_booster().get_score(importance_type='gain').items()]
feature_importance.sort(key = lambda x: x[1], reverse = False)

# #Plot feature importance
# plt.figure(figsize = (6,16))
# ax = plt.gca()
# plt.barh([x[0] for x in feature_importance], [x[1] for x in feature_importance])
# plt.xlabel('Feature importance', fontSize = 13)
# ax.xaxis.set_tick_params(labelsize = 13)
# ax.yaxis.set_tick_params(labelsize = 13)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.savefig('Results/FeatureImportance_' + model_name + '.png', dpi = 300, bbox_inches = 'tight')
# plt.show()

#Split feature importance by TF, type, condition and interval
feature_importance_categories = ['tf', 'condition', 'feature type', 'interval']
feature_importance_split = {x : {} for x in feature_importance_categories}
for fi in feature_importance:
    for tmp_cat, tmp_value in zip(feature_importance_categories, fi[0].split('_')):
        feature_importance_split[tmp_cat][tmp_value] = feature_importance_split[tmp_cat].get(tmp_value, []) + [fi[1]]

#change interval names
key_list = list(feature_importance_split['interval'].keys())
for key in key_list:
    new_key = key.replace('-', '\nto ') .replace('m', '-')
    feature_importance_split['interval'][new_key] = feature_importance_split['interval'].pop(key)
    
#change feature type name
feature_importance_split['feature type']['Pos > zero'] = feature_importance_split['feature type'].pop('az')
feature_importance_split['feature type']['Pos < zero'] = feature_importance_split['feature type'].pop('bz')

#convert dict into list for sorting
feature_importance_split_list = {x : [] for x in feature_importance_split.keys()}
for tmp_cat in feature_importance_split.keys():
    for tmp_key, tmp_value in feature_importance_split[tmp_cat].items():
        feature_importance_split_list[tmp_cat].append([tmp_key, np.mean(tmp_value)])
    feature_importance_split_list[tmp_cat].sort(key = lambda x: x[1], reverse = True)   

#resort tf the other way
feature_importance_split_list['tf'].sort(key = lambda x: x[1], reverse = False)  



plt.figure(figsize = (16.8 / 2.54, 16.8 / 2.54))
plt.rcParams['font.sans-serif'] = 'Arial'
ax = {}
ax[1] = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
ax[2] = plt.subplot2grid((3, 2), (0, 1))
ax[3] = plt.subplot2grid((3, 2), (1, 1))
ax[4] = plt.subplot2grid((3, 2), (2, 1))

ax[1].barh([x[0] for x in feature_importance_split_list['tf']], [x[1] for x in feature_importance_split_list['tf']])
ax[1].set_xlabel('Feature importance', fontSize = 11)
ax[1].xaxis.set_tick_params(labelsize = 11)
ax[1].yaxis.set_tick_params(labelsize = 11)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_title('Grouped by transcription factor', fontSize = 12)

#display it for condition interval and feature type
for i, cat in enumerate(['condition', 'feature type', 'interval',]):
    ax[i + 2].bar([x[0] for x in feature_importance_split_list[cat]], [x[1] for x in feature_importance_split_list[cat]])
    ax[i + 2].set_ylabel('Feature importance', fontSize = 11)
    ax[i + 2].xaxis.set_tick_params(labelsize = 11)
    ax[i + 2].yaxis.set_tick_params(labelsize = 11)
    ax[i + 2].spines['top'].set_visible(False)
    ax[i + 2].spines['right'].set_visible(False)
    ax[i + 2].set_title('Grouped by ' + cat, fontSize = 12)
  
plt.tight_layout()
plt.savefig('Results/FeatureImportanceSplit_' + model_name + '.png', dpi = 600, tight = True)
plt.show()