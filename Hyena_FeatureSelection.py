# -*- coding: utf-8 -*-
"""
Script to run the feature selection pipeline using mlxtend
Part of the Hyena Toolbox (see https://github.com/SysBioChalmers/Hyena)
@author: Christoph S. Börlin; Chalmers University of Technology, Gothenburg Sweden
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_predict
from xgboost import XGBRegressor
from datetime import datetime
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
import csv
import math

def evaluate_model(exp_data, tf_data, model, name):
    scores = cross_validate(model, tf_data, exp_data, scoring = ['neg_mean_squared_error', 'r2'], cv = 5)
    predictions = cross_val_predict(model, tf_data, exp_data, cv = 5)
    print(name)
    print('MSE: ' + str(round(-1 * scores['test_neg_mean_squared_error'].mean(),3)))
    print('R2: ' + str(round(scores['test_r2'].mean(),3)))   
    scatter_plotter(exp_data, predictions, scores, name)

def scatter_plotter(exp_data, predictions, scores, name):
    plt.figure(figsize = (8,6))
    ax = plt.gca()
    min_value = math.floor(min([min(exp_data), min(predictions)]))
    max_value = math.ceil(max([max(exp_data), max(predictions)]))
    plt.plot([min_value, max_value],[min_value, max_value],'r--', linewidth = 1.5, zorder = 0)
    plt.scatter(exp_data, predictions, s = 40, zorder = 1, alpha = 0.7, linewidth = 1)
    plt.xlabel('Experimental Log2 Expression Ratio', fontSize = 18)
    plt.ylabel('Predicted Log2 Expression Ratio Ratio', fontSize = 18)
    plt.title(name + ' - R2: ' + str(round(scores['test_r2'].mean(),3)), fontSize = 18)
    ax.xaxis.set_tick_params(labelsize = 18)
    ax.yaxis.set_tick_params(labelsize = 18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('Results/ModelPerformance_' + name.replace(' ', '') + '_' + results_timestring + '.png', dpi = 300, tight = True)
    plt.show()


#import data
tf_data_proc = pd.read_csv('Data/CombinedTFdata_features.csv', index_col = 0)
#seperate expression data
exp_data = np.array(tf_data_proc['exp_data'])
tf_data_proc.drop('exp_data', inplace = True, axis = 1)
print('Done loading features - ' + datetime.now().strftime('%H:%M:%S'))

#define initial model
model = XGBRegressor(n_estimators = 100, objective = 'reg:squarederror', seed = 1234, learning_rate = 0.1, max_depth = 2)

#define outputname part
results_timestring = datetime.now().strftime('%y%m%d')

#run sequential feature selection
sfs = SFS(model, 
           k_features=(10,20), 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='r2',
           cv=5,
           n_jobs = 6)

sfs = sfs.fit(tf_data_proc, exp_data)

selected_features = tf_data_proc.loc[:,sfs.k_feature_names_].columns.tolist()
tf_data_sel = tf_data_proc.loc[:,selected_features]
print('\n\nDone feature selection - ' + datetime.now().strftime('%H:%M:%S'))
print('Optimal number of features : ' + str(tf_data_sel.shape[1]) + ' out of ' + str(tf_data_proc.shape[1]))

#create figure for feature selection process
num_features = max(sfs.get_metric_dict().keys())
plt.figure(figsize = (8,6))
ax = plt.gca()
plt.axvline(x = len(selected_features), ymin = 0, ymax = 1, linewidth = 2, color = 'red')
plt.plot(list(sfs.get_metric_dict().keys()), [x['avg_score'] for x in sfs.get_metric_dict().values()], linewidth = 3, marker = 'o', markersize = 8)
ax.fill_between(list(sfs.get_metric_dict().keys()),
                [x['avg_score'] - x['std_dev'] for x in sfs.get_metric_dict().values()],
                [x['avg_score'] + x['std_dev'] for x in sfs.get_metric_dict().values()], alpha=0.2)
plt.text(len(selected_features) +0.5, 0.3, str(len(selected_features)) + ' features\n selected', fontsize = 18)
plt.ylabel('Crossvalidated R$^2$ score', fontSize = 18)
plt.xlabel('Number of features selected', fontSize = 18)
plt.xticks(range(1,num_features + 1), labels = [x if (x == 1 or x % 5 == 0) else '' for x in range(1,num_features + 1)])
ax.xaxis.set_tick_params(labelsize = 18)
ax.yaxis.set_tick_params(labelsize = 18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('/Results/FeatureSelection_Scores' + results_timestring + '.png', dpi = 300, tight = True)


#save selected features
with open('Results/FeatureSelection_Features_' + results_timestring + '.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for selected_feature in selected_features:
        wr.writerow([selected_feature])

#Evalute performance of model on all data
evaluate_model(exp_data, tf_data_proc, model, 'All Features')

#Evalute performance of model on selected features
evaluate_model(exp_data, tf_data_sel, model, 'Selected Features')

#train model on full dataset
model.fit(tf_data_sel, exp_data)

#save model
model.save_model('Results/FeatureSelection_Model_' + results_timestring + '.model')