# -*- coding: utf-8 -*-
"""
Script to run the feature selection pipeline using mlxtend
Part of the Hyena Toolbox (see https://github.com/SysBioChalmers/Hyena)
@author: Christoph S. Börlin; Chalmers University of Technology, Gothenburg Sweden
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_predict
from xgboost.sklearn import XGBRegressor
from datetime import datetime
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
import csv
import math
import pickle

def evaluate_model(exp_data, tf_data, model, name):
    scores = cross_validate(model, tf_data, exp_data, scoring = ['neg_mean_squared_error', 'r2'], cv = 5)
    predictions = cross_val_predict(model, tf_data, exp_data, cv = 5)
    print(name)
    print('MSE: ' + str(round(-1 * scores['test_neg_mean_squared_error'].mean(),3)))
    print('R2: ' + str(round(scores['test_r2'].mean(),3)))   
    
    return round(scores['test_r2'].mean(),3), predictions

#import data
tf_data_proc = pd.read_csv('Data/CombinedTFdata_features.csv', index_col = 0)
#seperate expression data
exp_data = np.array(tf_data_proc['exp_data'])
tf_data_proc.drop('exp_data', inplace = True, axis = 1)
print('Done loading features - ' + datetime.now().strftime('%H:%M:%S'))

#define initial model
model = XGBRegressor(n_estimators = 100, objective = 'reg:squarederror', seed = 1234, learning_rate = 0.1, max_depth = 3)

#define outputname part
results_timestring = datetime.now().strftime('%y%m%d')

#run sequential feature selection
sfs = SFS(model, 
           k_features=(10,100), 
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

#save scores
scores_sfs = pd.DataFrame([[i, x['avg_score'], x['std_dev']] for i, x in sfs.get_metric_dict().items()], columns = ['Num_features', 'Avg_score', 'Std_dev'])
scores_sfs.to_csv('Results/FeatureSelection_Scores_' + results_timestring + '.csv')

#save selected features
with open('Results/FeatureSelection_Features_' + results_timestring + '.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for selected_feature in selected_features:
        wr.writerow([selected_feature])

#Evalute performance of model on all data
scores_all, predictions_all = evaluate_model(exp_data, tf_data_proc, model, 'All Features')

#Evalute performance of model on selected features
scores_sel, predictions_sel = evaluate_model(exp_data, tf_data_sel, model, 'Selected Features')

#save predictions and r2 scores together
model_output = pd.DataFrame([exp_data, predictions_all, predictions_sel]).transpose()
model_output.columns = ['exp_data', 'predictions_all_' + str(scores_all), 'predictions_sel_' + str(scores_sel)]
model_output.index = tf_data_proc.index
model_output.to_csv('Results/FeatureSelection_Predictions_' + results_timestring + '.csv')

#train model on full dataset
model.fit(tf_data_sel, exp_data)

#save model
pickle.dump(model, open('Results/FeatureSelection_Model_' + results_timestring + '.pkl', "wb"))