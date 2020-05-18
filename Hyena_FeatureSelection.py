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

#import data
tf_data_proc = pd.read_csv('Data/CombinedTFdata_features.csv', index_col = 0)
#sepearte expression data
exp_data = np.array(tf_data_proc['exp_data'])
tf_data_proc.drop('exp_data', inplace = True, axis = 1)

print('Done loading features - ' + datetime.now().strftime('%H:%M:%S'))

#define initial model
model = XGBRegressor(n_estimators = 100, objective = 'reg:squarederror', seed = 1234, learning_rate = 0.1, max_depth = 2)

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

results_timestring = datetime.now().strftime('%y%m%d')

fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (w. StdDev)')
plt.xlabel('Number of features selected')
plt.ylabel('CV r2')
plt.savefig('Results/FeatureSelection_Scores_' + results_timestring + '.png', dpi = 300, bbox_inches='tight')
plt.show

#save selected features
with open('Results/FeatureSelection_Features_' + results_timestring + '.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for selected_feature in selected_features:
        wr.writerow([selected_feature])

#run CV to get scores of initial model on all data
scores_initial = cross_validate(model, tf_data_proc, exp_data, scoring = ['neg_mean_squared_error', 'r2'], cv = 5)
print('\nInitial model')
print('MSE: ' + str(round(-1 * scores_initial['test_neg_mean_squared_error'].mean(),3)))
print('R2: ' + str(round(scores_initial['test_r2'].mean(),3)))   

#run CV with selected features
scores_featuresel = cross_validate(model, tf_data_sel, exp_data, scoring = ['neg_mean_squared_error', 'r2'], cv = 5)
predictions_featuresel = cross_val_predict(model, tf_data_sel, exp_data, cv = 5)
print('\nWith reverse feature selection')
print('MSE: ' + str(round(-1 * scores_featuresel['test_neg_mean_squared_error'].mean(),3)))
print('R2: ' + str(round(scores_featuresel['test_r2'].mean(),3)))


#train model on full dataset
model.fit(tf_data_sel, exp_data)

#save model
model.save_model('Results/FeatureSelection_Model_' + results_timestring + '.model')