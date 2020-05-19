# -*- coding: utf-8 -*-
"""
Script to filter all interval data using only selected features for the final model
Part of the Hyena Toolbox (see https://github.com/SysBioChalmers/Hyena)
@author: Christoph S. Börlin; Chalmers University of Technology, Gothenburg Sweden
"""

import pandas as pd
import csv

#load intervall data
tf_data = pd.read_csv('Data/CombinedTFdata_intervals.csv', index_col = 0)

#load selected features
with open('Results/FeatureSelection_Features.csv', 'r') as myfile:
    reader = csv.reader(myfile)
    selected_features = [line[0] for line in reader]
    
#get used tf / type combos for selection
tf_type_combos = [[x.split('_')[0], '_'.join(x.split('_')[1:-1])] for x in selected_features]

tf_data_sel = []
for tf, type_name in tf_type_combos:
    tf_data_sel_tmp = tf_data.loc[(tf_data.TF == tf) & (tf_data.Type == type_name)]
    tf_data_sel.append(tf_data_sel_tmp)
    
tf_data_sel = pd.concat(tf_data_sel, ignore_index = True)

#save selected intervall data
tf_data_sel.to_csv('Results/CombinedTFdata_intervals_selected.csv')
