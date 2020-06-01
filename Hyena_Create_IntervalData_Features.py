# -*- coding: utf-8 -*-
"""
Main data processing script, loads ChIP-exo transcription factor data and processes into interval based data.
Also creates the features used for the machine learning models.
Part of the Hyena Toolbox (see https://github.com/SysBioChalmers/Hyena)
@author: Christoph S. Börlin; Chalmers University of Technology, Gothenburg Sweden
"""

import pandas as pd
import numpy as np
from datetime import datetime

def parseWigLike(selected_genes, selected_tf):
    #load data
    path_to_ChIPexo = 'Data_ChIPexo/'
    data_Glu=[[x for x in line.rstrip('\n\r').split('\t')] for line in open(path_to_ChIPexo + selected_tf+'_Glu.wigLike')]
    data_Glu = {x[0] + '_' + x[1] : float(x[2]) for x in data_Glu if x[0] in selected_genes}
    
    data_Eth=[[x for x in line.rstrip('\n\r').split('\t')] for line in open(path_to_ChIPexo + selected_tf+'_Eth.wigLike')]
    data_Eth = {x[0] + '_' + x[1] : float(x[2]) for x in data_Eth if x[0] in selected_genes}
    
    combined_keys = list(set(list(data_Glu.keys()) + list(data_Eth.keys())))
    combined_keys = [x.split('_') for x in combined_keys]
    combined_keys.sort(key = lambda x : (x[0], int(x[1])))

    tf_data = []
    for gene, pos in combined_keys:
        tf_data.append({'Gene' : gene,
                        'TF' : selected_tf,
                        'Pos' : int(pos) - 1000,
                        'Glu_Value' : data_Glu.get(gene + '_' + pos, 0),
                        'Eth_Value' : data_Eth.get(gene + '_' + pos, 0),
                        'Diff_Value' : data_Eth.get(gene + '_' + pos, 0) - data_Glu.get(gene + '_' + pos, 0)})
        
    tf_data = pd.DataFrame(tf_data)
    
    return tf_data       


def above_zero(data):
    '''
    Function to get number of positions with a tf binding value above zero, return integer
    '''
    return sum(data>0)

def below_zero(data):
    '''
    Function to get number of positions with a tf binding value above zero, return integer
    '''
    return sum(data<0)
    

print('Start loading and creating features - ' + datetime.now().strftime('%H:%M:%S'))

selected_tfs = ['Cat8','Cbf1', 'Cst6', 'Ert1', 'Gcn4', 'Gcr1', 'Gcr2', 'Hap1', 'Ino2', 'Ino4', 'Leu3', 'Oaf1', 'Pip2', 'Rds2', 'Rgt1', 'Rtg1', 'Rtg3', 'Sip4', 'Stb5', 'Sut1', 'Tye7']

#Choose a gene list
selected_genes = [line.strip('\r\n') for line in open('Data/GenesWithTSSAnnotation_Yeast82.csv')]
    
#load gene expression data
exp_data=pd.read_csv('Data/RNAseqData.csv',index_col=0)
exp_data=exp_data[['Glu','Eth']]
#select only genes where both are expressed
exp_data = exp_data[exp_data.min(axis=1) >= 1]
exp_data['Ratio'] = np.log(exp_data['Eth'] / exp_data['Glu'])

#update gene list and reindex
selected_genes = [x for x in selected_genes if x in exp_data.index]
exp_data = exp_data.reindex(selected_genes)
exp_data = np.array(exp_data.Ratio)

#load tf data
tf_data = pd.DataFrame(columns = ['Gene', 'TF', 'Pos', 'Glu_Value', 'Eth_Value', 'Diff_Value'])
for selected_tf in selected_tfs:
        tf_data = tf_data.append(parseWigLike(selected_genes, selected_tf))
print('Done loading data - ' + datetime.now().strftime('%H:%M:%S'))

tf_data_intervals = pd.DataFrame()

data_columns = ['Glu_sum', 'Glu_az', 'Eth_sum', 'Eth_az', 'Diff_sum', 'Diff_az', 'Diff_bz']

for i in range(-1000,500,50):
    tf_data_intervals_tmp = tf_data.loc[(tf_data.Pos >= i) & (tf_data.Pos < i + 50)].groupby(['Gene','TF']).agg({'Glu_Value': ['sum', above_zero], 'Eth_Value': ['sum', above_zero], 'Diff_Value' : ['sum', above_zero, below_zero]})
    tf_data_intervals_tmp.columns = data_columns
    for col in ['Glu_az', 'Eth_az', 'Diff_az', 'Diff_bz']:
        tf_data_intervals_tmp[col] = tf_data_intervals_tmp[col].astype(np.int64)
    tf_data_intervals_tmp.loc[:,'Interval'] = i
    tf_data_intervals = tf_data_intervals.append(tf_data_intervals_tmp)

tf_data_intervals.reset_index(inplace=True)  

#reorder columns
cols = tf_data_intervals.columns.tolist()
tf_data_intervals = tf_data_intervals.loc[:, cols[0:2] + [cols[-1]] + cols[2:-1]]

#sort and round
tf_data_intervals.sort_values(by = ['Gene', 'TF', 'Interval'], inplace = True, ignore_index = True)
tf_data_intervals = tf_data_intervals.round(decimals = 3)

#melt it for reordering
tf_data_intervals = pd.melt(tf_data_intervals, id_vars = ['Gene', 'TF', 'Interval'], value_vars = data_columns,
            var_name = 'Type', value_name = 'Value')

print('Done gattering data - ' + datetime.now().strftime('%H:%M:%S'))

#reorder it into a wide format
tf_data_wide = pd.DataFrame(columns = ['Interval', 'TF', 'Type'] + selected_genes)
tf_data_wide_list = []

#create list of index
index_interval = {}
for i in range(-1000,500,50):
    index_interval[i] = tf_data_intervals.Interval == i
index_tf = {}
for tf in selected_tfs:
    index_tf[tf] = tf_data_intervals.TF == tf
index_type = {}
for data_column in data_columns:
    index_type[data_column] = tf_data_intervals.Type == data_column
        
for i in range(-1000,500,50):
    for tf in selected_tfs:
        for data_column in data_columns:
            index_combined = index_interval[i] & index_tf[tf] & index_type[data_column]
            tf_data_wide_tmp = tf_data_intervals.loc[index_combined, ['Gene','Value']].set_index('Gene').transpose()
            
            tf_data_wide_tmp.loc['Value', 'Interval'] = i
            tf_data_wide_tmp.loc['Value', 'TF'] = tf
            tf_data_wide_tmp.loc['Value', 'Type'] = data_column
            
            tf_data_wide_list.append(tf_data_wide_tmp)

print('Done reordering data part 1 - ' + datetime.now().strftime('%H:%M:%S'))

tf_data_wide = tf_data_wide.append(tf_data_wide_list)
tf_data_wide.fillna(0, inplace = True)
tf_data_wide.reset_index(inplace = True)
tf_data_wide.drop(columns = 'index', inplace = True)

print('Done reordering data part 2 - ' + datetime.now().strftime('%H:%M:%S'))

#add sum for TF groups
groups = {}
groups['All'] = selected_tfs
groups['Zipper'] =  ['Cbf1', 'Cst6', 'Gcn4', 'Ino2', 'Ino4', 'Rtg1', 'Rtg3']
groups['ZincCluster'] = ['Cat8', 'Ert1', 'Hap1', 'Leu3', 'Oaf1', 'Pip2', 'Rds2', 'Rgt1', 'Sip4', 'Sut1', 'Stb5']
groups['Ino2-Ino4'] = ['Ino2', 'Ino4'] 
groups['Oaf1-Pip2'] = ['Oaf1', 'Pip2']
groups['Gcr1-Gcr2'] = ['Gcr1', 'Gcr2']
groups['Cat8-Sip4'] = ['Cat8', 'Sip4']
groups['Ert1-Rds2'] = ['Ert1', 'Rds2']
groups['Rtg1-Rtg3'] = ['Rtg1', 'Rtg3']

tf_data_wide_np = np.array(tf_data_wide.drop(columns = ['Interval', 'TF', 'Type']).transpose())
tf_data_wide_tfgroups = []
for i in range(-1000,500,50):
    index = tf_data_wide.Interval == i
    for group, group_tfs in groups.items():
        tf_index = tf_data_wide.TF.isin(group_tfs)
        for data_column in data_columns:
            index_combined = index & (tf_data_wide.Type == data_column) & tf_index
            tf_data_wide_tmp = tf_data_wide_np[:, index_combined].sum(axis = 1)
            tf_data_wide_tmp = pd.DataFrame(tf_data_wide_tmp, index = selected_genes)
            tf_data_wide_tmp.loc['Interval'] = i
            tf_data_wide_tmp.loc['TF'] = group
            tf_data_wide_tmp.loc['Type'] = data_column
            
            #append it to list
            tf_data_wide_tfgroups.append(tf_data_wide_tmp)
#add it to overall table
tf_data_wide = tf_data_wide.append(pd.concat(tf_data_wide_tfgroups, axis = 1).transpose())

#save it
tf_data_wide.to_csv('Data/CombinedTFdata_intervals.csv')

print('Done creating groups - ' + datetime.now().strftime('%H:%M:%S'))

#create features based on intervals
interval_list = [[-1000, -500], [-500,0], [0, 500], [-1000, 500]]
tf_features = []
for interval in interval_list:
    tmp = tf_data_wide.loc[(tf_data_wide.Interval >= interval[0]) & (tf_data_wide.Interval < interval[1])].groupby(['TF','Type']).sum().drop(columns = 'Interval')
    tmp.reset_index(inplace = True)
    tmp.loc[:,'Interval'] = str(interval[0]).replace('-', 'm') + '-' + str(interval[1]).replace('-', 'm')
    tmp.loc[:, 'Index'] = tmp.TF + '_' + tmp.Type + '_' + tmp.Interval
    tmp.set_index('Index', inplace = True)
    tmp.drop(columns = ['TF', 'Type', 'Interval'], inplace = True)
    tf_features.append(tmp)

tf_features = pd.concat(tf_features)
tf_features = tf_features.transpose()

#save all data
tf_features['exp_data'] = exp_data
tf_features.to_csv('Data/CombinedTFdata_features.csv')

print('Done creating features - ' + datetime.now().strftime('%H:%M:%S'))
    
    
    
    
    
    
    
    