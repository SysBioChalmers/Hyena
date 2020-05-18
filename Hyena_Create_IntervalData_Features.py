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

def parseWigLike(selected_genes, selected_tf, path, date):
    #load data
    data_Glu=[[x for x in line.rstrip('\n\r').split('\t')] for line in open(path + selected_tf+'_Glu_ol_combRep_geneAssigned_' + date + '.wigLike')]
    data_Glu = {x[0] + '_' + x[1] : float(x[2]) for x in data_Glu if x[0] in selected_genes}
    
    data_Eth=[[x for x in line.rstrip('\n\r').split('\t')] for line in open(path + selected_tf+'_Eth_ol_combRep_geneAssigned_' + date + '.wigLike')]
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

main_chip_path = '../180126_ChIPexo_CENPK/Results_190402/'
main_chip_date = '190314'
selected_tfs = {x : [main_chip_path, main_chip_date] for x in ['Cat8','Cbf1','Ert1','Gcn4','Gcr1','Gcr2','Hap1','Ino2','Ino4','Leu3','Oaf1','Pip2','Rds2','Rgt1','Rtg1','Rtg3','Sip4','Stb5','Sut1','Tye7']}

#add Cst6
selected_tfs['Cst6'] = ['../191203_Cst6_ChIPexo/', '191203']

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
for selected_tf, (path, date) in selected_tfs.items():
        tf_data = tf_data.append(parseWigLike(selected_genes, selected_tf, path, date))
print('Done loading and data - ' + datetime.now().strftime('%H:%M:%S'))



tf_data_intervals = pd.DataFrame()

for i in range(-1000,500,50):
    tf_data_intervals_tmp = tf_data.loc[(tf_data.Pos >= i) & (tf_data.Pos < i + 50)].groupby(['Gene','TF']).agg({'Glu_Value': ['sum', above_zero], 'Eth_Value': ['sum', above_zero], 'Diff_Value' : ['sum', above_zero, below_zero]})
    tf_data_intervals_tmp.columns = ['Glu_sum', 'Glu_az', 'Eth_sum', 'Eth_az', 'Diff_sum', 'Diff_az', 'Diff_bz']
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

#set multiindex
tf_data_intervals.set_index(['Gene', 'TF', 'Interval'], inplace = True)

#melt it for reordering
tf_data_intervals = pd.melt(tf_data_intervals, id_vars = ['Gene', 'TF', 'Interval'],
            value_vars=['Glu_sum', 'Glu_az', 'Eth_sum', 'Eth_az', 'Diff_sum', 'Diff_az', 'Diff_bz'],
            var_name = 'Type', value_name = 'Value')

#reorder it into a wide format
types = ['Glu_sum', 'Glu_az', 'Eth_sum', 'Eth_az', 'Diff_sum', 'Diff_az', 'Diff_bz']
tf_data_wide = pd.DataFrame(columns = ['Interval', 'TF', 'Type'] + selected_genes)
tf_data_wide_list = []
for i in range(-1000,500,50):
    for tf in selected_tfs.keys():
        for type_name in types:
                tf_data_wide_tmp = tf_data_intervals.loc[
                    (tf_data_intervals.TF == tf) & 
                    (tf_data_intervals.Interval == i) &
                    (tf_data_intervals.Type == type_name), ['Gene','Value']].set_index('Gene').transpose()
                
                tf_data_wide_tmp.loc['Value', 'Interval'] = i
                tf_data_wide_tmp.loc['Value', 'TF'] = tf
                tf_data_wide_tmp.loc['Value', 'Type'] = type_name

                
                tf_data_wide_list.append(tf_data_wide_tmp)

tf_data_wide = tf_data_wide.append(tf_data_wide_list)
tf_data_wide.fillna(0, inplace = True)
tf_data_wide.reset_index(inplace = True)
tf_data_wide.drop(column = 'Value', inplace = True)

#add sum for all TFs
tf_data_wide_alltfs = pd.DataFrame(columns = tf_data_wide.columns)
for i in range(-1000,500,50):
    for type_name in types:
        tf_data_wide_tmp = tf_data_wide.loc[(tf_data_wide.Interval == i) & (tf_data_wide.Type == type_name)].drop(columns = ['Interval', 'TF', 'Type']).sum(axis = 0).transpose()
        tf_data_wide_tmp.loc['Interval'] = i
        tf_data_wide_tmp.loc['TF'] = 'AllTFs'
        tf_data_wide_tmp.loc['Type'] = type_name
        
        #add it to specific table
        tf_data_wide_alltfs = tf_data_wide_alltfs.append(tf_data_wide_tmp, ignore_index = True)
#add it to overall table
tf_data_wide = tf_data_wide.append(tf_data_wide_alltfs, ignore_index = True)

#add sum for TF groups
tf_data_wide_tfgroups = pd.DataFrame(columns = tf_data_wide.columns)
groups = {'ZincCluster' : ['Cat8', 'Ert1', 'Hap1', 'Leu3', 'Sip4', 'Oaf1', 'Pip2', 'Rds2', 'Sut1', 'Rgt1', 'Stb5'],
          'Zipper' : ['Cbf1', 'Ino2', 'Ino4', 'Tye7', 'Rtg1', 'Rtg3', 'Gcn4', 'Gcr1', 'Gcr2']}
for i in range(-1000,500,50):
    for group, group_tfs in groups.items():
        for type_name in types:
            tf_data_wide_tmp = tf_data_wide.loc[(tf_data_wide.Interval == i) & (tf_data_wide.Type == type_name) & (tf_data_wide.TF.isin(group_tfs))].drop(columns = ['Interval', 'TF', 'Type']).sum(axis = 0).transpose()
            tf_data_wide_tmp.loc['Interval'] = i
            tf_data_wide_tmp.loc['TF'] = group
            tf_data_wide_tmp.loc['Type'] = type_name
            
            #add it to specific table
            tf_data_wide_tfgroups = tf_data_wide_tfgroups.append(tf_data_wide_tmp, ignore_index = True)        
#add it to overall table
tf_data_wide = tf_data_wide.append(tf_data_wide_tfgroups, ignore_index = True)

#add sum for TF pairs
tf_data_wide_tfpairs = pd.DataFrame(columns = tf_data_wide.columns)
pairs = [['Ino2', 'Ino4'],
          ['Oaf1', 'Pip2'],
          ['Gcr1', 'Gcr2'],
          ['Cat8', 'Sip4'],
          ['Ert1', 'Rds2'],
          ['Rtg1', 'Rtg3']]
for i in range(-1000,500,50):
    for pair in pairs:
        for type_name in types:
            tf_data_wide_tmp = tf_data_wide.loc[(tf_data_wide.Interval == i) & (tf_data_wide.Type == type_name) & (tf_data_wide.TF.isin(pair))].drop(columns = ['Interval', 'TF', 'Type']).sum(axis = 0).transpose()
            tf_data_wide_tmp.loc['Interval'] = i
            tf_data_wide_tmp.loc['TF'] = pair[0] + '-' + pair[1]
            tf_data_wide_tmp.loc['Type'] = type_name
            
            #add it to specific table
            tf_data_wide_tfpairs = tf_data_wide_tfpairs.append(tf_data_wide_tmp, ignore_index = True)        
#add it to overall table
tf_data_wide = tf_data_wide.append(tf_data_wide_tfpairs, ignore_index = True)

#save it
tf_data_wide.to_csv('Data/CombinedTFdata_intervals.csv')


#create features based on intervals
interval_list = [[-1000, -500], [-500,0], [0, 500]]
tf_data_proc = []
for interval in interval_list:
    tmp = tf_data_wide.loc[(tf_data_wide.Interval >= interval[0]) & (tf_data_wide.Interval < interval[1])].groupby(['TF','Type']).sum().drop(columns = 'Interval')
    tmp.reset_index(inplace = True)
    tmp.loc[:,'Interval'] = str(interval[0]).replace('-', 'm') + '-' + str(interval[1]).replace('-', 'm')
    tmp.loc[:, 'Index'] = tmp.TF + '_' + tmp.Type + '_' + tmp.Interval
    tmp.set_index('Index', inplace = True)
    tmp.drop(columns = ['TF', 'Type', 'Interval'], inplace = True)
    tf_data_proc.append(tmp)

tf_data_proc = pd.concat(tf_data_proc)
tf_data_proc = tf_data_proc.transpose()

#save all data
tf_data_proc['exp_data'] = exp_data
tf_data_proc.to_csv('Data/CombinedTFdata_features.csv')


    
    
    
    
    
    
    
    