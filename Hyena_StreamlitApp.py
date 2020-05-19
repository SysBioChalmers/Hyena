# -*- coding: utf-8 -*-
"""
Main Hyena app
Part of the Hyena Toolbox (see https://github.com/SysBioChalmers/Hyena)
@author: Christoph S. Börlin; Chalmers University of Technology, Gothenburg Sweden
"""


import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor
import csv
import streamlit as st
from PIL import Image
import pickle

@st.cache
def load_data():
    #load intervall data
    tf_data = pd.read_csv('Results/CombinedTFdata_intervals_selected.csv', index_col = 0)
    
    #load gene expression data
    exp_data=pd.read_csv('Data/RNAseqData.csv',index_col=0)
    exp_data=exp_data[['Glu','Eth']]
    #select only genes where both are expressed
    exp_data = exp_data[exp_data.min(axis=1) >= 1]
    exp_data['Ratio'] = np.log(exp_data['Eth'] / exp_data['Glu'])
    
    #get genes
    gene_list = [x for x in tf_data.columns if x not in ['Interval', 'TF', 'Type']]
    
    #load selected features
    with open('Results/FeatureSelection_Features.csv', 'r') as myfile:
        reader = csv.reader(myfile)
        selected_features = [line[0] for line in reader]
        
    #load sequence data
    sequences = [[x for x in line.rstrip('\n\r').split('\t')] for line in open('Data/CENPK_1kbPromoterSequence.tsv', 'r')]
    sequences = {x[0].split(':')[0]: x[1] for x in sequences}
   
    return (tf_data, selected_features, exp_data, gene_list, sequences)

@st.cache
def create_artificial_promoters(tf_data, selected_gene):
    
    other_genes = [x for x in tf_data.columns if x not in ['Interval', 'TF', 'Type'] + [selected_gene]]
    tf_data_np = np.array(tf_data)
    tf_data_geneindex = {gene_name : gene_index for gene_index, gene_name in enumerate(tf_data.columns)}
    hybrid_promoter = np.zeros(shape = (tf_data.shape[0], len(range(-1000, -250, 50)) * len(other_genes)))
    tf_data_oriprom = tf_data_np[:, tf_data_geneindex[selected_gene]].copy()
    tf_data_newnames = []
    pos_counter = 0

    for i in range(-1000, -250, 50):
        index = tf_data.Interval == i
        
        for other_gene in other_genes:
        
            hybrid_promoter_tmp = tf_data_oriprom.copy()
            tf_data_newnames.append(selected_gene + '_' + other_gene + '_' + str(i).replace('-','m'))
            
            #change original prom for the selected interval
            hybrid_promoter_tmp[index] = tf_data_np[index, tf_data_geneindex[other_gene]]
    
            hybrid_promoter[:, pos_counter] = hybrid_promoter_tmp
            pos_counter += 1
    #combine data
    hybrid_promoter = pd.DataFrame(hybrid_promoter, columns = tf_data_newnames, dtype = np.float64)
    hybrid_promoter = tf_data.loc[:,['Interval', 'TF', 'Type']].join(hybrid_promoter)
    return hybrid_promoter

@st.cache
def create_features(tf_data, selected_gene):
    
    #get hybrid promoters
    hybrid_promoter = create_artificial_promoters(tf_data, selected_gene)
    
    #create features based on intervals
    interval_list = [[-1000, -500], [-500,0], [0, 500], [-1000, 500]]
    hybrid_promoter_features = []
    for interval in interval_list:
        tmp = hybrid_promoter.loc[(hybrid_promoter.Interval >= interval[0]) & (hybrid_promoter.Interval < interval[1])].groupby(['TF','Type']).sum().drop(columns = 'Interval')
        tmp.reset_index(inplace = True)
        tmp.loc[:,'Interval'] = str(interval[0]).replace('-', 'm') + '-' + str(interval[1]).replace('-', 'm')
        tmp.loc[:, 'Index'] = tmp.TF + '_' + tmp.Type + '_' + tmp.Interval
        tmp.set_index('Index', inplace = True)
        tmp.drop(columns = ['TF', 'Type', 'Interval'], inplace = True)
        hybrid_promoter_features.append(tmp)
    
    hybrid_promoter_features = pd.concat(hybrid_promoter_features)
    hybrid_promoter_features = hybrid_promoter_features.transpose()

    #take only neeeded features
    hybrid_promoter_features = hybrid_promoter_features.loc[:,selected_features]
    return hybrid_promoter_features

@st.cache(suppress_st_warning=True)                             
def get_predictions_overview(hybrid_promoter_features, target_value):
    predictions = pd.DataFrame(model.predict(hybrid_promoter_features), columns = ['Prediction'], index = hybrid_promoter_features.index)
    predictions.insert(0, 'Donor Gene', [x.split('_')[1] for x in hybrid_promoter_features.index])
    predictions.insert(1, 'Changed Region Start', [int(x.split('_')[2].replace('m', '-')) for x in hybrid_promoter_features.index])
    predictions.insert(2, 'Changed Region End', predictions['Changed Region Start'] + 49)
    predictions['Changed Region'] = predictions['Changed Region Start'].apply(lambda x: str(x) + ' to ' + str(x + 49))  
    predictions['Difference'] = target_value - predictions.Prediction
    predictions['Difference_Abs'] = np.abs(predictions['Difference'])
    predictions.sort_values(by = 'Difference_Abs', inplace = True)
    return predictions

#load data
(tf_data, selected_features, expression_data, gene_list, sequences) = load_data()
model = pickle.load(open('Results/FeatureSelection_Model.pkl', 'rb'))

#load logo
logo = Image.open('Logo_small.png')
st.image(logo, width = 400)

st.markdown('# Hybrid promoter design using advanced transcription factor binding predictions')
st.markdown('### Select options in the sidebar on the left')

st.sidebar.markdown('## This tool allows you to create hybrid promoters for fine tuning their conditional gene expression. Readme and Code can be found under https://github.com/SysBioChalmers/Hyena')
st.sidebar.markdown('### Written by Christoph S. Börlin, Chalmers University of Technology')

#select gene for modification and show expression details
selected_gene = st.sidebar.selectbox('Choose the promoter you would like to modify.', ['No selection'] + gene_list)

if selected_gene != 'No selection':
    
    selected_gene_expression = pd.DataFrame(expression_data.loc[[selected_gene], :])
    selected_gene_expression.columns = ['Exp. Glucose [TPM]', 'Exp. Ethanol [TPM]', 'Log2 Ratio Eth / Glu']
    
    st.markdown('### Expression overview for your choosen gene: ' + selected_gene)
    st.write(selected_gene_expression)
    
    #select targe value
    target_value = st.sidebar.number_input('Choose the desired expression ratio Ethanol / Glucose', value = 0.0, step = 0.1)
    
    #create the hybrid promoters
    hybrid_promoter_features = create_features(tf_data, selected_gene)
    
    #get prediction outcomes
    predictions = get_predictions_overview(hybrid_promoter_features, target_value)
    
    #select number of results to display
    show_top_x = st.sidebar.selectbox('Choose how many results will be shown', [2, 5, 10, 15, 20, 25, 50])
    
    st.markdown('### Best hybrid promoters with a desired expression ratio closest to ' + str(target_value))
    st.write(predictions.loc[:,['Donor Gene', 'Changed Region', 'Prediction', 'Difference']].head(show_top_x))
    
    #select hybrid promoter for sequence
    sequence_to_display = st.selectbox('Choose which hybrid promoter sequence to display', predictions.index[0:show_top_x].tolist())
    
    sequence_ori_up = sequences[selected_gene][0:1000 + predictions.loc[sequence_to_display, 'Changed Region Start']]
    sequence_ori_down = sequences[selected_gene][1000 + predictions.loc[sequence_to_display, 'Changed Region End'] + 1 :]
    sequence_donor = sequences[predictions.loc[sequence_to_display, 'Donor Gene']][
                            1000 + predictions.loc[sequence_to_display, 'Changed Region Start'] :
                            1000 + predictions.loc[sequence_to_display, 'Changed Region End'] + 1]
    
    #add bold tag for donor sequence
    sequence_donor = '<b>' + sequence_donor + '</b>'
    #combine sequences   
    sequence_hybrid_raw = sequence_ori_up + sequence_donor + sequence_ori_down
    
    #add line breaks every 70 bases  
    sequence_hybrid = []
    sequence_hybrid_tmp = ''
    for i in range(0,len(sequence_hybrid_raw)):   
        if len([x for x in sequence_hybrid_tmp if x in ['A', 'C', 'G', 'T']]) < 70:
            sequence_hybrid_tmp +=  sequence_hybrid_raw[i]
        else:
            sequence_hybrid.append(sequence_hybrid_tmp)
            sequence_hybrid_tmp = ''
       
    sequence_hybrid_markdown = '<p style="font-family:Courier New">' + '<br>'.join(sequence_hybrid) + '</p>'
  
    st.markdown('### Promoter sequence for artifical promoter ' + sequence_to_display + '.')
    st.markdown('#### The region ' + predictions.loc[sequence_to_display, 'Changed Region'] + 'bp upstream of the TSS of ' + selected_gene +' is replaced with the sequence from ' + predictions.loc[sequence_to_display, 'Donor Gene'] + ' (marked in bold)')
    st.markdown(sequence_hybrid_markdown, unsafe_allow_html = True)
