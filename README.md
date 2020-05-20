![Hyena Logo](/Logo_small.png)
# Hybrid promoter design using advanced transcription factor binding predictions

This repostiory contains the Hyena toolbox scripts needed to run the Hyena streamlit app as well as all data preprocessing scripts.

- Last update: 2020-05-19

This repository is administered by Christoph Boerlin (https://github.com/ChristophBoerlin), Division of Systems and Synthetic Biology, Department of Biology and Biological Engineering, Chalmers University of Technology

# Introduction
This toolbox uses gradient boosting regression trees (xgboost) to create a model predicting the ratio of gene expression between growth in glucose and growth in ethanol for the yeast _Saccharomyces cerevisiae_.
This model is then used to predict the expression ratios of hybrid promoters that are created by swapping out a 50bp long stretch of the original promoter with that of other promoters. Then the hybrid promoters closest to the target gene expression ratios are displayed including the sequence for easy metabolic engineering.

## How to use the toolbox
All settings in the toolbox are available in the panel on the left hand side.
1) First the user chooses a promoter to modify (currently all 1038 metabolic genes included in the Yeast GEM version 8.2 are available).
2) On the main screen the measured expression levels in both conditions as well as the log2 ratio is shown for the chosen gene.
3) The user can now select the target log2 ratio (ethanol / glucose) and how many options should be displayed in a table.
4) The app now created all hybrid promoters and sorts them by the deviation to the target log2 ratio.
5) As a final step the user can now choose which hybrid promoter sequence should be displayed, the swapped part is displayed in bold.

### Using the toolbox online
The Hyena streamlit app is hosted using Heroku and is accessible on https://hyena-toolbox.herokuapp.com/

### Using the toolbox offline
The Hyena toolbox can also be used completly offline following the outlined steps below.

#### Setting up python environment
The used python environemnt to run all scripts was created using pipenv (with Python 3.7) and the following command
pipenv install streamlit pandas numpy scikit-learn xgboost mlxtend matplotlib

Using the pip freeze command a requirement file was created as requirements.txt (pipenv run pip freeze > requirements.txt).
This file can be used to reinstall the environment using pipenv with the following command:
pipenv install 
or using pip with
pip install -r requirements.txt 

#### Start app
Start the streamlit app using:
pipienv run streamlit run Hyena_StreamlitApp.py

## Contributors
- [Christoph Boerlin](https://www.chalmers.se/en/staff/Pages/borlinc.aspx); Chalmers University of Technology, Gothenburg Sweden
