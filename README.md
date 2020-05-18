![Hyena Logo](/Logo/Hyena_Logo.png =500x)
# Hybrid promoter design using advanced transcription factor binding analytics

This repostiory contains the Hyena toolbox scripts needed to run the Hyena streamlit app as well as all data preprocessing scripts.

- Last update: 2020-05-18

This repository is administered by Christoph Boerlin (https://github.com/ChristophBoerlin), Division of Systems and Synthetic Biology, Department of Biology and Biological Engineering, Chalmers University of Technology

# Introduction

## Using the toolbox online
The Hyena streamlit app is hosted using Heroku and is accessible as Hyena.herokuapp.com

## Using the toolbox offline
The Hyena toolbox can also be used completly offline following the outlined steps below.

### Setting up python environment
The used python environemnt to run all scripts was created using pipenv (with Python 3.7) and the following command
pipenv install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn

Using the pip freeze command a requirement file was created as requirements.txt (pipenv run pip freeze > requirements.txt).
This file can be used to reinstall the environment using pipenv with the following command:
pipenv install
or using pip with
pip install -r requirements.txt 
### Start app
Start the streamlit app using:
pipienv run streamlit run Hyena_StreamlitApp.py


## Contributors
- [Christoph Boerlin](https://www.chalmers.se/en/staff/Pages/borlinc.aspx); Chalmers University of Technology, Gothenburg Sweden
