# DS-340W-Group-Project
## Overview
This repository contains code for predicting Chronic Kidney Disease (CKD) using various machine learning models and state-of-the-art deep learning approaches. The project includes a comparison of traditional machine learning techniques, such as Logistic Regression and Random Forest, with advanced deep learning models, TabNet and TabTransformer, specifically designed for tabular data. These models not only improve prediction accuracy but also enhance interpretability, making them suitable for critical applications in healthcare.

## Repository Structure
Dataset (kidney_disease.csv): A dataset containing patient records with features. This dataset is used for training and evaluating all models.

kidney_disease.py: Contains the implementation of traditional machine learning models(Parent Paper Code)
tabnet.py: Implements the TabNet model
tabtransformer.py: Implements the TabTransformer model for CKD prediction


## Usage
Run Traditional Machine Learning Models

Execute the following script to train and evaluate traditional ML models:

python kidney_disease.py

## Run TabNet Model
Train and evaluate the TabNet model:

python tabnet.py

## Run TabTransformer Model
Train and evaluate the TabTransformer model:

python tabtransformer.py

## Analyze Feature Importance
Both tabnet.py and tabtransformer.py scripts include SHAP-based feature importance analysis. The results will be displayed as visualizations highlighting the contribution of features to the models' predictions.

## Performance Metrics
The project compares various models on metrics such as accuracy, precision, recall, and F1-score.

## Contributors
Slava Hlushko (Penn State University, Computational Data Science): vqh5091@psu.edu

Laurence Castillo (Penn State University, Computational Data Science): llc5360@psu.edu

Ajinkya Kondaskar (Penn State University, Computational Data Science): aak5683@psu.edu


