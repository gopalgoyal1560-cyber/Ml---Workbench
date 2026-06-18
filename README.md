# ML Workbench
## Train, Evaluate, Save, and Reuse Machine Learning Models Without Writing Code
ML Workbench is a Streamlit-based machine learning platform that simplifies the end-to-end workflow of building predictive models from tabular datasets.
The application allows users to upload datasets, select features and targets, train multiple machine learning algorithms, evaluate model performance, save trained models,
and perform future predictions using either uploaded files or individual inputs.
This project was designed and developed as a complete machine learning workbench that combines data loading, preprocessing, model training, evaluation, model persistence,
and inference into a single interactive web application.
## Key Features

### Dataset Loading
* Upload datasets from local storage
* Load datasets directly from URLs
* Support for CSV, Excel (XLSX), and JSON files
* Instant dataset preview

### Automated Data Preprocessing
* Missing value handling
* Numerical data imputation
* Categorical data imputation
* One-Hot Encoding
* Ordinal Encoding
* Scikit-Learn preprocessing pipelines

### Machine Learning Models
**Regression**
* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor
**Classification**
* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

### Model Evaluation
 **Regression Metrics**
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score
  **Classification Metrics**
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix Visualization

### Model Management
* Save trained models using Joblib
* Store preprocessing pipeline with model artifacts
* Preserve feature metadata for future predictions

### Prediction System
**Batch Prediction**
* Upload new datasets
* Generate predictions for entire files
* Export prediction results
**Single Record Prediction**
* Dynamic input generation based on saved model features
* Real-time prediction for individual samples

## Technology Stack
**Frontend - Streamlit**
**Machine Learning - Scikit-Learn,XGBoost**
**Data Processing - Pandas,NumPy**
**Visualization - Matplotlib**
**Model Persistence - Joblib**

## Project Architecture
Dataset Input
↓
Automated Preprocessing
↓
Feature Encoding & Imputation
↓
Model Training
↓
Performance Evaluation
↓
Model Saving
↓
Prediction Interface

## How to Run
**Installation**
pip install -r requirements.txt
**Run Application**
streamlit run app.py

## Learning Outcomes
Through this project, I gained practical experience in:
* Machine Learning Pipelines
* Data Preprocessing
* Model Evaluation
* Streamlit Application Development
* User Interface Design for ML Applications

## Future Scope
The architecture has been designed to support future expansion into advanced model management, enhanced visual analytics, additional algorithms, and broader deployment capabilities.
I will try to add cross validation, hyper parameters tunning and inbuilt pretrained ml models, and better layout

## Author
**Gopal**
Computer Science Diploma Student
Machine Learning and Software Development Enthusiast
