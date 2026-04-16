# FraudDetection-Financial-Model
Machine learning model to predict fraudulent transactions for a financial company using transaction data.

Dataset used in this project: [Google drive Link](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0)
                              | [Kaggle Link](https://www.kaggle.com/datasets/ealaxi/paysim1)

## Project Overview
This project focuses on building a machine learning model to detect fraudulent activities within transaction data.  
The objective is to accurately differentiate between legitimate and fraudulent transactions, minimizing false positives while maximizing fraud detection rates.

This notebook covers:
- Data preprocessing
- Model building using XGB/Random Forest/Decision Tree/Logistic Regression
- Model evaluation using Accuracy, Confusion Matrix, Classification Report, and ROC Curve

## Repository Structure

```text
fraud-detection-model/
├── README.md                             <- The top-level README for developers
├── data/                                 <- Directory for datasets
│   └── raw/                              <- The original, immutable data dump
├── notebooks/                            <- Jupyter notebooks 
│   ├── 01_eda_and_data_overview.ipynb    <- Initial data exploration
│   └── 02_model_training_and_eval.ipynb  <- Model training and evaluation
├── models/                               <- Trained and serialized models
├── reports/                              <- Generated analysis and docs
│   ├── Q&A_Internship_Guideline.pdf      
│   └── figures/                          <- Generated graphics
└── src/                                  <- Source code for this project
```
  ## Project Workflow
- **Data Loading**: Import and inspect the dataset.
- **Exploratory Data Analysis (EDA)**: Identify patterns, outliers, and missing values.
- **Data Preprocessing**: Feature selection and data preparation.
- **Model Building**: Train a XGB Classifier/Random Forest/Decision Tree on the processed data.
- **Model Evaluation**: Analyze performance metrics to validate the model.

## Technology Stack
- Python  Jupyter  Notebook  Pandas  NumPy  Seaborn  Matplotlib  Scikit-learn

## Results
- Achieved more than 95%+ accuracy for all the models with different the fraud:non_fraud ratios.
- Analyzed model performance through confusion matrix, classification report, and ROC-AUC curve.
- Identified key factors affecting fraudulent activity detection.

![xgb model](https://github.com/user-attachments/assets/90d87d73-7cdf-4d0f-8e69-4bbdede391f3)


## Contact
For any queries or collaboration opportunities, feel free to reach out:

- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/jayanta-nath-972a04282/)
