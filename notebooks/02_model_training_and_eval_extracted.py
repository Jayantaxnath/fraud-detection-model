import pandas as pd
import numpy as np
import math
from scipy.stats import zscore

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score
print('done')

chunksize = 100_000
dfs = []

for chunk in pd.read_csv("../data/raw/Fraud.csv", chunksize=chunksize):
    dfs.append(chunk)

df = pd.concat(dfs)

df['isFraud'].value_counts()

# 8213/6354407 = 0.00129

df.isna().sum()

# Storing the categorical and numerical columns

cat_col = df.select_dtypes(include='object').columns
cat_col = [col for col in cat_col]
num_col = df.select_dtypes(exclude='object').columns
num_col = [num for num in num_col]

# Categorical column's unique value count

for col in cat_col:
    print(df[col].unique(), len(df[col].unique()))

df2 = df.drop(['nameOrig', 'nameDest'], axis=1)
df2.head()

outliers = df2[(
        ((df2['oldbalanceOrg'] == 0) & (df2['amount'] > 0)) |
        (df2['newbalanceOrig'] == df2['oldbalanceOrg']) |
        ((df2['oldbalanceDest'] == 0) & (df2['newbalanceDest'] > 1e7))
    )]

print(outliers['isFraud'].value_counts())
outliers

# outlier remove
df3 =  df2[~(
            ((df['oldbalanceOrg'] == 0) & (df['amount'] > 0)) |
            (df['newbalanceOrig'] == df['oldbalanceOrg']) |
            ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] > 1e7))
        )]

df3.reset_index(drop=True)
df3.head()

plt.figure(figsize=(10, 4))
sns.countplot(x='type', hue='isFraud', data=df3)
plt.yscale('log', base = 10)
plt.title('Fraud Count by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#df2[(df2['type'] == 'TRANSFER') | (df2['type'] == 'CASH_OUT')]
fraud_df = df3[df3['type'].isin(['TRANSFER', 'CASH_OUT'])]
fraud_df['isFraud'].value_counts()

fraud_df[fraud_df['isFraud'] == 1].sample(10)

df3['balanceDiffOrig'] = df3['oldbalanceOrg'] - df3['newbalanceOrig']
df3['balanceDiffDest'] = df3['oldbalanceDest'] - df3['newbalanceDest']
#df2['errorOrig'] = df2['oldbalanceOrg'] - df2['amount'] - df2['newbalanceOrig'] skipping due to data leakage 

df3.groupby('type')['isFraud'].value_counts()

# Plotting
cols = ['amount',	'oldbalanceOrg',	'newbalanceOrig',	'oldbalanceDest']
plt.figure(figsize=(10, 6))
for i, col in enumerate(cols):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df3['amount'], kde=True, log_scale=10)
    plt.title(col)
    plt.tight_layout()

plt.suptitle("Distributions", fontsize=16, y=1.02)
plt.show()

cols = ['balanceDiffOrig',	'balanceDiffDest']

plt.figure(figsize=(10, 4))
for i, col in enumerate(cols):
    plt.subplot(1, 2, i + 1)
    sns.histplot(df3[col], kde=True, log_scale=10)
    plt.title(col)
    plt.tight_layout()

plt.suptitle("Distributions", fontsize=16, y=1.02)
plt.show()

dummies = pd.get_dummies(df3['type'])
df3 = pd.concat([df3, dummies], axis=1)

df3.reset_index(drop=True, inplace=True)
print(df3.shape)
df3.tail()

df4 = df3.drop(['type'], axis=1)

df4 = df4.reset_index(drop=True)
print(df4['isFraud'].value_counts())
df4.sample(5)

corr = df4.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True)

df5 = df4.drop(['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'], axis=1)

corr = df5.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True)

print(df5['isFraud'].value_counts())

from sklearn.utils import resample

# Separate majority and minority
df_majority = df5[df5.isFraud == 0]
df_minority = df5[df5.isFraud == 1]

# Downsample majority class
df_majority_downsampled = resample(
    df_majority, 
    replace=False, 
    n_samples=8156,  # 8156/16312/24468/81560
    random_state=42
)

# Combine
df_balanced = pd.concat([df_majority_downsampled, df_minority], axis=0)

df_balanced.reset_index(drop=True, inplace=True)
df_balanced.head()

df_type = df_balanced[['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER','isFraud']]
from sklearn.preprocessing import  StandardScaler
sc = StandardScaler()
scaled = sc.fit_transform(df_balanced[['amount','balanceDiffOrig','balanceDiffDest']])
scaled_df = pd.DataFrame(scaled, columns=['amount', 'balanceDiffOrig', 'balanceDiffDest'])
df_final = pd.concat([scaled_df, df_type], axis=1)
df_final

X = df_final.drop('isFraud', axis=1)
y = df_final['isFraud']
X.shape, y.shape

# Now split this 1M into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)

# Quick sanity check
print(f"Train size X: {X_train.shape}, Test size X: {X_test.shape}")
print(f"Train frauds y: {y_train.shape}, Test frauds y: {y_test.shape}")

print(y_train.value_counts())
print(y_test.value_counts())

def model_performance(model):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate metrics for class 1 (fraud)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)

    print(f"Recall (Fraud):     {round(recall*100, 2)}%")
    print(f"Precision (Fraud):  {round(precision*100, 2)}%%")
    print(f"F1 Score:           {round(f1*100, 2)}%")
    print(f"AUC-ROC Score:      {round(auc_roc*100, 2)}%")

def classification_report_detailed(model):
    y_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Training accuracy of Voting Classifier is : {train_acc}")
    print(f"Test accuracy of Voting Classifier is : {test_acc}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def precision_recall_auc_curve(model):  # Plot 1: Precision-Recall Curve
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    # Create subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Precision-Recall Curve
    axes[0].plot(recall, precision, color='b')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision-Recall Curve (AUC = {pr_auc:.2f})')

    # Plot 2: Precision vs Recall vs Threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    axes[1].plot(thresholds, precisions[:-1], label='Precision')
    axes[1].plot(thresholds, recalls[:-1], label='Recall')
    axes[1].set_xlabel('Threshold')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Precision vs Recall vs Threshold')

    # Show the plots
    plt.tight_layout()
    plt.show()

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Compute scale_pos_weight for imbalanced classes
scale_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
scale_weight

from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=8,
)

xgb.fit(X_train, y_train)
print('Model score:',xgb.score(X_test, y_test))
cm = confusion_matrix(y_test, xgb.predict(X_test))
classification_report_detailed(xgb)

xgb.predict([[-0.43029183, -0.38538989,  0.26725852, 0, 1,
        0, 0 , 0]]) # expected 0

import pickle
# Save the model to a .pkl file
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(xgb, file)

precision_recall_auc_curve(xgb)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
print('Model score:',rf_model.score(X_test, y_test))
cm = confusion_matrix(y_test, rf_model.predict(X_test))
classification_report_detailed(rf_model)
#model_performance(rf_model)

import pickle
# Save the model to a .pkl file
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

precision_recall_auc_curve(rf_model)

dt_model = DecisionTreeClassifier(max_depth=5, class_weight={0: 1, 1: 10}) # {0: 1, 1: 10} : performing very well well than 'balanced'
dt_model.fit(X_train, y_train)
print('Model score:',dt_model.score(X_test, y_test))
cm = confusion_matrix(y_test, dt_model.predict(X_test))
classification_report_detailed(dt_model)

import pickle
# Save the model to a .pkl file
with open('dt_model.pkl', 'wb') as file:
    pickle.dump(dt_model, file)

precision_recall_auc_curve(dt_model)

lr_model = LogisticRegression(C=10, solver = 'liblinear', class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)
print('Model score:',lr_model.score(X_train, y_train))
cm = confusion_matrix(y_test, lr_model.predict(X_test))
classification_report_detailed(lr_model)

precision_recall_auc_curve(lr_model)



