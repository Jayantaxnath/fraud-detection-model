import pandas as pd
import numpy as np
import math

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
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, auc

chunksize = 200_000
dfs = []

for chunk in pd.read_csv("../data/raw/Fraud.csv", chunksize=chunksize):
    dfs.append(chunk)

df = pd.concat(dfs)

df

df.info()

df['isFraud'].value_counts()

df['isFraud'].hist(log=True, color='skyblue')

# 8213/6354407 = 0.00129

df.isna().sum()

# Storing the categorical and numerical columns

cat_col = df.select_dtypes(include='object').columns
cat_col = [col for col in cat_col]
num_col = df.select_dtypes(exclude='object').columns
num_col = [num for num in num_col]

num_col, cat_col

# Categorical column's unique value count

for col in cat_col:
    print(df[col].unique(), len(df[col].unique()))

df[num_col].describe()

df2 = df.drop(['nameOrig', 'nameDest'], axis=1)
df2.head()

df2['balanceDiffOrig'] = df2['oldbalanceOrg'] - df2['newbalanceOrig']
df2['balanceDiffDest'] = df2['oldbalanceDest'] - df2['newbalanceDest']
df2['errorOrig'] = df2['oldbalanceOrg'] - df2['amount'] - df2['newbalanceOrig']

df2[df2['errorOrig'] > 0]['isFraud'].value_counts()

df2[df2['errorOrig'] == 0]['isFraud'].value_counts()

df2[df2['errorOrig'] < 0]['isFraud'].value_counts()

df2

df2.groupby('type')['isFraud'].value_counts()

df2.groupby(['type', 'isFraud']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 4))
sns.countplot(x='type', hue='isFraud', data=df2)
plt.yscale('log', base = 10)
plt.title('Fraud Count by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#df2[(df2['type'] == 'TRANSFER') | (df2['type'] == 'CASH_OUT')]
fraud_df = df2[df2['type'].isin(['TRANSFER', 'CASH_OUT'])]

fraud_df

fraud_df['isFraud'].value_counts()

fraud_df[fraud_df['isFraud'] == 1].sample(20)

fraud_df[(fraud_df['isFraud'] == 1) & (fraud_df['newbalanceOrig'] == 0)]

df2

sns.scatterplot(data=df2.sample(10000), x='amount', y='isFraud', hue='type')

'''plt.figure(figsize=(12, 10))
sns.scatterplot(data=df2,  # sample for speed
                x='amount', 
                y='newbalanceOrig', 
                hue='isFraud', 
                alpha=0.5, 
                palette={0: 'green', 1: 'red'})
plt.title("Transaction Amount vs New Balance (Orig) — Fraud Highlighted")
plt.xlabel("Amount")
plt.ylabel("New Balance of Origin Account")
plt.yscale('log')  # to deal with large spread
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()'''

Q1 = df2['amount'].quantile(0.25)
Q3 = df2['amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df2[(df2['amount'] < lower_bound) | (df2['amount'] > upper_bound)]

print(outliers['isFraud'].value_counts())
outliers

from scipy.stats import zscore

df2['z_amount'] = zscore(df2['amount'])
outliers = df2[df2['z_amount'].abs() > 3]
print(outliers['isFraud'].value_counts())
outliers

df3 = df2[df2['z_amount'].abs() <= 3]

print(df3['isFraud'].value_counts())
df3

cols = ['amount',	'oldbalanceOrg',	'newbalanceOrig',	'oldbalanceDest']

# Plotting
cols = ['amount',	'oldbalanceOrg',	'newbalanceOrig',	'oldbalanceDest']
plt.figure(figsize=(12, 6))
for i, col in enumerate(cols):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(data=df3, y=col, color='lightgreen')
    plt.title(col)
    plt.tight_layout()

plt.suptitle("Outlier Detection", fontsize=16, y=1.02)
plt.show()

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

cols = ['balanceDiffOrig',	'balanceDiffDest',	'errorOrig']

plt.figure(figsize=(10, 6))
for i, col in enumerate(cols):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df3[col], kde=True, log_scale=10)
    plt.title(col)
    plt.tight_layout()

plt.suptitle("Distributions", fontsize=16, y=1.02)
plt.show()

outliers =  df3[(
    ((df['oldbalanceOrg'] == 0) & (df['amount'] > 0)) |
    (df['newbalanceOrig'] == df['oldbalanceOrg']) |
    ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] > 1e7))
)]

print(outliers['isFraud'].value_counts())
outliers

df3[df['amount'] > 1000000]

dummies = pd.get_dummies(df3['type'])
df3 = pd.concat([df3, dummies], axis=1)

df3.tail()

df4 = df3.drop(['type', 'z_amount'], axis=1)

df4 = df4.reset_index(drop=True)
print(df4['isFraud'].value_counts())
df4

X = df4.drop('isFraud', axis=1)
y = df4['isFraud']
X.shape, y.shape

X

from sklearn.model_selection import train_test_split

# Sample 1 million rows from the full dataset with stratification (to preserve fraud ratio)
X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=1_000_000, stratify=y, random_state=42)

# Now split this 1M into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42)

# Quick sanity check
print(f"Train size X: {X_train.shape}, Test size X: {X_test.shape}")
print(f"Train frauds y: {y_train.shape}, Test frauds y: {y_test.shape}")

print(y_train.value_counts())
print(y_test.value_counts())

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

y_res.value_counts()

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, auc

lr_model = LogisticRegression(C=0.01, solver = 'liblinear', class_weight='balanced')
lr_model.fit(X_train, y_train)
print(lr_model.score(X_res, y_res))
cm = confusion_matrix(y_test, lr_model.predict(X_test))
print(cm)

y_pred = lr_model.predict(X_test)
y_proba = lr_model.predict_proba(X_test)[:, 1]

# Evaluate metrics for class 1 (fraud)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_proba)  # use probabilities for AUC!

# Print results
print(f"Recall (Fraud):     {recall:.4f}")
print(f"Precision (Fraud):  {precision:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"AUC-ROC Score:      {auc_roc:.4f}")

y_scores = lr_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
plt.show()

y_scores = lr_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# Plotting threshold vs precision/recall
import matplotlib.pyplot as plt
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.legend()
plt.grid(True)
plt.title('Precision vs Recall vs Threshold')
plt.show()

rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
print(rf_model.score(X_res, y_res))
cm = confusion_matrix(y_test, rf_model.predict(X_test))
print(cm)

y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate metrics for class 1 (fraud)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_proba)  # using probabilities for AUC!

print(f"Recall (Fraud):     {recall:.4f}")
print(f"Precision (Fraud):  {precision:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"AUC-ROC Score:      {auc_roc:.4f}")

# 
feature_importances = rf_model.feature_importances_
feature_names = X_res.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

plt.figure(figsize=(12, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=90)
plt.title('Random Forest Feature Importances')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

y_scores = rf_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
plt.show()

dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)
print(dt_model.score(X_res, y_res))
cm = confusion_matrix(y_test, dt_model.predict(X_test))
print(cm)

y_pred = dt_model.predict(X_test)
y_proba = dt_model.predict_proba(X_test)[:, 1]

# Evaluate metrics for class 1 (fraud)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_proba)  # use probabilities for AUC!

# Print results
print(f"Recall (Fraud):     {recall:.4f}")
print(f"Precision (Fraud):  {precision:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"AUC-ROC Score:      {auc_roc:.4f}")



