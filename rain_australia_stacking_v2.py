# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:52:31 2025

@author: avina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_ra = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Advanced ML Algorithms/Rain_inaus_feature_engineering_final.csv')
df_ra.columns
df_ra.dtypes
df_ra.shape
df_ra.head()
        
missing_values = df_ra.isnull().sum()
print(missing_values)

df_ra['Date'] = pd.to_datetime(df_ra['Date'])
df_ra['Year'] = df_ra['Date'].dt.year
df_ra['Month'] = df_ra['Date'].dt.month
df_ra['Day'] = df_ra['Date'].dt.day
df_ra['Weekday'] = df_ra['Date'].dt.weekday

df_ra = df_ra.drop('Date', axis=1)

categorical_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
df_encoded = pd.get_dummies(df_ra, columns=categorical_columns, drop_first=True)
df_encoded['RainToday'] = df_encoded['RainToday'].map({'No': 0, 'Yes': 1})
df_encoded['RainTomorrow'] = df_encoded['RainTomorrow'].map({'No': 0, 'Yes': 1})
df_encoded.dtypes

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


X = df_encoded.drop(columns=['RainTomorrow'])
y = df_encoded['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Feature Importance Visualization (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False)[:10]

plt.figure(figsize=(10, 5))
sns.barplot(x=top_features.values, y=top_features.index)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances - Random Forest")
plt.show()

#('knn', KNeighborsClassifier(n_neighbors=5)),

estimators = [
    ('ridge', LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='saga')),
    ('lasso', LogisticRegression(penalty='l1', C=1.0, max_iter=1000, solver='saga')),
    ('rf', RandomForestClassifier(max_depth=20, min_samples_leaf=10, n_estimators=50, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=400, learning_rate=0.3, max_depth=10, random_state=42)), 
    ('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42))
]


###  Voting Classifier (Soft Voting)
voting_clf = VotingClassifier(estimators=estimators, voting='soft')
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, y_pred_voting)
print(f"Voting Classifier Accuracy: {voting_accuracy:.4f}")


###  Stacking Classifier with Logistic Regression as final estimator
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=5000, solver='lbfgs'),
    passthrough=True
)
stacking_clf.fit(X_train, y_train)
y_pred_stacking = stacking_clf.predict(X_test)
stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
print(f"Stacking Classifier Accuracy: {stacking_accuracy:.4f}")

###  Stacking Classifier with Gradient boosting as final estimator
stacking_clf2 = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    passthrough=True
)
stacking_clf2.fit(X_train, y_train)
y_pred_stacking2 = stacking_clf2.predict(X_test)
stacking_accuracy2 = accuracy_score(y_test, y_pred_stacking2)
print(f"Stacking Classifier Accuracy: {stacking_accuracy2:.4f}")

###  Blending
#Blending using Logistic regression
X_train_meta = np.column_stack([
    cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]
    for _, model in estimators
])

meta_model = LogisticRegression()
meta_model.fit(X_train_meta, y_train)

X_test_meta = np.column_stack([model.fit(X_train, y_train).predict_proba(X_test)[:, 1] for _, model in estimators])
y_pred_blending = meta_model.predict(X_test_meta)
blending_accuracy = accuracy_score(y_test, y_pred_blending)
print(f"Blending Classifier Accuracy: {blending_accuracy:.4f}")

#Blending using Gradient Boosting
X_train_meta2 = np.column_stack([
    cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]
    for _, model in estimators
])

meta_model2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
meta_model2.fit(X_train_meta, y_train)

X_test_meta2 = np.column_stack([model.fit(X_train, y_train).predict_proba(X_test)[:, 1] for _, model in estimators])
y_pred_blending2 = meta_model2.predict(X_test_meta)
blending_accuracy2 = accuracy_score(y_test, y_pred_blending2)
print(f"Blending Classifier Accuracy: {blending_accuracy2:.4f}")

# Confusion Matrix Visualization - voting 
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_voting), annot=True, fmt="d", cmap="Blues", xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Voting")
plt.show()

# Confusion Matrix Visualization - stacking with logistic regression
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_stacking), annot=True, fmt="d", cmap="Blues", xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Stacking Model with Logistic Reg")
plt.show()

# Confusion Matrix Visualization - stacking with gradient boosting
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_stacking2), annot=True, fmt="d", cmap="Blues", xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Stacking Model with GB")
plt.show()

# Confusion Matrix Visualization - Blending with logistic regression
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_blending), annot=True, fmt="d", cmap="Blues", xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Blending Model with Logistic Reg")
plt.show()

# Confusion Matrix Visualization - Blending with gradient boosting
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_blending2), annot=True, fmt="d", cmap="Blues", xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Blending Model with GB")
plt.show()

# Final Comparison across voting, stacking & blending
print(f"Voting Classifier Accuracy: {voting_accuracy:.4f}")
print(f"Stacking Classifier Accuracy: {stacking_accuracy:.4f}")
print(f"Stacking Classifier Accuracy: {stacking_accuracy2:.4f}")
print(f"Blending Classifier Accuracy: {blending_accuracy:.4f}")
print(f"Blending Classifier Accuracy: {blending_accuracy2:.4f}")
