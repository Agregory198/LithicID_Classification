# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:07:59 2026

@author: Alex Gregory
"""

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

df = pd.read_csv("comsafrica_complete_adjusted.csv", encoding="cp1252")

numeric_cols = ['MaximumThickness', 'MaximumWidth', 'MaximumDimension', 'Mass']

X = df[numeric_cols]
y = df["flake_id"]

# groups = participant IDs
groups = df["analyst_id"]

logo = LeaveOneGroupOut()

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

scores = []

for train_idx, test_idx in logo.split(X, y, groups):

    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)

    scores.append(acc)

    print("Fold Accuracy:", acc)

print("\nMean Accuracy:", sum(scores) / len(scores))