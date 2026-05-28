# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:53:34 2026

@author: Alex Gregory
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut


df = pd.read_csv("comsafrica_complete_adjusted.csv", encoding="cp1252")
print(df)

X = df.drop(columns=["flake_id"])
y = df["flake_id"]


X_numeric = X.select_dtypes(include=["int64", "float64"])

X_numeric = X_numeric[['MaximumThickness', 'MaximumWidth', 'MaximumDimension', 'Mass']]
print(X_numeric)


X_train, X_test, y_train, y_test = train_test_split(
    X_numeric,
    y,
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))