# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:10:13 2026

@author: grego
"""

import pandas as pd
import itertools

df = pd.read_csv("comsafrica_complete_adjusted.csv", encoding="cp1252")

numeric_cols = ['MaximumThickness', 'MaximumWidth', 'MaximumDimension', 'Mass']

pairs = []

# compare every pair
for i, j in itertools.combinations(df.index, 2):

    row1 = df.loc[i]
    row2 = df.loc[j]

    # absolute differences
    diff_vector = abs(row1[numeric_cols] - row2[numeric_cols])

    # same flake?
    same_flake = int(row1["flake_id"] == row2["flake_id"])

    pair_data = diff_vector.to_dict()
    pair_data["same_flake"] = same_flake

    pairs.append(pair_data)

pair_df = pd.DataFrame(pairs)

print(pair_df.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = pair_df.drop(columns=["same_flake"])
y = pair_df["same_flake"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(classification_report(y_test, preds))