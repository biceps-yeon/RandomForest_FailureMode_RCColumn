import pandas as pd
import numpy as np
from engine.DataLoader import load_data
from engine.Trees import decision_tree, random_forest, grad_boost_DT, xg_boost
from engine.SplitData import data_split

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
from sklearn import tree

import pandas as pd
import xgboost as xgb
from engine.Prcder_Dt import preprocessor

x, y = load_data()
X_train, X_test, y_train, y_test = data_split(x, y)

#model = decision_tree(X_train, y_train)

#model = random_forest(X_train, y_train)

model = grad_boost_DT(X_train, y_train)

"""X_train, y_train, map_idx_to_name, map_name_to_idx = preprocessor(X_train, y_train)
X_test, y_test, map_idx_to_name, map_name_to_idx = preprocessor(X_test, y_test)
#print(map_idx_to_name)
model = xg_boost(X_train, y_train)"""

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

###################xgboost가 왜 GBDT보다 성능이 낮지????????################
