import pandas as pd
from engine.DataLoader import load_data
from engine.Trees import decision_tree
from engine.SplitData import data_split

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
from sklearn import tree

x, y = load_data()
X_train, X_test, y_train, y_test = data_split(x, y)
model = decision_tree(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

