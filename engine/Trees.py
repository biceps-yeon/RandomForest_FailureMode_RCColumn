import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
from sklearn import tree

import seaborn as sns
import numpy as np

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def decision_tree(X_train, y_train):

    # 모델 학습
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    """
    # 결정트리 규칙을 시각화
    plt.figure( figsize=(20,15) )
    tree.plot_tree(model, 
               class_names=y_train.unique(),
               feature_names=X_train.columns,
               impurity=True, filled=True,
               rounded=True)
    plt.show()
    """

    """
    # feature별 importance 매핑
    for name, value in zip(X_train.columns , model.feature_importances_):
        print('{} : {:.3f}'.format(name, value))

    # feature importance를 column 별로 시각화 하기 
    sns.barplot(x=model.feature_importances_ , y=X_train.columns)
    plt.show()
    """

    return model

def random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=45, max_depth=6, max_features='sqrt', bootstrap=True, max_samples=0.8, random_state=42)
    model.fit(X_train, y_train)
    return model

def grad_boost_DT(x_train, y_train):
    model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(x_train, y_train)
    return model

def xg_boost(x_train, y_train):
    # 모델 학습
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(x_train, y_train)   # ✅ DMatrix 대신 DataFrame/array 직접 사용

    return model
    """
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(train_set, y_train)
    return model, map_idx_to_name, map_name_to_idx"""