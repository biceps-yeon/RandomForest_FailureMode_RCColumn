import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
from sklearn import tree

def decision_tree(X_train, y_train):

    # 모델 학습
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    """
    # 결정트리 규칙을 시각화
    plt.figure( figsize=(20,15) )
    tree.plot_tree(model, 
               class_names=y.unique(),
               feature_names=x.columns,
               impurity=True, filled=True,
               rounded=True)
    plt.show()
    """

    """
    import seaborn as sns
    import numpy as np
    #matplotlib inline

    # feature별 importance 매핑
    for name, value in zip(x.columns , model.feature_importances_):
        print('{} : {:.3f}'.format(name, value))

    # feature importance를 column 별로 시각화 하기 
    sns.barplot(x=model.feature_importances_ , y=x.columns)
    plt.show()
    """

    return model
