import sklearn
from sklearn.model_selection import train_test_split

def data_split(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)
    return X_train, X_test, y_train, y_test