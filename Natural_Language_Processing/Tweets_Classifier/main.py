import numpy as np
from sklearn.ensemble import RandomForestClassifier
import utils

if __name__ == '__main__':
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)


    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    utils.AssessModel(clf, X_test, y_test)

