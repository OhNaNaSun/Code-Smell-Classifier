from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def train_model(self):
    tree_clf = KNeighborsClassifier(n_neighbors=5)
    tree_clf.fit(self.X_train, self.y_train)

    depths = np.arange(10, 21)

    num_leafs = [1, 5, 10, 20, 50, 100]

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": depths,
        "min_samples_leaf": num_leafs,
    }

    grid_search = GridSearchCV(
        tree_clf, param_grid, cv=10, scoring="accuracy", return_train_score=True
    )

    grid_search.fit(self.X_train, self.y_train)

    best_model = grid_search.best_estimator_

    tree_clf.fit(self.X_train, self.y_train)
    # best_model = grid_search.best_estimator_

    # tree_clf.fit(X_train, y_train)

    train_f1 = f1_score(self.y_train, tree_clf.predict(self.X_train))
    train_accuracy = accuracy_score(self.y_train, tree_clf.predict(self.X_train))
    test_f1 = f1_score(self.y_test, tree_clf.predict(self.X_test))
    test_accuracy = accuracy_score(self.y_test, tree_clf.predict(self.X_test))

    # Training set accuracy

    print("The training accuracy is: ")

    print(
        "DT default hyperparameters: "
        + str(accuracy_score(self.y_train, tree_clf.predict(self.X_train)))
    )

    print(
        "DT optimized hyperparameters: "
        + str(accuracy_score(self.y_train, best_model.predict(self.X_train)))
    )

    # Test set accuracy

    print("\nThe test accuracy is: ")

    print(
        "DT default hyperparameters: "
        + str(accuracy_score(self.y_test, tree_clf.predict(self.X_test)))
    )

    print(
        "DT optimized hyperparameters: "
        + str(accuracy_score(self.y_test, best_model.predict(self.X_test)))
    )
    # print(test_accuracy, test_f1, train_accuracy, train_f1)
    # TODO: grid search
    # return test_accuracy, test_f1, train_accuracy, train_f1
