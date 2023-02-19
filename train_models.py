from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


def process_data(file_path):
    data = arff.loadarff("A1-files/" + file_path)
    df = pd.DataFrame(data[0])
    X = df.iloc[:, :-1].values
    Y_data = df.iloc[:, -1].values
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(Y_data)
    X_copy = df.iloc[:, :-1].copy()
    imputer = SimpleImputer(strategy="most_frequent")
    new_X = imputer.fit_transform(X_copy)
    new_X_df = pd.DataFrame(new_X, columns=X_copy.columns, index=X_copy.index)
    X_train, X_test, y_train, y_test = train_test_split(
        new_X, y, test_size=0.15, random_state=42
    )
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    # X_train = X_train
    # X_test = X_test
    # normalize data
    X_train = X_train_norm
    X_test = X_test_norm
    y_train = y_train
    y_test = y_test

    return X_train, X_test, y_train, y_test


def train_model(smell_file_path, classifier, param_grid):
    X_train, X_test, y_train, y_test = process_data(smell_file_path)
    tree_clf = classifier
    tree_clf.fit(X_train, y_train)
    grid_search = GridSearchCV(
        tree_clf, param_grid, cv=10, scoring="accuracy", return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    tree_clf.fit(X_train, y_train)

    train_f1 = f1_score(y_train, best_model.predict(X_train))
    train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
    test_f1 = f1_score(y_test, best_model.predict(X_test))
    test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
    return test_accuracy, test_f1, train_accuracy, train_f1
