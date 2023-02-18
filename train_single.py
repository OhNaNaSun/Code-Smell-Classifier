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
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


class Main:
    def __init__(self):
        self.data = None

    def _load_data(self):
        # data = arff.loadarff("A1-files/data-class.arff")
        data = arff.loadarff("A1-files/long-method.arff")
        df = pd.DataFrame(data[0])
        # print(df.keys())
        # Count the number of instances of each target variable class
        # class_counts = df["is_feature_envy"].value_counts()

        # Calculate the percentage of instances of each target variable class
        # class_percentages = (class_counts / len(df)) * 100

        # Plot a bar chart to visualize the distribution of the target variable classes
        # class_percentages.plot(kind="bar")
        # plt.xlabel("Target Variable Class")
        # plt.ylabel("Percentage")
        # plt.title("Distribution of Target Variable Classes")
        # plt.show()

        # Compare the distribution of the target variable classes to the distribution of the entire dataset
        # dataset_percentages = [100 / len(df)] * len(class_counts)
        # print("Target variable distribution:", class_percentages.tolist())
        # print("Dataset distribution:", dataset_percentages)
        # data = arff.loadarff("./feature-envy.arff")
        # TODO: normalize data
        df = pd.DataFrame(data[0])
        X = df.iloc[:, :-1].values
        # print(X)
        Y_data = df.iloc[:, -1].values

        encoder = preprocessing.LabelEncoder()

        y = encoder.fit_transform(Y_data)

        # tree_clf.fit(X, y)

        X_copy = df.iloc[:, :-1].copy()

        imputer = SimpleImputer(strategy="most_frequent")

        new_X = imputer.fit_transform(X_copy)

        new_X_df = pd.DataFrame(new_X, columns=X_copy.columns, index=X_copy.index)
        X_train, X_test, y_train, y_test = train_test_split(
            new_X, y, test_size=0.15, random_state=42
        )
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.15, random_state=42
        # )

        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        # self.X_train = X_train
        # self.X_test = X_test
        self.X_test = X_test_norm
        self.X_train = X_train_norm
        self.y_train = y_train
        self.y_test = y_test

        # return X_train, X_test, y_train, y_test

    def train_model(self):
        # tree_clf = DecisionTreeClassifier()
        tree_clf = KNeighborsClassifier(n_neighbors=5)
        tree_clf.fit(self.X_train, self.y_train)

        depths = np.arange(10, 21)

        num_leafs = [1, 5, 10, 20, 50, 100]
        param_grid = {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30],
            "p": [1, 2],
        }

        # param_grid = {
        #     "criterion": ["gini", "entropy"],
        #     "max_depth": depths,
        #     "min_samples_leaf": num_leafs,
        # }

        grid_search = GridSearchCV(
            tree_clf, param_grid, cv=10, scoring="accuracy", return_train_score=True
        )

        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_

        tree_clf.fit(self.X_train, self.y_train)
        # best_model = grid_search.best_estimator_

        # tree_clf.fit(X_train, y_train)

        train_f1 = f1_score(self.y_train, best_model.predict(self.X_train))
        train_accuracy = accuracy_score(self.y_train, best_model.predict(self.X_train))
        test_f1 = f1_score(self.y_test, best_model.predict(self.X_test))
        test_accuracy = accuracy_score(self.y_test, best_model.predict(self.X_test))
        df = pd.DataFrame(
            {
                "feature-envy": ["Training_set", "Test_set"],
                "accuracy": [
                    train_accuracy,
                    test_accuracy,
                ],
                "f1_score": [train_f1, test_f1],
            }
        )
        print(df)
        # Training set accuracy

        # print("The training accuracy is: ")

        # print(
        #     "DT default hyperparameters: "
        #     + str(accuracy_score(self.y_train, tree_clf.predict(self.X_train)))
        # )

        # print(
        #     "DT optimized hyperparameters: "
        #     + str(accuracy_score(self.y_train, best_model.predict(self.X_train)))
        # )

        # Test set accuracy

        # print("\nThe test accuracy is: ")

        # print(
        #     "DT default hyperparameters: "
        #     + str(accuracy_score(self.y_test, tree_clf.predict(self.X_test)))
        # )

        # print(
        #     "DT optimized hyperparameters: "
        #     + str(accuracy_score(self.y_test, best_model.predict(self.X_test)))
        # )
        # print(test_accuracy, test_f1, train_accuracy, train_f1)
        # TODO: grid search
        # return test_accuracy, test_f1, train_accuracy, train_f1


if __name__ == "__main__":
    main = Main()
    main._load_data()
    main.train_model()
