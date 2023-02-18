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
import signal
import sys


def signal_handler(signal, frame):
    print("Program terminated by user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Main:
    def __init__(self):
        self.data = None
        self.model_dict = {
            "decision_tree": {"classifier": DecisionTreeClassifier(), "results": {}},
            # TODO
            "random_forest": {"classifier": RandomForestClassifier(), "results": {}},
            # "naive_bayes": {"classifier": GaussianNB(), "results": {}},
            "k-neighbors": {
                "classifier": KNeighborsClassifier(n_neighbors=5),
                "results": {},
            },
        }
        self.smells = {
            "feature_envy": "feature-envy.arff",
            # "data_class": "data-class.arff",
            # "god_class": "god-class.arff",
            # "long_method": "long-method.arff",
        }
        self.selected_model = None
        self.selected_smell_types = None

    def _load_data(self, file_path):
        # data = arff.loadarff("A1-files/" + file_path)
        data = arff.loadarff("./feature-envy.arff")
        # TODO: normalize data
        df = pd.DataFrame(data[0])
        X = df.iloc[:, :-1].values
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
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test

    def train_model(self, model_name):
        # tree_clf = self.model_dict[model_name]["classifier"]
        tree_clf = DecisionTreeClassifier()
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

        train_f1 = f1_score(self.y_train, best_model.predict(self.X_train))
        train_accuracy = accuracy_score(self.y_train, best_model.predict(self.X_train))
        test_f1 = f1_score(self.y_test, best_model.predict(self.X_test))
        test_accuracy = accuracy_score(self.y_test, best_model.predict(self.X_test))
        # TODO: grid search
        return test_accuracy, test_f1, train_accuracy, train_f1

    def process_trainings(self):
        model_names = self.model_dict.keys()
        for model in model_names:
            for smell_type in list(self.smells.keys()):
                self._load_data(self.smells[smell_type])
                (
                    accuracy,
                    f1_score_value,
                    train_accuracy,
                    train_f1_score_value,
                ) = self.train_model(model)
                self.model_dict[model]["results"][smell_type] = {
                    "accuracy": accuracy,
                    "f1_score_value": f1_score_value,
                    "train_accuracy": train_accuracy,
                    "train_f1_score_value": train_f1_score_value,
                }

    def print_compared_result(self):
        model = self.selected_model
        for smell_type in self.selected_smell_types:
            chosen_model = self.model_dict[model]["results"][smell_type]
            df = pd.DataFrame(
                {
                    smell_type: ["Training_set", "Test_set"],
                    "accuracy": [
                        chosen_model["train_accuracy"],
                        chosen_model["accuracy"],
                    ],
                    "f1_score": [
                        chosen_model["train_f1_score_value"],
                        chosen_model["f1_score_value"],
                    ],
                }
            )
            print(df)
            print("------------------")

    def print_selected_results(self):
        model = self.selected_model
        df_results = []

        for smell_type in self.selected_smell_types:
            chosen_model = self.model_dict[model]["results"][smell_type]
            printed_result = pd.DataFrame(
                [smell_type, chosen_model["accuracy"], chosen_model["f1_score_value"]]
            ).transpose()
            printed_result.columns = ["Smell Type", "accuracy", "f1_score"]
            df_results.append(printed_result)
        printed_result = pd.concat(df_results, axis=0)
        print(printed_result)

    def select_smells(self, callback):
        options = list(self.smells.keys())
        selection = [-1]
        while not all(1 <= x <= len(options) for x in selection):
            print("Please select one or more options (comma-separated):")
            for i, option in enumerate(options):
                print(f"{i + 1}. {option}")

            selection = input("Enter the numbers of your selections: ")

            try:
                selection = [int(x.strip()) for x in selection.split(",")]
                if not all(1 <= x <= len(options) for x in selection):
                    print("Invalid selection. Please try again")
                    selection = [-1]
            except ValueError:
                print("Invalid input. Please try again.")
                selection = [-1]

        selected_options = [options[x - 1] for x in selection]
        self.selected_smell_types = selected_options
        print(f'You selected: {", ".join(selected_options)}')
        callback()

    def compare_or_retrain(self):
        # print("Compare or Retrain")
        options = ["Compare", "Retrain"]
        selection = -1
        while not 1 <= selection <= len(options):
            for i, option in enumerate(options):
                print(f"{i + 1}. {option}")

            selection = input("Enter the number of your selections: ")
            try:
                selection = int(selection)
                if not 1 <= selection <= len(options):
                    selection = -1
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
                selection = -1
        print(f'You selected "{options[selection - 1]}"')
        if selection == 1:
            # Compare Test and Training
            self.print_compared_result()
        else:
            # Retrain
            self.enter_model_type(self.start_train)

    def enter_model_type(self, callback):
        options = list(self.model_dict.keys())
        selection = -1
        while not 1 <= selection <= len(options):
            for i, option in enumerate(options):
                print(f"{i + 1}. {option}")
            selection = input("Enter the number of your selection: ")
            try:
                selection = int(selection)
                if not 1 <= selection <= len(options):
                    print("Invalid option. Please try again.")
            except ValueError:
                selection = -1
                print("Invalid option. Please try again.")
        print(f'You selected "{options[selection - 1]}"')
        self.selected_model = options[selection - 1]
        callback()

    def start_train(self):
        self.select_smells(self.print_selected_results)
        self.compare_or_retrain()

    def run_application(self):
        self.process_trainings()
        self.enter_model_type(self.start_train)


if __name__ == "__main__":
    main = Main()
    main.run_application()
