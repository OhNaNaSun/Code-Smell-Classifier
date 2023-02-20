import os
import sys
import json
import signal
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from train_models import train_model
from utils import user_input_selection
from sklearn.naive_bayes import GaussianNB


def signal_handler(signal, frame):
    print("Program terminated by user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Main:
    def __init__(self):
        self.model_dict = [
            {
                "key": "decision_tree",
                "name": "Decision Tree",
                "classifier": DecisionTreeClassifier(),
                "param_grid": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": np.arange(10, 21),
                    "min_samples_leaf": [1, 5, 10, 20, 50, 100],
                },
                "results": {},
            },
            {
                "key": "naive_bayes",
                "name": "Naïve Bayes",
                "classifier": GaussianNB(),
                "param_grid": {"var_smoothing": np.logspace(0, -9, num=100)},
                "results": {},
            },
            {
                "key": "k-neighbors",
                "name": "k-Nearest Neighbors",
                "classifier": KNeighborsClassifier(n_neighbors=5),
                "results": {},
                "param_grid": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["ball_tree", "kd_tree", "brute"],
                    "leaf_size": [10, 20, 30],
                    "p": [1, 2],
                },
            },
        ]

        self.smell_files = [
            {
                "key": "feature_envy",
                "name": "Feature Envy",
                "file_path": "feature-envy.arff",
            },
            {
                "key": "data_class",
                "name": "Data Class",
                "file_path": "data-class.arff",
            },
            {"key": "god_class", "name": "God Class", "file_path": "god-class.arff"},
            {
                "key": "long_method",
                "name": "Long Method",
                "file_path": "long-method.arff",
            },
        ]
        self.selected_model = None
        self.selected_smell_types = None

    def process_trainings(self):
        for model in self.model_dict:
            for smell_type in self.smell_files:
                filename = "results/" + model["key"] + "-" + smell_type["key"] + ".json"
                # Save or Read data
                if not os.path.isfile(filename) or not os.path.getsize(filename) > 0:
                    with open(filename, "w") as f:
                        (
                            accuracy,
                            f1_score_value,
                            train_accuracy,
                            train_f1_score_value,
                        ) = train_model(
                            smell_type["file_path"],
                            model["classifier"],
                            model["param_grid"],
                        )
                        data = {
                            "accuracy": accuracy,
                            "f1_score_value": f1_score_value,
                            "train_accuracy": train_accuracy,
                            "train_f1_score_value": train_f1_score_value,
                        }
                        json.dump(data, f)
                        model["results"][smell_type["key"]] = data
                else:
                    with open(filename, "r") as f:
                        data = json.load(f)
                        model["results"][smell_type["key"]] = data

    def print_compared_result(self):
        print(
            "The comparison between the test set and training set for each of your selected smells:\n"
        )
        model = list(
            filter(lambda x: x["name"] == self.selected_model, self.model_dict)
        )[0]
        for selected_smell in self.selected_smell_types:
            smell = list(
                filter(lambda x: x["name"] == selected_smell, self.smell_files)
            )[0]
            smell_type = smell["key"]
            chosen_model = model["results"][smell_type]
            df = pd.DataFrame(
                {
                    smell["name"]: ["Training Set", "Test Set"],
                    "Accuracy": [
                        chosen_model["train_accuracy"],
                        chosen_model["accuracy"],
                    ],
                    "F1-score": [
                        chosen_model["train_f1_score_value"],
                        chosen_model["f1_score_value"],
                    ],
                }
            )
            df.index += 1
            print(df)
            print("----------------------------------------")

    def print_selected_results(self):
        model = list(
            filter(lambda x: x["name"] == self.selected_model, self.model_dict)
        )[0]
        df_results = []

        for selected_smell in self.selected_smell_types:
            smell = list(
                filter(lambda x: x["name"] == selected_smell, self.smell_files)
            )[0]
            smell_type = smell["key"]
            chosen_model = model["results"][smell_type]
            printed_result = pd.DataFrame(
                [
                    smell["name"],
                    chosen_model["accuracy"],
                    chosen_model["f1_score_value"],
                ]
            ).transpose()
            printed_result.columns = ["Smell", "Accuracy", "F1-score"]
            df_results.append(printed_result)
        printed_result = pd.concat(df_results, axis=0).reset_index(drop=True)
        printed_result.index += 1
        print(
            "The accuracy and the F1-score of the test set for your selected smells:\n"
        )
        print(printed_result)
        print("------------------------------------")

    def select_smells(self):
        options = [model["name"] for model in self.smell_files]
        self.selected_smell_types = user_input_selection(
            "Select one or more code smells (comma-separated) or exit: ",
            options,
            True,
        )

    def compare_or_retrain(self):
        # print("Compare or Retrain")
        options = ["compare test and training sets", "retrain"]
        selection = user_input_selection(
            "Select the option you want to do next or exit: ", options, False
        )
        if selection == options[0]:
            # Compare Test and Training
            self.print_compared_result()
            selection = user_input_selection(
                "Select the option you want to do next or exit: ", ["retrain"], False
            )
            if selection == "retrain":
                self.start_interaction()
        else:
            # Retrain
            self.start_interaction()

    def enter_model_type(self):
        options = [model["name"] for model in self.model_dict]
        self.selected_model = user_input_selection(
            "Select one trained model or exit: ",
            options,
            False,
        )

    def start_interaction(self):
        self.enter_model_type()
        self.select_smells()
        self.print_selected_results()
        self.compare_or_retrain()

    def run_application(self):
        self.process_trainings()
        self.start_interaction()


if __name__ == "__main__":
    main = Main()
    main.run_application()
