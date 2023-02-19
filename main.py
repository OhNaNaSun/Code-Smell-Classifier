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
        self.model_dict = {
            "decision_tree": {
                "classifier": DecisionTreeClassifier(),
                "param_grid": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": np.arange(10, 21),
                    "min_samples_leaf": [1, 5, 10, 20, 50, 100],
                },
                "results": {},
            },
            "naive_bayes": {
                "classifier": GaussianNB(),
                "param_grid": {"var_smoothing": np.logspace(0, -9, num=100)},
                "results": {},
            },
            "k-neighbors": {
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
        }

        self.smell_files = {
            "feature_envy": "feature-envy.arff",
            "data_class": "data-class.arff",
            "god_class": "god-class.arff",
            "long_method": "long-method.arff",
        }
        self.selected_model = None
        self.selected_smell_types = None

    def process_trainings(self):
        model_names = self.model_dict.keys()
        for model in model_names:
            for smell_type in list(self.smell_files.keys()):
                current_model = self.model_dict[model]
                filename = "results/" + model + "-" + smell_type + ".json"
                # Save or Read data
                if not os.path.isfile(filename) or not os.path.getsize(filename) > 0:
                    with open(filename, "w") as f:
                        (
                            accuracy,
                            f1_score_value,
                            train_accuracy,
                            train_f1_score_value,
                        ) = train_model(
                            self.smell_files[smell_type],
                            current_model["classifier"],
                            current_model["param_grid"],
                        )
                        data = {
                            "accuracy": accuracy,
                            "f1_score_value": f1_score_value,
                            "train_accuracy": train_accuracy,
                            "train_f1_score_value": train_f1_score_value,
                        }
                        json.dump(data, f)
                        self.model_dict[model]["results"][smell_type] = data
                else:
                    with open(filename, "r") as f:
                        data = json.load(f)
                        self.model_dict[model]["results"][smell_type] = data

    def print_compared_result(self):
        print(
            "The comparison between the test set and training set for each of your selected smells:\n"
        )
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
            print("----------------------------------------")

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
        print(
            "The accuracy and the F1-score of the test set for your selected smells:\n"
        )
        print(printed_result)
        print("------------------------------------")

    def select_smells(self):
        options = list(self.smell_files.keys())
        self.selected_smell_types = user_input_selection(
            "Select one or more code smells or exit: ",
            options,
            True,
        )

    def compare_or_retrain(self):
        # print("Compare or Retrain")
        options = ["compare test and training", "retrain"]
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
        options = list(self.model_dict.keys())
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
