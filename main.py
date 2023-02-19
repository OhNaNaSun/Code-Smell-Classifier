import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import signal
import sys
from train_models import train_model


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
            # "naive_bayes": {
            #     "classifier": GaussianNB(),
            #     "param_grid": {"var_smoothing": np.logspace(0, -9, num=100)},
            #     "results": {},
            # },
            # "k-neighbors": {
            #     "classifier": KNeighborsClassifier(n_neighbors=5),
            #     "results": {},
            #     "param_grid": {
            #         "n_neighbors": [3, 5, 7],
            #         "weights": ["uniform", "distance"],
            #         "algorithm": ["ball_tree", "kd_tree", "brute"],
            #         "leaf_size": [10, 20, 30],
            #         "p": [1, 2],
            #     },
            # },
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

    def user_input_selection(self, prompt, options, is_multiple_selection):
        options.append("exit")
        while True:
            print(prompt)
            for i, option in enumerate(options):
                print(f"{i + 1}. {option}")

            user_input = input("Enter the number of your selection:")

            try:
                if user_input == str(len(options)):
                    sys.exit(0)

                if is_multiple_selection:
                    user_input = [int(x) for x in user_input.split(",")]
                    for i in user_input:
                        if i < 1 or i > len(options[:-1]):
                            raise ValueError
                else:
                    user_input = int(user_input)
                    if user_input < 1 or user_input > len(options[:-1]):
                        raise ValueError

            except ValueError:
                print("Invalid input. Please try again.")
                continue

            if is_multiple_selection:
                selected_options = [options[:-1][i - 1] for i in user_input]
                print(f"You selected: {', '.join(selected_options)}\n")
                return selected_options
            else:
                selected_option = options[:-1][user_input - 1]
                print(f"You selected: {selected_option}\n")
                return selected_option

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
        self.selected_smell_types = self.user_input_selection(
            "Select one or more code smells or exit: ",
            options,
            True,
        )

    def compare_or_retrain(self):
        # print("Compare or Retrain")
        options = ["compare test and training", "retrain"]
        selection = self.user_input_selection(
            "Select the option you want to do next or exit: ", options, False
        )
        if selection == options[0]:
            # Compare Test and Training
            self.print_compared_result()
            selection = self.user_input_selection(
                "Select the option you want to do next or exit: ", ["retrain"], False
            )
            if selection == "retrain":
                self.start_interaction()
        else:
            # Retrain
            self.start_interaction()

    def enter_model_type(self):
        options = list(self.model_dict.keys())
        self.selected_model = self.user_input_selection(
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
