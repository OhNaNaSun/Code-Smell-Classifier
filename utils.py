import sys


def user_input_selection(prompt, options, is_multiple_selection):
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
