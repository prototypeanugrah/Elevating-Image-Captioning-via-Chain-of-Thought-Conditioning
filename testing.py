import argparse


def main(args):
    user_input_list = []
    for i in range(args.num_input):
        user_input = input("Enter a number: ")
        user_input_list.append(user_input)
    print(user_input_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debugging mode")
    parser.add_argument(
        "--num_input",
        type=int,
        required=True,
        help="Directory to save the output file",
    )
    args = parser.parse_args()

    main(args)