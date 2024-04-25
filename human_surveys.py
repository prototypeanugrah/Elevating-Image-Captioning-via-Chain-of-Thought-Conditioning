import os
import pandas as pd
from PIL import Image
import argparse
import requests
import random


def main(args):

    # if args.clear_previous, then clear the human_output column of the input_csv
    if args.clear_previous:
        all_images = pd.read_csv(args.input_csv)
        print(
            f'Number of evaluations to be cleared: {all_images["human_output"].notnull().sum()}'
        )
        all_images["human_output"] = None
        all_images.to_csv(args.input_csv, index=False)
        print("Cleared the previous human survey results.")
        exit()

    all_images = pd.read_csv(args.input_csv)
    results_df = all_images.copy()

    # Filter out the images that have already been evaluated
    results_df = results_df[results_df["human_output"].isnull()]
    # print(f"Number of images to be evaluated: {results_df.shape[0]}")
    results_df = results_df.sample(args.num_sample_images).reset_index(drop=True)
    results_df["human_evaluator"] = args.evaluator

    if not os.path.exists(args.survey_dir):
        os.makedirs(args.survey_dir)

    for idx, row in results_df.iterrows():
        image_url = row["image_url"]
        image = Image.open(requests.get(image_url, stream=True).raw)
        captions = [row["caption_A"], row["caption_B"]]
        human_pref = get_human_pref_caption(image, captions)
        results_df.loc[idx, "human_output"] = human_pref

    results_df = results_df.reset_index(drop=True)
    results_df.to_csv(
        os.path.join(args.survey_dir, f"{args.evaluator}_survey.csv"), index=False
    )

    # Now, map the human preferences to the original dataframe
    all_images["human_output"] = all_images["image_url"].map(
        results_df.set_index("image_url")["human_output"]
    )
    all_images["human_evaluator"] = all_images["image_url"].map(
        results_df.set_index("image_url")["human_evaluator"]
    )

    all_images.to_csv(args.input_csv, index=False)


def get_human_pref_caption(image, captions):
    image.show()
    print(
        "Image displayed. Please evaluate the image with the following captions. NOTE: You will get 3 attempts to enter the correct option."
    )
    print("There are 2 captions. Which one do you prefer from the ones? \n")

    # Shuffle the captions to avoid bias
    random.shuffle(captions)

    print(f"Caption (A): {captions[0]} \n")
    print(f"Caption (B): {captions[1]} \n")

    possible_inputs = ["A", "B"]
    for attempts in range(3):
        human_input = input(f"Attempt: {attempts+1}. Enter either option 'A' or 'B': ")
        if human_input in possible_inputs:
            break
        else:
            print("Invalid input. Please try again!")

    if human_input == "A":
        return 0
    else:
        return 1
    # return human_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conduct human surveys for stored images. Get their opinions on the images."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="CSV file containing the URLs of the images to be evaluated",
    )
    parser.add_argument(
        "--survey_dir",
        type=str,
        required=False,
        help="Directory to store the survey results conducted",
    )
    parser.add_argument(
        "--num_sample_images",
        type=int,
        default=10,
        help="Number of images to be sampled for the survey",
    )
    parser.add_argument(
        "--evaluator", type=str, required=False, help="Name of the survey taker."
    )
    parser.add_argument(
        "--clear_previous",
        type=bool,
        required=False,
        help="Clear the previous survey results or not. Default: False",
    )
    args = parser.parse_args()

    main(args)

    # Script to run the human surveys
    # python human_surveys.py --input_csv results/generated_captions.csv --survey_dir surveys/ --num_sample_images 3 --evaluator test_user

    # Script to clear the previous survey results
    # python human_surveys.py --input_csv results/generated_captions.csv --clear_previous True
