import pandas as pd
import argparse
import json
import requests
import os
from PIL import Image


def main(args):

    if args.debug:
        debug(args)
        return None

    assert os.path.exists(args.input_dir), f"Input file {args.input_dir} does not exist"

    if not os.path.exists(args.output_dir):
        # Create the output directory if it does not exist
        os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, "visual_genome_available_images.csv")

    with open(args.input_dir, "r") as f:
        objects = json.load(f)

    objects = pd.DataFrame(objects, columns=["image_url"])

    # Get only the first 20 elements for testing
    objects = objects.head(1000)
    print(out_path)
    objects["image_exists"] = objects["image_url"].apply(load_image)

    objects = objects[objects["image_exists"] == 1]
    print(f"Number of images available: {len(objects)}")

    save_report(objects, out_path, args.num_images)


def load_image(image_url):
    try:
        _ = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        print(f"Image {image_url} loaded successfully")
        return 1
    except Exception as e:
        return 0


def save_report(chunks, out_path, num_images):
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
    else:
        df = pd.DataFrame(columns=["image_url"])
    chunks = chunks.reset_index(drop=True)
    df = pd.concat([df, chunks])

    # Add blank columns to the dataframe for later use
    df["caption_A"] = [None] * df.shape[0]
    df["caption_B"] = [None] * df.shape[0]
    df["human_output"] = [None] * df.shape[0]
    df["llm_output"] = [None] * df.shape[0]
    df["clip_small_output"] = [None] * df.shape[0]
    df["clip_large_output"] = [None] * df.shape[0]
    df["human_evaluator"] = [None] * df.shape[0]

    df = df[
        [
            "image_url",
            "caption_A",
            "caption_B",
            "human_output",
            "llm_output",
            "clip_small_output",
            "clip_large_output",
            "human_evaluator"
        ]
    ]

    df = df.sample(num_images).reset_index(drop=True)

    df.to_csv(out_path, index=False)


def debug(args):
    print("Debugging mode")
    assert os.path.exists(args.input_dir), f"Input file {args.input_dir} does not exist"
    if not os.path.exists(args.output_dir):
        # Create the output directory if it does not exist
        os.makedirs(args.output_dir, exist_ok=True)

    user = str(args.output_dir).split("/")[-1]
    out_path = os.path.join(args.output_dir, f"{user}_survey.csv")

    objects = pd.DataFrame(
        [{"image_url": "https://www.google.com"}], columns=["image_url"]
    )

    save_report(objects, out_path)
    if os.path.exists(out_path):
        print(f"Output file {out_path} exists")

    # Remove the folder containing the output file even if the file exists
    os.rmdir(args.output_dir)
    if not os.path.exists(out_path):
        print(f"Output file {out_path} removed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select the top report/sentences based on CXR-RePaiR method"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="File containing the URLs of the images to be extracted",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="File containing the extracted URLs of the images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        required=True,
        help="How many images to run survey for",
    )
    parser.add_argument("--debug", type=bool, required=False, help="Debugging mode")

    args = parser.parse_args()

    main(args)
