import pandas as pd
import random
import os
import json
import torch
import clip
import requests
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse
import re
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # if os.path.exists(os.path.join(args.output_dir, f'{args.user_name}_survey.csv')):
    #     print(f"Survey file for {args.user_name} already exists. Please delete the file and try again.")
    #     return None

    # Get the images from the input file
    input_df = pd.read_csv(args.input_file)

    # Use images that do not have any captions
    input_df = input_df[input_df["caption_A"].isnull() & input_df["caption_B"].isnull()]

    # Sample the images
    input_df = input_df.sample(args.num_sample_images).reset_index(drop=True)
    results_df = input_df.copy()

    # Get the models and processors
    llava_model, llava_processor = get_model_processor("llava")
    clip_small_model, clip_small_processor = get_model_processor("clip-small")
    clip_large_model, clip_large_processor = get_model_processor("clip-large")

    for image_url in tqdm(input_df.image_url.values):

        # Load the image
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Generate Captions:
        # WCOT: without chain-of-thought/without any conditioned prompt
        # COT: with chain-of-thought/with a pre-determined conditioned prompt
        wcot_caption, cot_caption = generate_caption_from_image(
            llava_model, llava_processor, image
        )
        # print(f"WCOT Caption: {wcot_caption}")
        # print(f"COT Caption: {cot_caption}")

        # Process the generated captions
        process_wcot_caption = process_caption(wcot_caption)
        process_cot_caption = process_caption(cot_caption)
        # print(f"Processed WCOT Caption: {process_wcot_caption}")
        # print(f"Processed COT Caption: {process_cot_caption}")

        valid_outcomes = [process_wcot_caption, cot_caption]

        # Evaluate which caption is better according to CLIP: small and large
        clip_small_pred = get_predicted_labels_clip(
            clip_small_model, clip_small_processor, image, valid_outcomes
        )
        clip_large_pred = get_predicted_labels_clip(
            clip_large_model, clip_large_processor, image, valid_outcomes
        )

        # Evaluate which caption is better according to LLM
        llm_pred = get_predicted_labels_llm(
            llava_model, llava_processor, image, valid_outcomes, max_new_tokens=50
        )

        # Evaluate which caption is better according to a human
        # human_pred = get_human_pref_caption(image, valid_outcomes)

        results_df.loc[results_df["image_url"] == image_url, "caption_A"] = (
            process_wcot_caption
        )
        results_df.loc[results_df["image_url"] == image_url, "caption_B"] = (
            process_cot_caption
        )
        results_df.loc[results_df["image_url"] == image_url, "llm_output"] = int(
            llm_pred
        )
        results_df.loc[results_df["image_url"] == image_url, "clip_small_output"] = int(
            clip_small_pred
        )
        results_df.loc[results_df["image_url"] == image_url, "clip_large_output"] = int(
            clip_large_pred
        )

    # Save the results
    results_df.to_csv(os.path.join(args.output_dir, "generated_captions.csv"))


def get_images(input_file, output_file, num_images=10):
    objects = pd.read_csv(input_file)["image_url"]

    random_numbers = random.sample(range(0, len(objects)), num_images)
    images = [objects[i] for i in random_numbers]

    output_file = output_file.append(pd.DataFram([images], columns=["image_url"]))

    return output_file


def get_model_processor(model_name):
    if model_name == "llava":
        model_pref = "llava-hf/llava-v1.6-mistral-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_pref)

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_pref, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        model.to(device)
    elif model_name == "clip-small":
        model, processor = clip.load("ViT-B/32", device=device)
    elif model_name == "clip-large":
        model, processor = clip.load("ViT-L/14", device=device)

    print(f"Model: {model_name} loaded successfully!")

    return model, processor


def generate_caption_from_image(model, processor, image, max_new_tokens=50):

    # Generate prompt
    wcot_prompt = "[INST] <image>\nGenerate a caption for this image in 50 words[/INST]"
    cot_prompt = "[INST] <image>\nGenerate a caption for this image, and the description should include the number of objects in the image without explicitly mentioning it in 50 words[/INST]"

    # Process prompt
    wcot_inputs = processor(wcot_prompt, image, return_tensors="pt").to(device)
    cot_inputs = processor(cot_prompt, image, return_tensors="pt").to(device)

    # autoregressively complete prompt
    wcot_output = model.generate(**wcot_inputs, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.pad_token_id)
    cot_output = model.generate(**cot_inputs, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.pad_token_id)

    # Decode output
    wcot_final_output = processor.decode(wcot_output[0], skip_special_tokens=True)
    cot_final_output = processor.decode(cot_output[0], skip_special_tokens=True)

    wcot_final_output = wcot_final_output.split("[/INST]")[-1].strip()
    cot_final_output = cot_final_output.split("[/INST]")[-1].strip()

    return wcot_final_output, cot_final_output


def process_caption(caption):
    """
    Clean the caption. If the last sentence doesn't end with a period, remove it.

    Args:
        caption (str): Caption generated by the LLM model

    Returns:
        str: Processed caption with a period at the end
    """

    # if '"' in caption:
    if caption[0] == '"':
        if caption[-1] != '"':
            caption = caption + '"'
        caption = re.findall(r'"(.*?)"', caption)
        caption = caption[0] if caption else ""

    # second_split = caption.split(". ")  # split into sentences

    # # remove last sentence if it doesn't end with a period
    # if not second_split[-1].endswith("."):
    #     second_split.pop()

    # final_string = ". ".join(second_split)  # join sentences
    if not caption.endswith("."):
        caption += "."
    return caption


def get_predicted_labels_clip(model, preprocess, image, captions):
    """
    Get the predicted label for the given image and captions generated by LLM model using the CLIP model.

    Args:
        model (clip_model): CLIP model
        preprocess (clip_preprocess): CLIP preprocess function
        image (Image): Image to be evaluated
        captions (list): List of captions generated by the LLM model (WCOT and COT)

    Returns:
        int: Predicted label for the given image
    """

    img = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(captions).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(img, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    final_pred = probs[0].argmax()
    # print("Label probs:", final_pred)
    return final_pred


def get_predicted_labels_llm(model, processor, image, captions, max_new_tokens=50):

    # Generate prompt
    prompt = f"[INST] <image>\nWhich caption is a better choice for the given image: (A) {captions[0]} or (B) {captions[1]}? Give only the option letter to me.[/INST]"

    # Process prompt
    inputs = processor(prompt, image, return_tensors="pt").to(device)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode output
    final_output = processor.decode(output[0], skip_special_tokens=True)

    final_output = final_output.split("[/INST]")[-1].strip()  # remove prompt

    if final_output == "A":
        # print(f"Selected caption: WCOT: {captions[0]}")
        return 0
    else:
        # print(f"Selected caption: COT: {captions[1]}")
        return 1


def get_human_pref_caption(image, captions):
    image.show()
    print("There are 2 captions. Which one do you prefer from the ones?")

    # Shuffle the captions to avoid bias
    random.shuffle(captions)

    print(f"Caption (A): {captions[0]}")
    print(f"Caption (B): {captions[1]}")

    possible_inputs = ["A", "B"]
    for attempts in range(3):
        human_input = input(f"Attempt: {attempts+1}. Enter either option 'A' or 'B'")
        if human_input in possible_inputs:
            break
        else:
            print("Invalid input. Please try again!")

    return human_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate captions for the given image and evaluate the captions using LLM and CLIP models."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=False,
        help="File containing the URLs of the images to be evaluated",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated captions",
    )
    parser.add_argument(
        "--num_sample_images",
        type=int,
        default=500,
        help="Number of images to be sampled for the survey",
    )
    # parser.add_argument(
    #     "--user_name", type=str, required=True, help="Name of the survey taker."
    # )
    args = parser.parse_args()

    main(args)
    
    # python generate_captions_predictions.py --input_file /path/to/input_file.csv --output_dir /path/to/output_dir --num_sample_images 500
