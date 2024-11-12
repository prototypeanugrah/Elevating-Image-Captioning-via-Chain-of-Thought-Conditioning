# Elevating Image Captioning via Chain-of-Thought Conditioning

## Overview

This project aims to conduct an analysis on prompt engineering for image captioning task.

### Extract the images from the dataset

This project uses the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) for evaluating the different models. Extract the image URLs from the objects.json file from the data page.

```
python extract_images.py \
    --input_dir <path_to_input_file> \
    --output_dir <path_to_output_dir> \
    --num_images <number_of_images_to_evaluate>

```

### Generate captions

For the images in the dataset, generate captions using an image captioning model. Here, image captions are generated using the [llava-v1.6-mistral-7b-hf](https://llava-vl.github.io) model.

For this analysis, we provide two separate prompts to the model to generate captions -
1. Without Chain-of-thought prompt (A): ```[INST] <image>\nGenerate a caption for this image in 50 words[/INST]```
2. With Chain-of-thought prompt (B): ```[INST] <image>\nGenerate a caption for this image, and the description should include the number of objects in the image without explicitly mentioning it in 50 words[/INST]```

Pass the generated captions to the CLIP ```ViT-L/14``` and the CLIP ```"ViT-B/32``` models.
Evaluate which image better according to the following models -
1. [CLIP Large](https://github.com/openai/CLIP): ```ViT-L/14```
2. [CLIP Base](https://github.com/openai/CLIP): ```"ViT-B/32```
3. [LLaVA](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf): ```llava-v1.6-mistral-7b-hf```

Running this script

```
python generate_captions_predictions.py \
    --input_file /path/to/input_file.csv \
    --output_dir /path/to/output_dir \
    --num_sample_images 500
```

### Conducting human surveys

This script is designed to conduct human-based surveys for image evaluation. It reads an input CSV file, clears previous survey results if needed, and randomly selects images for human evaluation. It allows human evaluators to choose their preferred captions based on the given image, and stores the results in a new CSV file.

Running this script

```
python human_surveys.py --input_csv /path/to/output_file.csv \
    --survey_dir /path/to/output_dir \
    --num_sample_images=<number_of_images_to_evaluate_per_evaluator> \
    --evaluator=<name_of_evaluator>
```

### Dependencies

```
Python 3.11.8
PyTorch 2.2.1
HuggingFace Hub 0.21.4
transformers 4.38.2
NumPy 1.26.4
Pandas 2.2.1
requests 2.31.0
```
