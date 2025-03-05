# Takes in an existing dataset and shrink it
# Requres: 
#   1. csv file containing the image paths and their corresponding captions
#   2. Precomputed image embedding (optional)

# Usage:
# Always take the first n samples from the dataset
# Will save to same directory as the original dataset

import csv
import logging
import os
import torch
import random

DEFAULT_DATASET_NAME = 'val'
DEFAULT_CSV_PATH = f'collected_data/{DEFAULT_DATASET_NAME}.csv'
DEFAULT_IMAGE_EMBEDDINGS_PATH = f'collected_data/{DEFAULT_DATASET_NAME}_img_mbd.pt'

def reduce_dataset(csv_path: str=DEFAULT_CSV_PATH, image_embeddings_path: str=DEFAULT_IMAGE_EMBEDDINGS_PATH, size: int=30):
    '''
    csv_path: str
        Path to the csv file containing the dataset
    image_embeddings_path: str
        Path to the precomputed image embeddings
    size: int
        Number of samples to take from the dataset
    '''
    original_name = os.path.basename(csv_path).split(".")[0]
    data = None

    total_size = 0

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    ### Take subset of csv file
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        total_size = len(data)

    if size > total_size:
        logging.warning("Requested size is larger than the dataset. Taking the whole dataset instead.")
        size = total_size

    select_index = random.sample(range(0, total_size - 1), size)

    data = [data[i] for i in select_index]  

    for d in data:
        print(d)

    with open(csv_path.replace(original_name, f'{original_name}{size}'), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    logging.info(f"Reduced dataset from {total_size} to {size} samples ({size/total_size*100:.2f}%)")
    logging.info(f"Saved to {csv_path.replace(original_name, f'{original_name}{size}')}")



    ### Take corresponding subset of image embeddings (assuming they are aligned)
    if image_embeddings_path is not None:

        if not os.path.exists(image_embeddings_path):
            logging.error("Image embeddings not found at %s. Skipping...", image_embeddings_path)
            return

        image_embeds = torch.load(image_embeddings_path, weights_only=True)[select_index]
        torch.save(image_embeds, image_embeddings_path.replace(original_name, f'{original_name}{size}'))

    logging.info(f"Saved {size} embeddings to {image_embeddings_path.replace(original_name, f'{original_name}{size}')}")


logging.basicConfig(level=logging.INFO)
reduce_dataset()
