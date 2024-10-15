import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_ROOT_DIR = "../datasets/flickr8k"
SEED = 42

def clean_caption(text):
    # Remove extra space in front of punctuations
    text = re.sub(r'\s+([.,!?;:])', r'\1',text)
    # Append a period at the end of the sentence if it doesn't have one
    if not text.endswith("."):
        text += "."
    return text

def sample_captions(df):
    '''
    Sample one caption per image. This is for the Flickr8k dataset
    which has 5 captions per image.
    '''
    return df.groupby("image") \
        .apply(lambda x: x.sample(1, random_state=SEED)) \
        .reset_index(drop=True)

# Path to your Flickr8k dataset images and captions
src_images_folder = os.path.join(DATASET_ROOT_DIR, "Images")
src_captions_file = os.path.join(DATASET_ROOT_DIR, "captions.txt")

# Path to save the train and val splits
train_images_folder = os.path.join(DATASET_ROOT_DIR, "train_raw")
val_images_folder = os.path.join(DATASET_ROOT_DIR, "val_raw")
train_captions_file = os.path.join(DATASET_ROOT_DIR, "train.csv")
val_captions_file = os.path.join(DATASET_ROOT_DIR, "val.csv")

# Load captions data
# Assuming captions.txt has two columns: "image_name" and "caption"
captions_df = pd.read_csv(src_captions_file, delimiter=",", names=["image", "caption"])

captions_df["caption"] = captions_df["caption"].apply(clean_caption)

# Get the list of unique images
unique_images = captions_df["image"].unique()

# Split the data into training and valing sets (80% train, 20% val)
train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=SEED)

# Save the split to separate files
train_df = sample_captions(captions_df[captions_df["image"].isin(train_images)])
val_df = sample_captions(captions_df[captions_df["image"].isin(val_images)])

# Save to CSV or TXT for further use
train_df.to_csv(train_captions_file, index=False)
val_df.to_csv(val_captions_file, index=False)

# Move images to train and val folders
# if not os.path.exists(train_images_folder):
#     os.makedirs(train_images_folder)
# if not os.path.exists(val_images_folder):
#     os.makedirs(val_images_folder)
# for image in train_images:
#     os.rename(os.path.join(src_images_folder, image), os.path.join(train_images_folder, image))
# for image in val_images:
#     os.rename(os.path.join(src_images_folder, image), os.path.join(val_images_folder, image))

assert len(train_images) + len(val_images) == len(unique_images), \
    "Images from train + val doesn't match the total images"
assert len(os.listdir(train_images_folder)) == len(train_images), \
    "Some images are missing in the train folder"
assert len(os.listdir(val_images_folder)) == len(val_images), \
    "Some images are missing in the val folder"

# Print summary
print(f"Total images: {len(unique_images)}")
print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
