import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class MulticaptionDataset(Dataset):
    def __init__(self, root, mode: str, 
                image_transform=None, 
                text_tokenizer=None):
        self.root_dir = root
        assert mode in ["train", "val"], "mode should be either 'train' or 'val'"

        # load images
        self.images_dir = os.path.join(root, mode)
        print(self.images_dir)
        
        #load captions
        caption_path = os.path.join(root, f"{mode}.csv")
        self.captions_df = pd.read_csv(caption_path)
        caption_path2 = os.path.join(root, f"{mode}2.csv")
        self.captions_df2 = pd.read_csv(caption_path2)

        self.transform = image_transform
        self.tokenizer = text_tokenizer

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        caption = self.captions_df.iloc[idx, 1]
        # caption2 = self.captions_df2.iloc[idx, 1]
        img_name = self.captions_df.iloc[idx, 0]
        
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.tokenizer:
            caption = self.tokenizer(caption)
            # caption2 = self.tokenizer(caption2)

        # return image, caption
        return image, caption

# Example usage
# import clip
# dataset = MulticaptionDataset(
#                 "/home/phli/genAI/datasets/flickr8k", 
#                 "train", 
#                 image_transform=ToTensor(),
#                 text_tokenizer=lambda x: clip.tokenize(x, context_length=77))
# loader = DataLoader(dataset, batch_size=3, shuffle=True)

# print(len(loader))

# for i, (images, captions) in enumerate(loader):
#     print(i, images.shape, captions.shape)
#     vocab_size = captions.shape[-1]
#     print(vocab_size)
#     break