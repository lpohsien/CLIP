import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class EmbedDataset(Dataset):
    def __init__(self, data_dir, img_embed_dir=None, csv_file="train0", preprocessor=None, context_length=77):
        '''
            data_dir: directory containing images and captions
            img_embed_dir: directory containing image embeddings
            csv_file: name of the csv file containing image paths and captions, without the extension
            preprocessor: CLIPProcessor object
            context_length: maximum number of tokens in a caption
        '''
        self.data_dir = data_dir

        # load images
        self.images_dir = os.path.join(data_dir, 'images')
        # print(self.images_dir)
        
        #load captions
        caption_path = os.path.join(data_dir, f"{csv_file}.csv")
        self.captions_df = pd.read_csv(caption_path, delimiter=">", header=None)
        print("Number of captions:", len(self.captions_df))
        # print(self.captions_df.head(10))


        self.img_embed_dir = img_embed_dir if img_embed_dir is not None else data_dir
        embed_file_path = os.path.join(self.img_embed_dir, f"{csv_file}_img_mbd.pt")
        if not os.path.exists(embed_file_path):
            raise FileNotFoundError(f"Image embeddings file {embed_file_path} not found")
        self.image_embeds = torch.load(embed_file_path, weights_only=True)
        self.preprocessor = preprocessor
        self.context_length = context_length


    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        text_tokens = self.preprocessor.tokenizer(self.captions_df.iloc[idx, 1], return_tensors="pt")['input_ids']
        # text_tokens.shape = (1, num_tokens), num_tokens <= context_length
        #! TODO: Manage the case where num_tokens > context_length
        if text_tokens.shape[1] < self.context_length:
            text_tokens = torch.cat([text_tokens, torch.zeros(1, self.context_length - text_tokens.shape[1], dtype=torch.long)], dim=1)
        elif text_tokens.shape[1] > self.context_length:
            # print(self.captions_df.iloc[idx, 1], text_tokens.shape[1])
            # assert False, "Number of tokens in the caption is greater than the context length"
            text_tokens = text_tokens[:, :self.context_length]

        return self.image_embeds[idx], text_tokens
    
    def get_image_paths(self, start, end):
        return [os.path.join(self.images_dir, path) for path in self.captions_df.iloc[start:end, 0]]

    def get_dataset_representation(self):
        ''' 
            Return a list of image paths and a list of captions and the two lists 
            should be aligned such that the ith caption corresponds to the ith image.
        '''
        images = [os.path.join(self.images_dir, path) for path in self.captions_df.iloc[:, 0].tolist()]
        texts = self.captions_df.iloc[:, 1].tolist()
        return images, texts