import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class PrecomputationDataset(Dataset):
    def __init__(self, data_dir, csv_filename="train0"):
        self.data_dir = data_dir

        # load images
        self.images_dir = os.path.join(data_dir, 'images')
        print(self.images_dir)
        
        #load captions
        caption_path = os.path.join(data_dir, f"{csv_filename}.csv")
        self.captions_df = pd.read_csv(caption_path, delimiter=">", header=None)
        print("Number of captions:", len(self.captions_df))
        print(self.captions_df.head(10))

        self.to_tensor = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        # caption = self.captions_df.iloc[idx, 1]
        img_path = os.path.join(self.images_dir, self.captions_df.iloc[idx, 0])
        image = self.to_tensor(Image.open(img_path))

        return image
    
    def get_image_paths(self, start, end):
        return [os.path.join(self.images_dir, path) for path in self.captions_df.iloc[start:end, 0]]

# # Example usage
# # import clip
# dataset = PrecomputationDataset("/home/phli/genAI/cp4101_data", mode="train", group="0")
# loader = DataLoader(dataset, batch_size=3, shuffle=True)

# # print(len(loader))

# for i, (images, captions) in enumerate(loader):
#     print(i, images, captions)
