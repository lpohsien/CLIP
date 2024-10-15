import numpy as np
import matplotlib.pyplot as plt

from data.multicaption_dataset import MulticaptionDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import clip
import torch
import pandas as pd

losses = np.load("./train_losses-saved.npy")

print(losses[202])
print(losses[404])

# plt.plot(losses[0:203], scalex=True, scaley=True)
# plt.show()

DATASET_ROOT = "/home/phli/genAI/datasets/flickr8k"
TRAIN_BATCH_SIZE = 32
device = "cuda"
tokenizer = lambda x: clip.tokenize(x, context_length=77)[0]
train_set = MulticaptionDataset(DATASET_ROOT, "train", image_transform=ToTensor(), text_tokenizer=tokenizer)
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=5)

# captions_df = pd.read_csv(f"{DATASET_ROOT}/train.csv")

clip_model, _ = clip.load("ViT-B/32")
# clip_model, _ = clip.load("checkpoints/mini-vit_32_2.pth")
clip_model.to(device)

for i, (images, captions) in enumerate(train_loader):
    if i == 202 or i == 0:
        # for caption in captions:
        #     image_name = captions_df.loc[captions_df["caption"] == caption]["image"].values[0]
        #     print(f"{image_name}: {caption}")

        clip_model.eval()
        with torch.no_grad():
            logits, _ = clip_model(images.to("cuda"), captions.to("cuda"))
            print(logits)
            loss = clip.clip.InfoNCELoss(logits, device=device).item()
            print(loss)

    