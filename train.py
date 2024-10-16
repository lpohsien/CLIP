from data.multicaption_dataset import MulticaptionDataset

import clip
from clip.model import CLIP

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.cuda
import yaml
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.utils.data import Subset


CONFIG_PATH = "./configs/mini-vit.yaml"
# DATASET_ROOT = "/home/phli/genAI/datasets/flickr8k"
DATASET_ROOT = "/home/svu/e0268113/datasets/flickr8k"
# DATASET_ROOT = "/scratch/e0268113/datasets/flickr8k"
SEED = 42
OG_CONFIG_PATH = "./configs/vit-b32.yaml"
SAVE_CHECKPOINT_PATH = "./checkpoints"
# LOAD_CHECKPOINT_PATH = "./checkpoints/mini-vit_256_4.pth"
LOAD_CHECKPOINT_PATH = None
SAVE_CHECKPOINT = True

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F
def flat_loss(image, caption, model):
    _, _, img_feat, text_feat = model(image, caption)
    features = torch.cat([img_feat, text_feat], dim=0)
    # features_flipped = torch.cat([text_feat, img_feat], dim=0)
    # features = torch.cat([features, features_flipped], dim=1)
    
    labels = torch.cat([torch.arange(TRAIN_BATCH_SIZE) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    assert similarity_matrix.shape == (
        2 * TRAIN_BATCH_SIZE, 2 * TRAIN_BATCH_SIZE)
    assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    labels = torch.zeros(positives.shape[0], dtype=torch.long).to(device) #-

#        logits = logits / self.args.temperature #- 
    logits = (negatives - positives) * model.logit_scale.exp() # (512,510) #+

    v = torch.logsumexp(logits, dim=1, keepdim=True) #(512,1)
    loss_vec = torch.exp(v-v.detach())
    
    assert loss_vec.shape == (len(logits),1)
    dummy_logits = torch.cat([torch.zeros(logits.size(0),1).to(device), logits],1)
    loss = loss_vec.mean() - 1 + torch.nn.CrossEntropyLoss()(dummy_logits, labels) #+
    
    return loss, logits

# train the model using InfoNCE loss
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    epoch_losses = torch.zeros(len(train_loader)).to(device)
    model.train()
    total_loss = torch.tensor(0.0).to(device)
    for i, (images, captions) in enumerate(tqdm.tqdm(train_loader)):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        # logits, _ = model(images, captions)
        # loss = criterion(logits, device=device)
        if images.size(0) != TRAIN_BATCH_SIZE:
            continue
        loss, _ = flat_loss(images, captions, model)
        
        # print(logits)
        # input()
        epoch_losses[i] = loss
        loss.backward()
        optimizer.step()
        total_loss += loss

    total_loss = total_loss.item()
    return total_loss / len(train_loader), epoch_losses

def eval(model, val_loader, criterion, device):
    clip_model.eval()
    total_loss = torch.tensor(0.0).to(device)
    with torch.no_grad():
        for images, captions in tqdm.tqdm(val_loader):
            logits, _, _, _ = model(images.to("cuda"), captions.to("cuda"))
            total_loss += criterion(logits, device=device)
            torch.cuda.empty_cache()
        print(logits)
    total_loss = total_loss.item()
    return total_loss / len(val_loader)

def save_checkpoint(model):
    if SAVE_CHECKPOINT_PATH:
        if not os.path.exists(SAVE_CHECKPOINT_PATH):
            os.mkdir(SAVE_CHECKPOINT_PATH)
        config_filename = CONFIG_PATH.split("/")[-1].split(".")[0]
        prev_ckpt_name = LOAD_CHECKPOINT_PATH.split("/")[-1].split(".")[0] if LOAD_CHECKPOINT_PATH else None
        ckpt_name = config_filename
        batch_size_str = str(TRAIN_BATCH_SIZE)
        training_epochs_str = str(TRAINING_EPOCHS)
        if prev_ckpt_name:
            prev_batch_size = prev_ckpt_name.split("_")[1]
            prev_training_epochs = prev_ckpt_name.split("_")[2]
            if prev_batch_size != batch_size_str:
                batch_size_str = prev_batch_size + "/" + batch_size_str
            if prev_training_epochs != training_epochs_str:
                training_epochs_str = prev_training_epochs + "/" + training_epochs_str
        ckpt_name = config_filename + "_" + batch_size_str + "_" + training_epochs_str
        torch.save(model.state_dict(), f"{SAVE_CHECKPOINT_PATH}/{ckpt_name}.pth")
        print(f"Model saved to {SAVE_CHECKPOINT_PATH}/{ckpt_name}.pth")
    else:
        print("No checkpoint path provided! Not saving checkpoint.")




# 1. Load training configs
# Load model config from yaml
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# BATCH_SIZE = config.get("batch_size")
# TRAINING_EPOCHS = config.get("epochs")
TRAIN_BATCH_SIZE = 256
TRAINING_EPOCHS = 32

print(f"BATCH SIZE: {TRAIN_BATCH_SIZE}")
print(f"EPOCHS: {TRAINING_EPOCHS}")

# 2. Load model

# Load pretrained CLIP model
# clip_model, _clip_preprocessor = clip.load("ViT-B/32", log_config_path=OG_CONFIG_PATH)
# clip_model.to(device)

# Load custom CLIP model
clip_model = None
if LOAD_CHECKPOINT_PATH:
    print(f"Loading model from {LOAD_CHECKPOINT_PATH}")
    clip_model, _ = clip.load(LOAD_CHECKPOINT_PATH) # Don't need the preprocessor
else:
    print("Initializing model from scratch")
    clip_model = CLIP(
            embed_dim=config.get("embed_dim"),
            # vision
            image_resolution=config.get("visual").get("image_resolution"),
            vision_layers=config.get("visual").get("n_layers"),
            vision_width=config.get("visual").get("width"),
            vision_patch_size=config.get("visual").get("patch_size"),
            # text
            context_length=config.get("text").get("context_length"),
            vocab_size=config.get("text").get("vocab_size"),
            transformer_width=config.get("text").get("width"),
            transformer_heads=config.get("text").get("n_heads"),
            transformer_layers=config.get("text").get("n_layers"),
        ).to(device)
    clip_model.initialize_parameters()

# 3. Load the dataset
# We used the preprocessed image data, hence clip._transform is not needed
simple_tokenizer = lambda x: clip.tokenize(x, context_length=config.get("text").get("context_length"))[0]
train_set = MulticaptionDataset(DATASET_ROOT, "train", image_transform=ToTensor(), text_tokenizer=simple_tokenizer)
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_set = MulticaptionDataset(DATASET_ROOT, "val", image_transform=ToTensor(), text_tokenizer=simple_tokenizer)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# 3b. Train on only a subset of the data (Testing only)
# subset_indices = subset_indices = list(range(0, 1600))
# subset_train_set = Subset(train_set, subset_indices)
# train_loader = DataLoader(subset_train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

# 4. Define loss and optimizer
optimizer = optim.Adam(
    clip_model.parameters(),
    betas=(config.get("adam_beta1"), config.get("adam_beta2")),
    eps=float(config.get("adam_epsilon")),
    lr=config.get("lr"),
    weight_decay=config.get("weight_decay")
)

# 5. Train the model
train_losses = np.array([])
eval_losses = np.array([])
min_loss = float("inf")
for epoch in range(TRAINING_EPOCHS):
    training_loss, epoch_losses = train_one_epoch(clip_model, train_loader, optimizer, clip.clip.InfoNCELoss, device)
    print(f"Epoch {epoch + 1} | Average InfoNCE Loss (Training): {training_loss}")
    train_losses = np.append(train_losses, epoch_losses.cpu().detach().numpy())
    train_losses = train_losses[:-1] # Remove the average from last batch since it is likely << batch size

    current_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"Current VRAM Usage: {current_memory / 1024**2:.2f} MB")
    print(f"Peak VRAM Usage: {peak_memory / 1024**2:.2f} MB")

    if SAVE_CHECKPOINT and epoch % 4 == 0:
        eval_loss = eval(clip_model, val_loader, clip.clip.InfoNCELoss, device)
        print(f"Epoch {epoch + 1} | Average InfoNCE Loss (Eval): {eval_loss}")
        eval_losses = np.append(eval_losses, eval_loss)
        if eval_loss < min_loss:
            min_loss = eval_loss
            save_checkpoint(clip_model)
            
    if epoch % 4 == 0:
        with open("train_losses.npy", "wb") as f:
            np.save(f, train_losses)
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epoch')
        plt.grid(True)
        # plt.ylim(5.53, 5.55)
        plt.savefig("train_loss_plot.png")


# 6. Evaluate the model
clip_model.eval()
eval_loss = 0
with torch.no_grad():
    for images, captions in tqdm.tqdm(val_loader):
        logits, _ = clip_model(images.to("cuda"), captions.to("cuda"))
        eval_loss += clip.clip.InfoNCELoss(logits, device=device).item()
        torch.cuda.empty_cache()

print(f"Final Average Evaluation InfoNCE Loss: {eval_loss / len(val_loader)}")