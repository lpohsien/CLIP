from data.multicaption_dataset import MulticaptionDataset

import clip
from clip.model import CLIP

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.cuda

# Config parsing
import yaml
import argparse

# Statistics tracking
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import os

# Optional: Load a subset of the data
from torch.utils.data import Subset

CONFIG_PATH = "./configs/mini-vit.yaml"
DATASET_ROOT = "/home/phli/genAI/datasets/flickr8k"
# DATASET_ROOT = "/home/svu/e0268113/datasets/flickr8k"
SEED = 42
OG_CONFIG_PATH = "./configs/vit-b32.yaml"
SAVE_CHECKPOINT_DIR = "./checkpoints"
# LOAD_CHECKPOINT_PATH = "./checkpoints/mini-vit_256_32.pth"
LOAD_CHECKPOINT_PATH = None
SAVE_CHECKPOINT = False
DO_TRAIN = True

RUN_TYPE = "text-text"
TRAIN_BATCH_SIZE = 32
TRAINING_EPOCHS = 4
LOG_DIR = f"./logs"

RUN_NAME = f"run-{RUN_TYPE}-{TRAIN_BATCH_SIZE}-{TRAINING_EPOCHS}" + "-eval" if not DO_TRAIN else ""

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def recall_at_k(logits, k=1, dim=1):
    assert len(logits.shape) == 2, "Logits must be 2D"
    _, top_k = logits.topk(k, dim=dim)
    true_labels = torch.arange(logits.size(dim - 1)).repeat(k, 1).to(logits.device)
    if dim == 1:
        true_labels = true_labels.t()
    correct = top_k == true_labels    
    return correct.sum(dim=dim).float().mean().item() * 100.0


# train the model using InfoNCE loss
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    epoch_losses = torch.zeros(len(train_loader)).to(device)
    model.train()
    total_loss = torch.tensor(0.0).to(device)
    for i, (images, captions) in enumerate(tqdm.tqdm(train_loader)):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        logits, _ = model(images, captions)
        loss = criterion(logits, device=device)
        # print(logits)
        # input()
        epoch_losses[i] = loss
        loss.backward()
        optimizer.step()
        total_loss += loss
        if device == "cuda":
            torch.cuda.empty_cache()

    total_loss = (total_loss).item()
    return total_loss / (len(train_loader)), epoch_losses

def eval(model, val_loader, criterion, device):
    model.eval()
    total_loss = torch.tensor(0.0).to(device)
    with torch.no_grad():
        for images, captions in tqdm.tqdm(val_loader):
            logits, _ = model(images.to("cuda"), captions.to("cuda"))
            total_loss += criterion(logits, device=device)
            if device == "cuda":
                torch.cuda.empty_cache()
    total_loss = total_loss.item()
    return total_loss / len(val_loader)

def benchmark(model, bench_loader, topk=1, device="cuda", final=False):
    model.eval()
    recall = 0
    with torch.no_grad():
        images, captions = next(iter(bench_loader))
        logits, _ = model(images.to(device), captions.to(device))
        recall = recall_at_k(logits, k=topk)
        print(logits)

        if final:
            plt.figure(figsize=(10, 10))
            sns.heatmap(logits.cpu().numpy(), annot=False, fmt=".2f", cmap="coolwarm")
            plt.title("Logits")
            plt.xlabel("captions")  
            plt.ylabel("images")
            plt.savefig(os.path.join(LOG_DIR, f"{RUN_NAME}-logits.png"))

    return recall

def save_checkpoint(model):
    if SAVE_CHECKPOINT_DIR:
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
        torch.save(model.state_dict(), f"{SAVE_CHECKPOINT_DIR}/{ckpt_name}.pth")
        print(f"Model saved to {SAVE_CHECKPOINT_DIR}/{ckpt_name}.pth")
    else:
        print("No checkpoint path provided! Not saving checkpoint.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_root", type=str, default=DATASET_ROOT)
    argparser.add_argument("--save_checkpoint_dir", type=str, default=SAVE_CHECKPOINT_DIR)
    argparser.add_argument("--load_checkpoint_path", type=str, default=LOAD_CHECKPOINT_PATH)
    argparser.add_argument("--train_batch_size", type=int, default=TRAIN_BATCH_SIZE)
    argparser.add_argument("--training_epochs", type=int, default=TRAINING_EPOCHS)
    args = argparser.parse_args()

    # Directory Related
    DATASET_ROOT = args.dataset_root
    SAVE_CHECKPOINT_DIR = args.save_checkpoint_path
    LOAD_CHECKPOINT_PATH = args.load_checkpoint_path

    # Training Related
    TRAIN_BATCH_SIZE = args.train_batch_size
    TRAINING_EPOCHS = args.training_epochs
    
    
    print(f"-------------- {RUN_NAME} --------------")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Save checkpoint path: {SAVE_CHECKPOINT_DIR}")
    print(f"Load checkpoint path: {LOAD_CHECKPOINT_PATH}")
    print(f"")
    print("Training configurations:")
    print(f"  Train batch size: {TRAIN_BATCH_SIZE}")
    print(f"  Training epochs: {TRAINING_EPOCHS}")
    print(f"  Updating checkpoint: {SAVE_CHECKPOINT}")
    print(f"  Evalation only: {not DO_TRAIN}")
    print(f"----------------------------------------")

    # 1. Load training configs
    # Load model config from yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


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

    # 3. Load the dataset
    # We used the preprocessed image data, hence clip._transform is not needed
    simple_tokenizer = lambda x: clip.tokenize(x, context_length=config.get("text").get("context_length"))[0]
    train_set = MulticaptionDataset(DATASET_ROOT, "train", image_transform=ToTensor(), text_tokenizer=simple_tokenizer)
    val_set = MulticaptionDataset(DATASET_ROOT, "val", image_transform=ToTensor(), text_tokenizer=simple_tokenizer)
    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, drop_last=True)
    bench_loader = DataLoader(val_set, batch_size=64, shuffle=False, drop_last=True)

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
    if DO_TRAIN:
        train_losses = np.array([])
        eval_losses = np.array([])
        min_loss = float("inf")
        for epoch in range(TRAINING_EPOCHS):
            training_loss, epoch_losses = train_one_epoch(clip_model, train_loader, optimizer, clip.clip.InfoNCELoss, device)
            print(f"Epoch {epoch + 1} | Average InfoNCE Loss (Training): {training_loss}")
            train_losses = np.append(train_losses, epoch_losses.cpu().detach().numpy())

            if device == torch.device("cuda"):
                print(f"Peak VRAM Allocated : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

            if epoch % 2 == 0:
                if SAVE_CHECKPOINT:
                    eval_loss = eval(clip_model, val_loader, clip.clip.InfoNCELoss, device)
                    print(f"Epoch {epoch + 1} | Average InfoNCE Loss (Eval): {eval_loss}")
                    eval_losses = np.append(eval_losses, eval_loss)
                    if eval_loss < min_loss and epoch > 0: # Don't save the first epoch
                        min_loss = eval_loss
                        save_checkpoint(clip_model)

                with open(os.path.join(LOG_DIR, f"{RUN_NAME}-losses.npy"), "wb") as f:
                    np.save(f, train_losses)
                plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
                plt.xlabel('Cycles')
                plt.ylabel('Loss')
                plt.title('Training Loss vs. Epoch')
                plt.grid(True)
                plt.gca().xaxis.set_major_locator(MultipleLocator(len(train_loader)))
                plt.savefig(os.path.join(LOG_DIR, f"{RUN_NAME}-losses.png"))

                r_at_1 = benchmark(clip_model, bench_loader, topk=1, device=device)
                print(f"Intermediate R@1: {r_at_1}")


    # 6. Evaluate the model
    clip_model.eval()
    eval_loss = 0
    with torch.no_grad():
        for images, captions in tqdm.tqdm(val_loader):
            logits, _ = clip_model(images.to("cuda"), captions.to("cuda"))
            eval_loss += clip.clip.InfoNCELoss(logits, device=device).item()
            torch.cuda.empty_cache()

    print(f"Final Average Evaluation InfoNCE Loss: {eval_loss / len(val_loader)}")

    # 7. Benchmark the model
    r_at_1 = benchmark(clip_model, bench_loader, topk=1, device=device, final=True)
    print(f"Final R@1: {r_at_1}")