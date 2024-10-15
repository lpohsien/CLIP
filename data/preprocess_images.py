'''
Preprocess images to standardize the size and format before training the model.
Adapted from:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/contrastive-image-text/run_clip.py
'''
import os
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToPILImage 
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image
from torch.utils.data import DataLoader, Dataset
import tqdm
import yaml

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


DATASET_ROOT_DIR = "../datasets/flickr8k"
SEED = 42
IMAGE_SIZE = 224
N_CHANNELS = 3

_use_default_transforms = True # The model transform Normalize(mean, stdev) should match here if False

# Pre-computed mean and stdev of the training set of Flickr8k
# _flickr8k_train_mean = torch.tensor([[0.4578, 0.4462, 0.4043]])
# _flickr8k_train_stdev = torch.tensor([[0.2425, 0.2332, 0.2370]])

# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class ImagePreprocessor(torch.nn.Module):
    def __init__(self, image_size=224, 
                 # This is the default mean and stdev used by CLIP following ImageNet distribution
                 mean=(0.48145466, 0.4578275, 0.40821073), 
                 std=(0.26862954, 0.26130258, 0.27577711)):
        super().__init__()
        self.image_size = image_size
        self.image_mean = mean
        self.image_stdev = std
        self.transforms = torch.nn.Sequential(
            Resize([self.image_size], interpolation=BICUBIC),
            CenterCrop(self.image_size),
            ConvertImageDtype(torch.float),
            Normalize(self.image_mean, self.image_stdev)
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x
    
# Custom wrapper to load images from directory directly for pre-processing
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        # Get list of all image file paths
        self.image_paths = [os.path.join(root, img) for img in os.listdir(root) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Load image and ensure it's in RGB format
        if self.transform:
            image = self.transform(image)
        return image
    
######################
# Set up directories #
######################

train_images = os.path.join(DATASET_ROOT_DIR, "train_raw")
val_images = os.path.join(DATASET_ROOT_DIR, "val_raw")
target_train_dir = train_images.replace("_raw", "")
target_val_dir = val_images.replace("_raw", "")

if not os.path.exists(target_train_dir):
    os.makedirs(target_train_dir)
if not os.path.exists(target_val_dir):
    os.makedirs(target_val_dir)

image_transformations = ImagePreprocessor(IMAGE_SIZE)
image_transformations = torch.jit.script(image_transformations)

#######################
# Apply preprocessing #
#######################

# for image_folder in [train_images, val_images]:
#     dataset = ImageDataset(root=train_images, transform=transforms.ToTensor())
#     full_loader = DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())

#     # Compute mean and stdev of images in the dataset
#     if not _use_default_transforms:
#         dataset_mean = torch.zeros(N_CHANNELS)
#         dataset_std = torch.zeros(N_CHANNELS)
#         for images in tqdm.tqdm(full_loader):
#             for i in range(N_CHANNELS):
#                 dataset_mean[i] += images[:,i,:,:].mean()
#                 dataset_std[i] += images[:,i,:,:].std()
#         dataset_mean.div_(len(dataset))
#         dataset_std.div_(len(dataset))
#         print(f"Dataset {image_folder}\n mean: {dataset_mean}\n std: {dataset_std}")

#         # Apply the pre-processing transforms to the images
#         image_transformations = ImagePreprocessor(
#             IMAGE_SIZE, dataset_mean.tolist(), dataset_std.tolist()
#         )

#     n_images = 0
#     for image_name in tqdm.tqdm(os.listdir(image_folder)):
#         image_path = os.path.join(image_folder, image_name)
#         image = read_image(image_path, mode=ImageReadMode.RGB)
#         image = image_transformations(image)
#         image = ToPILImage()(image)
#         image.save(image_path.replace("_raw", ""))
#         n_images += 1

#     print(f"Preprocessed {n_images} images in {image_folder}")

#################
# Save metadata #
#################

n_train = len(os.listdir(target_train_dir))
n_val = len(os.listdir(target_val_dir))
dataset_metadata = {
    "image_size": IMAGE_SIZE,
    "n_channels": N_CHANNELS,
    "dataset_mean": list(image_transformations.image_mean),
    "dataset_std": list(image_transformations.image_stdev),
    "n_train": n_train,
    "n_val": n_val,
    "val_ratio": n_val / (n_train + n_val),
    "train_dir": target_train_dir,
    "val_dir": target_val_dir
}
metadata_path = os.path.join(DATASET_ROOT_DIR, "metadata.yaml")

with open(metadata_path, "w") as f:
    yaml.dump(dataset_metadata, f, sort_keys=False)