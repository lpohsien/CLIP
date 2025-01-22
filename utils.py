import torch # Needed for all

# Needed for contrastive loss
from torch.nn.functional import cross_entropy

# Needed for benchmark
from tqdm import tqdm

# Needed for get_gpu_memory
import subprocess as sp
import psutil

# Needed for plot_heatmap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from os.path import dirname, abspath, join

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    '''Computes the contrastive loss for a batch of logits.'''
    return cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    '''Mean of the contrastive loss for both the image and caption.'''
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def recall_at_k(logits: torch.Tensor, k=1, dim=1) -> float:
    ''' 
        Compute the recall at k for a batch of logits.
        dim=1 performs top-k along the rows, dim=0 along the columns.
    '''
    assert len(logits.shape) == 2, "Logits must be 2D"
    _, top_k = logits.topk(k, dim=dim)
    true_labels = torch.arange(logits.size(dim - 1)).repeat(k, 1).to(logits.device)
    if dim == 1:
        true_labels = true_labels.t()
    correct = top_k == true_labels    
    return correct.sum(dim=dim).float().mean().item() * 100.0

def rowwise_top_k_binary(matrix, k):
    """
    Convert a matrix to binary based on the row-wise top-k elements using PyTorch.

    Parameters:
        matrix (torch.Tensor): Input 2D tensor.
        k (int): Number of top elements to assign 1 in each row.

    Returns:
        torch.Tensor: Transformed binary matrix of the same dimensions.
    """
    _, top_k_indices = torch.topk(matrix, k, dim=1)
    binary_matrix = torch.zeros_like(matrix, dtype=torch.float)
    binary_matrix.scatter_(1, top_k_indices, 100)

    return binary_matrix


# Adpated from:
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model, simplified=True):
    '''Count the number of trainable parameters in a model.'''
    res = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if simplified:
        if res > 1e6:
            return f"{res/1e6:.2f}M"
        elif res > 1e3:
            return f"{res/1e3:.2f}K"
    return res


def check_memory_usage(threshold=15):
    '''
    Check the memory usage of the CPU and GPU, raise an error if precentage of free memory is below the threshold.
    threshold: minimum percentage of memory available to pass the test
    '''
    # CPU memory usage
    precent_free_ram = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    if precent_free_ram < threshold:
        raise MemoryError("Out of Memory (CPU)!")

    # GPU memory usage
    if torch.cuda.is_available():
        command = "nvidia-smi --query-gpu=memory.free,memory.total --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1]
        available, total = [int(x.split()[0]) for x in memory_free_info.split(", ")]
        precent_free_vram = available / total * 100
        if precent_free_vram < threshold:
            raise MemoryError("Out of Memory (GPU)!")

    return precent_free_ram, precent_free_vram


def plot_heatmap(probs, 
                 images, 
                 text, 
                 model_name="unknown", 
                 output_mode="none",
                 save_dir=dirname(abspath(__file__)),
                 fit_to_text=False):
    
    assert output_mode in ["show", "save", "both", "none"], \
        "Invalid output mode! Only 'none', 'show', 'save', and 'both' are allowed."
    # Create a figure and axis
    _, ax = plt.subplots(figsize=(8, 6))

    # Plot the probability matrix using coolwarm colormap
    im = ax.imshow(probs, cmap="coolwarm", aspect="auto", vmin=0, vmax=100)

    # Customize the axis labels
    ax.set_yticks(range(len(images)))
    ax.set_yticklabels([])  # Hide default y-axis labels (we'll use images instead)
    ax.set_xticks(range(len(text)))
    ax.set_xticklabels(text, rotation=20, ha="right")  # Rotate text labels for better readability

    # Add image thumbnails as y-axis labels
    for i, image_path in enumerate(images):
        # Load the image and create an OffsetImage
        img = Image.open(image_path)
        img.thumbnail((60, 60))  # Resize for the thumbnail
        offset_img = OffsetImage(img, zoom=1.0)

        # Create an AnnotationBbox for the image
        ab = AnnotationBbox(offset_img, (-0.5, i), frameon=False, box_alignment=(1.0, 0.5))
        ax.add_artist(ab)

    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Probability")

    # Adjust layout and display the plot
    plt.tight_layout()

    if output_mode in ["show", "both"]:
        plt.show()
    if output_mode in ["save", "both"]:
        if fit_to_text:
            plt.savefig(join(save_dir, f"heatmap_{model_name}.png"), bbox_inches="tight")
        else:
            plt.savefig(join(save_dir, f"heatmap_{model_name}.png"))