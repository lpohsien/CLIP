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
        dim=1 performs top-k along each row, dim=0 along each column.
    '''
    assert len(logits.shape) == 2, "Logits must be 2D"
    _, top_k = logits.topk(k, dim=dim)
    true_labels = torch.arange(logits.size(dim - 1)).repeat(k, 1).to(logits.device)
    if dim == 1:
        true_labels = true_labels.t()
    correct = top_k == true_labels    
    return correct.sum(dim=dim).float().mean().item() * 100.0

def boost_top_k(matrix, k, dim=1) -> torch.Tensor:
    """
    Convert a matrix to binary based on the row-wise top-k elements using PyTorch.

    Parameters:
        matrix (torch.Tensor): Input 2D tensor.
        k (int): Number of top elements to assign 1 in each row.
        dim (int): Dimension along which to perform the top-k operation. 
            dim=1 performs top-k along the rows, dim=0 along the columns.
            default: 1

    Returns:
        torch.Tensor: Transformed binary matrix of the same dimensions.
    """
    _, top_k_indices = torch.topk(matrix, k, dim=dim)
    matrix.scatter_(dim, top_k_indices, 1.0)

    return matrix


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

# Adapted from:
# https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb
def plot_heatmap(similarity,
                    images,
                    texts,
                    title_class="Image retrieval probility",
                    model_name="unknown",
                    output_mode="none",
                    save_dir=dirname(abspath(__file__)),
                    fit_to_text=True):
    assert output_mode in ["show", "save", "both", "none"], \
        "Invalid output mode! Only 'none', 'show', 'save', and 'both' are allowed."
    assert similarity.shape[0] == len(texts), "Number of texts should match the similarity matrix"
    assert similarity.shape[1] == len(images), "Number of images should match the similarity matrix"
    texts_count = len(texts)
    images_count = len(images)

    plt.figure(figsize=(max(20, images_count * 3.0), max(14, texts_count * 1.0)))
    if similarity.shape[0] == similarity.shape[1]:
        plt.imshow(torch.eye(similarity.shape[0]))
        plt.imshow(similarity, vmin=0.1, vmax=0.3, alpha=0.75)
    else:
        plt.imshow(similarity)
    plt.colorbar(label="Probability")
    plt.yticks(range(texts_count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        image = Image.open(image)
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, images_count - 0.5])
    plt.ylim([texts_count + 0.5, -2])

    plt.title(f"{title_class} for {model_name}", fontsize=60)

    if output_mode in ["save", "both"]:
        title_class = title_class.replace(" ", "_")
        if fit_to_text:
            plt.savefig(join(save_dir, 
                             f"{title_class.replace(" ", "_")}_heatmap_{model_name}.png"), bbox_inches="tight")
        else:
            plt.savefig(join(save_dir, 
                             f"{title_class.replace(" ", "_")}_heatmap_{model_name}.png"))
    if output_mode in ["show", "both"]:
        plt.show()

matrix = torch.Tensor([[1, 2, 8], 
                       [3, 4, 3]])
