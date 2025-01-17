import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def recall_at_k(logits, k=1, dim=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    assert len(logits.shape) == 2, "Logits must be 2D"
    _, top_k = logits.topk(k, dim=dim)
    true_labels = torch.arange(logits.size(dim - 1)).repeat(k, 1).to(logits.device)
    if dim == 1:
        true_labels = true_labels.t()
    correct = top_k == true_labels    
    return correct.sum(dim=dim).float().mean().item() * 100.0

def benchmark(model, bench_loader, topk=1, device="cuda", final=False, wandb=None, USE_WANDB=False, LOG_DIR=None, RUN_NAME=None):
    model.eval()
    with torch.no_grad():
        images, captions = next(iter(bench_loader))
        with torch.no_grad():
            logits, _ = model(images.to(device), captions.to(device))
        recall_image = recall_at_k(logits, k=topk)
        recall_text = recall_at_k(logits, k=topk, dim=0)
        print(logits)
        if USE_WANDB:
            wandb.log({"R@1 Image": recall_image, "R@1 Text": recall_text})
        if final:
            plt.figure(figsize=(10, 10))
            sns.heatmap(logits.cpu().numpy(), annot=False, fmt=".2f", cmap="coolwarm")
            plt.title("Logits")
            plt.xlabel("captions")  
            plt.ylabel("images")
            plt.savefig(os.path.join(LOG_DIR, f"{RUN_NAME}-logits.png"))
    return recall_image, recall_text

# Adpated from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model, simplified=True):
    res = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if simplified:
        if res > 1e6:
            return f"{res/1e6:.2f}M"
        elif res > 1e3:
            return f"{res/1e3:.2f}K"
        else:
            return res
