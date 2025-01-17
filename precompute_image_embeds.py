from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from data.precompute_dataset import PrecomputationDataset 
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from tqdm import tqdm


torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)

def plot_heatmap(probs, images, text):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the probability matrix using coolwarm colormap
    im = ax.imshow(probs, cmap="coolwarm", aspect="auto")

    # Customize the axis labels
    ax.set_yticks(range(len(images)))
    ax.set_yticklabels([])  # Hide default y-axis labels (we'll use images instead)
    ax.set_xticks(range(len(text)))
    ax.set_xticklabels(text, rotation=45, ha="right")  # Rotate text labels for better readability

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
    plt.show()


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

text = ["a photo taken in the day", 
        "a photo taken at night", 
        "a photo taken when it is raining",
        "a photo taken when it is cloudy",
        "a photo taken when it is partly cloudy",
        "a photo taken when it is sunny",
        "a photo taken when it is clear",
        "a photo taken in the urban setting"]

precompute_dataset = PrecomputationDataset("/home/phli/genAI/cp4101_data", mode="train", group="0")
precompute_loader = DataLoader(precompute_dataset, batch_size=10, shuffle=False)


@torch.no_grad()
def original_prob_calculation(inputs):
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)
    return probs


@torch.no_grad()
def compute_text_embeds(input):
    text_outputs = model.text_model(input_ids=input["input_ids"])
    text_embeds = text_outputs[1]
    text_embeds = model.text_projection(text_embeds)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return text_embeds

@torch.no_grad()
def compute_img_embeds(input):
    vision_outputs = model.vision_model(pixel_values=input["pixel_values"])
    image_embeds = vision_outputs[1]
    image_embeds = model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds



global_mean_embed = None
text_embeds = None
global_image_embeds = None

for i, img in tqdm(enumerate(precompute_loader)):
    # Preprocess text and image
    inputs = processor(text=text, images=img, return_tensors="pt", padding=True, do_rescale=False)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    
    if text_embeds is None:
        text_embeds = compute_text_embeds(inputs)
    
    image_embeds = compute_img_embeds(inputs)

    print(image_embeds.shape)

    if global_mean_embed is None:
        global_mean_embed = image_embeds
        global_mean_embed = global_mean_embed.mean(dim=0, keepdim=True)
        global_image_embeds = image_embeds
    else:
        print(image_embeds.shape, global_mean_embed.shape)
        global_mean_embed = torch.cat((global_mean_embed, image_embeds), dim=0)
        global_mean_embed = global_mean_embed.mean(dim=0, keepdim=True)
        global_image_embeds = torch.cat((global_image_embeds, image_embeds), dim=0)

    logits_per_text = torch.matmul(text_embeds, 
                                    image_embeds.t().to(text_embeds.device)) \
                                * model.logit_scale.exp().to(text_embeds.device)
    logits_per_image = logits_per_text.t()


    probs = torch.nn.functional.softmax(logits_per_image, dim=1)

    print(probs)
    print(probs.sum(dim=1))


    ## Check that the probabilities are the same as the original calculation
    # assert torch.allclose(probs, original_prob_calculation(inputs))
    # print("All good!")

    # images = precompute_dataset.get_image_paths(i*5, (i+1)*5)
    # plot_heatmap(probs.cpu().round().int(), images, text)

    if i == 0:  break


# normalized_embeds = global_image_embeds - global_mean_embed
# torch.save(normalized_embeds, "standardized_image_embeds.pt")
# torch.cuda.empty_cache()



############################################################
### Load the standardized embeddings and test the model ###
############################################################


# all_image_tensors = torch.load("standardized_image_embeds.pt", weights_only=True)[:10]

# print("Loaded tensor", all_image_tensors.shape)



# # Preprocess text and image
# inputs = processor(text=text, images=None, return_tensors="pt", padding=True, do_rescale=False)
# inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

# image_embeds = all_image_tensors
# text_embeds = compute_text_embeds(inputs)

# logits_per_text = torch.matmul(text_embeds, 
#                                 image_embeds.t().to(text_embeds.device)) \
#                             * model.logit_scale.exp().to(text_embeds.device)
# logits_per_image = logits_per_text.t()
# probs = torch.nn.functional.softmax(logits_per_image, dim=1)

# print(probs)
# print(probs.sum(dim=1))

# images = precompute_dataset.get_image_paths(0, 10)
# plot_heatmap(probs.cpu().round().int(), images, text)