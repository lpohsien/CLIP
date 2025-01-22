from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from data.precompute_dataset import PrecomputationDataset 
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils import plot_heatmap


torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_group = 0
dataset_mode = "val"

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

csv_filename = "val0"

precompute_dataset = PrecomputationDataset("/home/phli/genAI/data_collection/data", 
                                           csv_filename=csv_filename)
precompute_loader = DataLoader(precompute_dataset, 
                               batch_size=5, 
                               shuffle=False) # Shuffle must be set to false to


@torch.no_grad()
def original_prob_calculation(inputs):
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)
    return probs


@torch.no_grad()
def compute_text_embeds(inputs):
    text_outputs = model.text_model(input_ids=inputs["input_ids"])
    text_embeds = text_outputs[1]
    text_embeds = model.text_projection(text_embeds)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return text_embeds

@torch.no_grad()
def compute_img_embeds(inputs):
    vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
    image_embeds = vision_outputs[1]
    image_embeds = model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds



global_mean_embed = None
text_embeds = None
global_image_embeds = None

with torch.no_grad():
    for i, img in tqdm(enumerate(precompute_loader)):
        # Preprocess text and image
        inputs = processor(text=text, images=img, return_tensors="pt", padding=True, do_rescale=False)
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        
        if text_embeds is None:
            text_embeds = model.get_text_features(input_ids=inputs["input_ids"])
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

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


    # Check that the probabilities are the same as the original calculation
    assert torch.allclose(probs, original_prob_calculation(inputs))
    print("All good!")

    # images = precompute_dataset.get_image_paths(i*10, (i+1)*10)
    # plot_heatmap(probs.cpu().round().int(), images, text)

    # if i == 0:  break


normalized_embeds = global_image_embeds - global_mean_embed
output_path = f"{csv_filename}_std_img_mbd.pt"
torch.save(normalized_embeds, output_path)
print(f"Saved {normalized_embeds.shape[0]} embeddings to {output_path}")
torch.cuda.empty_cache()



############################################################
### Load the standardized embeddings and test the model ###
############################################################

sample_tensors = torch.load(output_path, weights_only=True)[:10]
print("Loaded tensor", sample_tensors.shape)


with torch.no_grad():
    # Preprocess text and image
    inputs = processor(text=text, images=None, return_tensors="pt", padding=True, do_rescale=False)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    image_embeds = sample_tensors
    text_embeds = model.get_text_features(input_ids=inputs["input_ids"])
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    logits_per_text = torch.matmul(text_embeds, 
                                    image_embeds.t().to(text_embeds.device)) \
                                * model.logit_scale.exp().to(text_embeds.device)
    logits_per_image = logits_per_text.t()
    probs = torch.nn.functional.softmax(logits_per_image, dim=1)

    print(probs)
    print(probs.sum(dim=1))

    probs *= 100
    images = precompute_dataset.get_image_paths(0, 10)
    plot_heatmap(probs.cpu().round().int(), images, text)
