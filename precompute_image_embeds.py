from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from data.precompute_dataset import PrecomputationDataset 
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils import plot_heatmap, boost_top_k


torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_group = 0
dataset_mode = "val"

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

questions = [
    "a photo taken in the day",
        "a photo taken at night",
        "",
        "",
    "a photo taken when it is raining",
        "a photo taken when it is cloudy",
        "a photo taken when it is partly cloudy",
        "a photo taken when it is sunny",
    "a photo taken when it is windy",
        "a photo taken when it is calm",
        "",
        "",
    "a photo taken when it is humidy",
        "a photo taken when it is dry",
        "",
        "",
    "a photo taken on a monday",
        "a photo taken on a tuesday",
        "a photo taken on a wednesday",
        ""
]

DO_PRECOMPUTE = True
csv_filename = dataset_mode
BATCH_SIZE = 100
precompute_dataset = PrecomputationDataset("./collected_data", 
                                           csv_filename=csv_filename)
precompute_loader = DataLoader(precompute_dataset, 
                               batch_size=BATCH_SIZE, 
                               drop_last=False,
                               shuffle=False) # Shuffle must be set to false to

print(f"Dataset csv: {csv_filename}.csv")
print("Dataset length:", len(precompute_dataset))
print("Expected number of iterations: ~", len(precompute_dataset) // BATCH_SIZE + 1)
do_proceed = input("Proceed? Input `precompute` to proceed with precomputing or `eval` to preview precomputed embeddings: ") 
if do_proceed.lower() == "eval":
    print("Evaluating...")
    DO_PRECOMPUTE = False
elif do_proceed.lower() == "precompute":
    print("Precomputing...")
    DO_PRECOMPUTE = True
else:
    print("None of `precompute` or `eval` selected! Exiting...")
    exit()




@torch.no_grad()
def original_prob_calculation(inputs):
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)

    print(probs.shape)
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


########################
### Main Computation ###
########################
if DO_PRECOMPUTE:
    global_image_embeds = None
    global_baseline_prediciton = None

    with torch.no_grad():
        for i, img in tqdm(enumerate(precompute_loader)):
            # Preprocess text and image
            inputs = processor(text=questions, images=img, return_tensors="pt", padding=True, do_rescale=False)
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

            
            text_embeds = model.get_text_features(input_ids=inputs["input_ids"])
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

            logits_per_image = (torch.matmul(text_embeds, 
                                            image_embeds.t().to(text_embeds.device)) \
                                        * model.logit_scale.exp().to(text_embeds.device)).t()

            if global_image_embeds is None:
                global_image_embeds = image_embeds
                global_baseline_prediciton = logits_per_image
            else:
                global_image_embeds = torch.cat((global_image_embeds, image_embeds), dim=0)
                global_baseline_prediciton = torch.cat((global_baseline_prediciton, logits_per_image), dim=0)


            probs = torch.nn.functional.softmax(logits_per_image, dim=1)

            # print(probs)
            # print(probs.sum(dim=1))
            # print(probs.shape)


            # Check that the probabilities are the same as the original calculation
            assert torch.allclose(probs, original_prob_calculation(inputs))
            print("All good!")


            # if i == 0:  
            #     break

    # global_mean_embed = global_image_embeds.mean(dim=0, keepdim=True)
    # normalized_embeds = global_image_embeds - global_mean_embed
    normalized_embeds = global_image_embeds
    output_path = f"./collected_data/{csv_filename}_img_mbd.pt"
    torch.save(normalized_embeds, output_path)
    print(f"Saved {normalized_embeds.shape[0]} embeddings to {output_path}")
    torch.cuda.empty_cache()
else:
    output_path = f"./collected_data/{csv_filename}_img_mbd.pt"



############################################################
### Load the standardized embeddings and test the model ###
############################################################

sample_tensors = torch.load(output_path, weights_only=True)[:10]
print("Loaded tensor", sample_tensors.shape)


with torch.no_grad():
    # Preprocess text and image
    inputs = processor(text=questions, images=None, return_tensors="pt", padding=True, do_rescale=False)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    image_embeds = sample_tensors
    text_embeds = model.get_text_features(input_ids=inputs["input_ids"])
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    print(image_embeds.shape)

    # Here we are viewing columnwise, each column for an input image
    logits_per_image = (torch.matmul(text_embeds, 
                                    image_embeds.t().to(text_embeds.device)) \
                                * model.logit_scale.exp().to(text_embeds.device))
    # probs = torch.nn.functional.softmax(logits_per_image, dim=0)
    # probs = boost_top_k(probs, k=1, dim=0)

    # print(probs)
    # print(probs.sum(dim=1))

    # print(probs.shape)

    images = precompute_dataset.get_image_paths(0, 10)
    plot_heatmap(logits_per_image.cpu().round(decimals=2), images, questions, output_mode="show")
