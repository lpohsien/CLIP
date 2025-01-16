import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _clip_preprocessor = clip.load("checkpoints/LiT-base_128_8.pth")

clip_model.to(device)

def image_to_embedding(image):
    image_input = _clip_preprocessor(image).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(image_input)
    return image_features

def text_to_embedding(text):
    text_input = clip.tokenize([text]).to(device)
    text_features = clip_model.encode_text(text_input)
    return text_features

def cosine_similarity(text_features, image_features):
    similarity = (100 * image_features @ text_features.T)
    return similarity

img = Image.open("/home/phli/genAI/photo_6068638646237511439_y.jpg")
text = "a photo of a dog"
text_features = text_to_embedding(text)
image_features = image_to_embedding(img)
print(text_features.shape)
print(image_features.shape)
print(cosine_similarity(text_features, image_features))