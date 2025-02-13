# from diffusers import DiffusionPipeline

# pipe = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE")

# print(pipe.text_encoder)


from safetensors.torch import load_file
from transformers import CLIPTextModel, AutoTokenizer

from transformers import CLIPTextConfig


def load_clip_from_state_dict(config, state_dict):
    model = CLIPTextModel(config)
    for key in model.state_dict().keys():
        model.state_dict()[key].copy_(state_dict["cond_stage_model.transformer." + key])
    return model

# default_config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
# model = CLIPTextModel(default_config)
incoming = load_file("/home/phli/genAI/ComfyUI/models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors")


for k in incoming.keys():
    if "." in k:
        print(k)




# model = load_clip_from_state_dict(default_config, incoming)

