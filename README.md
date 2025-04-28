# Sensor Encoder

This is the code for CS4101 Final Year Project (AY2025/S2) on Exploring Efficient Sensor Fusion Encoder for Low-Resource Stable Diffusion Pipelines. This repository contains the code to train a sensor encoder using the [Locked Image Tuning Approach (LiT)](https://arxiv.org/abs/2111.07991) with Low Rank Adaptation using the [Peft library](https://huggingface.co/docs/peft/en/index). Note that the base CLIP model used is [CLIP ViT-L14](https://huggingface.co/openai/clip-vit-large-patch14).

# Image embedding precomputation

Since the LiT training approach do not modify the image encoder, the image embeddings can be precomputed. This can be done by using the [precompute_image_embeds.py](/precompute_image_embeds.py) script. To precompute, the following directory structure is needed:
```
collected_data
    | -- {csv_filename}.csv
    | -- images
            | -- __image1__.jpg
            | -- __image2__.jpg
            ...
```
where each row in `./collected_data/{csv_filename}.csv` should contain the image names and the corresponding sensor data, while the images directory contains all the images listed in the csv file. (Note that when replacing the csv filename in [precompute_image_embeds.py](/precompute_image_embeds.py), the extension name, i.e. `.csv` should be excluded. There will also be input confirmation required when running the precomputation script) Once the script is run, the precomputed image embeddings will be saved in `./collected_data/{csv_filename}_img_mbd.pt`. Further note that the precomputation does not perform any train test split. The script also contains the code to check the image embedding using the original CLIP model, where the image-text association map will be shown at the end if the test is executed.

# Training

Once the image embeddings have been precomputed, the [benchmark.py](/benchmark.py) script can be used to train and benchmark the sensor encoder. Do make sure that the data csv file (`./collected_data/{csv_filename}.csv`) matches with that of the embedding tensor file (`./collected_data/{csv_filename}_img_mbd.pt`), as the `{csv_filename}` segment is used to identify the dataset to use.

Other relevant files include:
- [lora_finetune.py](/lora_finetune.py) This file sets up the LoRA for the sensor encoder. Only the LoRA weights will be trained.
- [trainer.py](/trainer.py): This file includes the code on how the sensor encoder is being trained per epoch
- [utils.oy](/utils.py): This file includes all the relevant functions such as the loss functions, evaluation metrics as well as functions to plot the relevant heatmaps.

Once the training is completed, the sensor encoder will be saved in the [checkpoints](/checkpoints/) directory. The benchmarking against the baseline model will also be run automatically, by swapping out the original text encoder with the trained sensor encoder.