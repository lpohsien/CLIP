import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTextConfig
from peft import get_peft_model, LoraConfig
import yaml

from utils import count_parameters
from data.embed_dataset import EmbedDataset
from trainer import CLIPTrainer
import torch.optim as optim

from os.path import dirname, abspath, join


DEFAULT_CONFIG_PATH = "./configs/default.yaml"
DEFAULT_BASE_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_DATA_ROOT_DIR = "/home/phli/genAI/data_collection/data"
DEFAULT_CHECKPOINT_DIR = "/home/phli/genAI/CLIP/checkpoints"

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPModelModifier:

    def __init__(self, 
                 base_model_id=DEFAULT_BASE_MODEL_ID,
                 finetune_type="lora",
                 train_projection=False,
                 save_dir=DEFAULT_CHECKPOINT_DIR):
        
        self.base_model_id = base_model_id
        self.peft_config = LoraConfig(
                                inference_mode=False, 
                                r=8, 
                                lora_alpha=32, 
                                lora_dropout=0.1,
                                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"] 
                            )
        
        self.train_projection = train_projection
        self.save_dir = save_dir

        self.model = CLIPModel.from_pretrained(base_model_id)
        self.processor = CLIPProcessor.from_pretrained(base_model_id)

        self.context_length = self.model.text_model.config.max_position_embeddings
        print("Context Length:", self.context_length)

        self.trainer = None
        self.model.to(DEFAULT_DEVICE)

        # At this step, we assume all parameters are trainable
        if finetune_type == "lora":
            self.setupLora()

        # At least step, we assume all non text model parameters are frozen
        if self.train_projection:
            self.enableProjectionTraining()

        self.output_model_name = "text"
        if self.train_projection:
            self.output_model_name += "_projection"
        if finetune_type == "lora":
            self.output_model_name += "_lora-finetuned"

        print("Total trainable", count_parameters(self.model, simplified=False))

    def setupLora(self):
            print("Total model trainable (CLIP)", count_parameters(self.model, simplified=False))
            print("Text model traininable (CLIP)", count_parameters(self.model.text_model, simplified=False))

            peft_text_model = get_peft_model(self.model.text_model, self.peft_config)
            self.model.text_model = peft_text_model

            for name, param in self.model.named_parameters():
                # Assume all requires_grad is True, freeze all layers except the text model
                # and the text projection layer (if train_projection is True)
                if "text_model" not in name:
                    param.requires_grad = False

            print("LoRA:", end=" ")
            peft_text_model.print_trainable_parameters()
            print("Total trainable (LORA)", count_parameters(self.model, simplified=False))

    def enableProjectionTraining(self):
        for name, param in self.model.text_projection.named_parameters():
            param.requires_grad = True
        print("Text Projection Layer is now trainable.")

    def setupTrainer(self,
                     data_root_dir=DEFAULT_DATA_ROOT_DIR, 
                     config_path=DEFAULT_CONFIG_PATH):
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        batch_size = config.get('batch_size')
        num_epochs = config.get('num_epochs')
        optimizer = optim.AdamW(
            self.model.parameters(),
            betas=(config.get("adam_beta1"), config.get("adam_beta2")),
            eps=float(config.get("adam_epsilon")),
            lr=config.get("lr"),
            weight_decay=config.get("weight_decay")
        )
        train_dataset = EmbedDataset(data_root_dir, 
                                     img_embed_dir=join(dirname(abspath(__file__)), "collected_data"),
                                     csv_file="train", 
                                     preprocessor=self.processor,
                                     context_length=self.context_length)
        eval_dataset = EmbedDataset(data_root_dir, 
                                    img_embed_dir=join(dirname(abspath(__file__)), "collected_data"),
                                    csv_file="val", 
                                    preprocessor=self.processor,
                                    context_length=self.context_length)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, 
                                                  batch_size=len(eval_dataset)//4, # Load all at once
                                                  shuffle=False)

        
        print("--------- Training Configuration ---------")
        print("Optimizer:", optimizer)
        print("Batch Size:", batch_size)
        print("Number of Epochs:", num_epochs)
        print("Train Dataset Size:", len(train_dataset))
        print("Eval Dataset:", len(eval_dataset))
        print("Train Projection:", self.train_projection)
        print("------------------------------------------")

        self.trainer = CLIPTrainer(
            model=self.model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_epochs=num_epochs,
            use_wandb=False,
            train_projection=self.train_projection,
            save_dir=self.save_dir,
            output_model_name=self.output_model_name,
        )

    def train(self):
        if self.trainer is None:
            raise ValueError("Trainer not set up. Please call setupTrainer() first.")
        self.trainer.train_from_img_embeds()