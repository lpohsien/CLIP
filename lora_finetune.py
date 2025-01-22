import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTextConfig
from peft import get_peft_model, LoraConfig
import yaml

from utils import count_parameters
from data.embed_dataset import EmbedDataset
from trainer import CLIPTrainer
import torch.optim as optim

from os.path import dirname, abspath


DEFAULT_CONFIG_PATH = "./configs/default.yaml"
DEFAULT_BASE_MODEL_NAME = "clip-vit-large-patch14"
DEFAULT_DATA_ROOT_DIR = "/home/phli/genAI/data_collection/data"

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPTextLoRAFinetune:

    def __init__(self, 
                 base_model=DEFAULT_BASE_MODEL_NAME):
        self.base_model = base_model
        self.peft_config = LoraConfig(
                                inference_mode=False, 
                                r=8, 
                                lora_alpha=32, 
                                lora_dropout=0.1,
                                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"] 
                            )
        self.model = CLIPModel.from_pretrained(f"openai/{base_model}")
        self.processor = CLIPProcessor.from_pretrained(f"openai/{base_model}")

        self.context_length = self.model.text_model.config.max_position_embeddings
        print("Context Length:", self.context_length)

        self.trainer = None
        self.model.to(DEFAULT_DEVICE)

        print("Total model trainable (CLIP)", count_parameters(self.model, simplified=False))
        print("Text model traininable (CLIP)", count_parameters(self.model.text_model, simplified=False))

        peft_text_model = get_peft_model(self.model.text_model, self.peft_config)
        self.model.text_model = peft_text_model

        for name, param in self.model.named_parameters():
            if "text_model" not in name:
                param.requires_grad = False

        print("LoRA:", end=" ")
        peft_text_model.print_trainable_parameters()
        print("Total trainable (LORA)", count_parameters(self.model, simplified=False))

    def setupTrainer(self,
                     data_root_dir=DEFAULT_DATA_ROOT_DIR, 
                     batch_size=60, 
                     config_path=DEFAULT_CONFIG_PATH):
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        batch_size = config.get('batch_size')
        optimizer = optim.Adam(
            self.model.parameters(),
            betas=(config.get("adam_beta1"), config.get("adam_beta2")),
            eps=float(config.get("adam_epsilon")),
            lr=config.get("lr"),
            weight_decay=config.get("weight_decay")
        )
        train_dataset = EmbedDataset(data_root_dir, 
                                     img_embed_dir=dirname(abspath(__file__)),
                                     csv_file="train0", 
                                     preprocessor=self.processor,
                                     context_length=self.context_length)
        eval_dataset = EmbedDataset(data_root_dir, 
                                    img_embed_dir=dirname(abspath(__file__)),
                                    csv_file="val0", 
                                    preprocessor=self.processor,
                                    context_length=self.context_length)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, 
                                                  batch_size=len(eval_dataset), # Load all at once
                                                  shuffle=False)
        
        print("--------- Training Configuration ---------")
        print("Batch Size:", batch_size)
        print("Optimizer:", optimizer)
        print("Train Dataset Size:", len(train_dataset))
        print("Eval Dataset:", len(eval_dataset))
        print("------------------------------------------")

        self.trainer = CLIPTrainer(
            model=self.model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            use_wandb=False
        )

    def train(self):
        if self.trainer is None:
            raise ValueError("Trainer not set up. Please call setupTrainer() first.")
        self.trainer.train_from_img_embeds()
