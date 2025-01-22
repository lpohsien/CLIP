import torch
from tqdm import tqdm
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, CLIPTextConfig
from peft import get_peft_model, LoraConfig
import yaml

from utils import count_parameters
from data.embed_dataset import EmbedDataset
from trainer import CLIPTrainer
import torch.optim as optim

from os.path import dirname, abspath
from utils import clip_loss, recall_at_k, plot_heatmap, rowwise_top_k_binary

from lora_finetune import CLIPTextLoRAFinetune


DEFAULT_CONFIG_PATH = "./configs/default.yaml"
DEFAULT_BASE_MODEL_NAME = "clip-vit-large-patch14"
DEFAULT_DATA_ROOT_DIR = "/home/phli/genAI/data_collection/data"

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPLoRABenchmarkRunner:

    def __init__(self, 
                 data_root_dir=DEFAULT_DATA_ROOT_DIR,
                 embedding_file="val0_std_img_mbd.pt",
                 baseline_model="openai/clip-vit-large-patch14",
                 models=None):
        self.baseline_model = baseline_model
        self.processor = CLIPProcessor.from_pretrained(baseline_model)
        self.models = models
        self.model = CLIPModel.from_pretrained(baseline_model)
        self.model.to(DEFAULT_DEVICE)
        self.context_length = self.model.text_model.config.max_position_embeddings
        print("Context Length:", self.context_length)

        benchmark_dataset = EmbedDataset(DEFAULT_DATA_ROOT_DIR, 
                                                img_embed_dir=dirname(abspath(__file__)),
                                                csv_file="val0", 
                                                preprocessor=self.processor,
                                                context_length=self.context_length)
        self.benchmark_size = len(benchmark_dataset)
        self.bench_loader = torch.utils.data.DataLoader(benchmark_dataset,
                                batch_size=self.benchmark_size, # Load all at once
                                shuffle=False)
        print("Benchmark Dataset Size:", self.benchmark_size)

    def run_baseline(self):

        # Reload baseline model
        del self.model
        self.model = CLIPModel.from_pretrained(self.baseline_model)
        self.model.to(DEFAULT_DEVICE)
        results = self.bench_from_img_embeds(self.model, 
                                             model_name="baseline",
                                             eval_loader=self.bench_loader,
                                             save_dir=dirname(abspath(__file__)))
        results["model_name"] = "baseline"
        print("Baseline Results:", results)
        return results

    @staticmethod
    def bench_from_img_embeds(model: CLIPModel, 
                              model_name: str,
                              eval_loader: torch.utils.data.DataLoader, 
                              top_k: int = 1,
                              save_dir: str = None):
        '''
            Evaluate the model.
            eval_loader: __getitem__ should return (image_embedding, tokenized_caption)
        '''
        assert type(eval_loader.dataset).__name__ == "EmbedDataset", "Loader must be of type EmbedDataset"
        assert len(eval_loader) == 1, "Batch size of the benchmark dataset must be equal to the dataset size"
        model.eval()
        with torch.no_grad():
            metric_acc = torch.zeros(3).to(model.device)
            for i, (image_embeds, input_ids) in enumerate(tqdm(eval_loader)):

                image_embeds = image_embeds.to(model.device)
                # image_embeds are already computed
            
                input_ids = input_ids.to(model.device)
                text_embeds = model.get_text_features(input_ids=input_ids)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

                logits_per_text = torch.matmul(text_embeds, 
                                                image_embeds.t().to(text_embeds.device)) \
                                                * model.logit_scale.exp().to(text_embeds.device)
                metric_acc[0] += clip_loss(logits_per_text)
                metric_acc[1] += recall_at_k(logits_per_text, k=top_k)
                metric_acc[2] += recall_at_k(logits_per_text.t(), k=top_k)
                probs = torch.nn.functional.softmax(logits_per_text.t(), dim=1)
                assert torch.allclose(probs.sum(dim=1), 
                                      torch.tensor(1.0, device=probs.device)), \
                        "Probabilities should sum to 1" 

                ## Convert to top-k (binary representation) for better visualization
                # probs = rowwise_top_k_binary(probs, k=top_k)

                if save_dir is not None:

                    torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)
                    
                    probs *= 100
                    print(probs)
                    # Ensure the loader is of type EmbedDataset
                    images, texts = eval_loader.dataset.get_dataset_representation()
                    plot_heatmap(probs.cpu().round().int(), 
                                 images, 
                                 texts,
                                 model_name=model_name,
                                 output_mode="save",
                                 save_dir=save_dir)

        metric_acc /= len(eval_loader)
        metric_acc = metric_acc.tolist()

        return {"eval_loss": metric_acc[0], 
                "eval_per_text_accuracy": metric_acc[1], 
                "eval_per_image_accuracy": metric_acc[2],
                "probs": probs}


runner = CLIPLoRABenchmarkRunner()
baseline_results = runner.run_baseline()

del runner

lora_finetune = CLIPTextLoRAFinetune()
lora_finetune.setupTrainer()

init_results = CLIPLoRABenchmarkRunner.bench_from_img_embeds(
    model=lora_finetune.model,
    model_name="lora_init", 
    eval_loader=lora_finetune.trainer.eval_loader,
    save_dir=dirname(abspath(__file__)))
init_results["model_name"] = "lora_init"

lora_finetune.train()
final_results = CLIPLoRABenchmarkRunner.bench_from_img_embeds(
    model=lora_finetune.model,
    model_name="lora_finetune", 
    eval_loader=lora_finetune.trainer.eval_loader,
    save_dir=dirname(abspath(__file__)))
final_results["model_name"] = "lora_finetune"

df = pd.DataFrame([baseline_results, init_results, final_results])
columns = [df.columns[-1]] + list(df.columns[:-1])
df = df[columns]
del df["probs"]
print(df) 