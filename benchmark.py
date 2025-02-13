from os import PathLike
from os.path import dirname, abspath, exists, join
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from peft import PeftConfig, PeftModel
from safetensors import safe_open

from data.embed_dataset import EmbedDataset
from utils import clip_loss, recall_at_k, plot_heatmap, boost_top_k

import pandas as pd
from lora_finetune import CLIPModelModifier
from trainer import CLIPTrainer
import torch.optim as optim


torch.manual_seed(0)

DEFAULT_BASE_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_DATA_ROOT_DIR = "./collected_data"

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_QUESTIONS = [
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
    "a photo taken when it is humid",
        "a photo taken when it is dry",
        "",
        "",
    "a photo taken on a monday",
        "a photo taken on a tuesday",
        "a photo taken on a wednesday",
        ""
]

def col_partial_argmax(logits: torch.Tensor, 
                group_size: int = 4,
                output_softmax: bool = True) -> tuple[torch.Tensor, (torch.Tensor | None)]:
    '''
        Perform softmax on each group of cols in the logits tensor.
        logits: Tensor of shape (N, M) where M is divisible by group_size
        group_size: Number of rows to group together
        output_argmax: If True, return the argmax of the softmax result
    '''
    print(logits.shape)
    assert logits.shape[1] % group_size == 0, "Number of col should be divisible by group size"

    num_groups = logits.shape[1] // group_size
    logits = logits.view(logits.shape[0], num_groups, group_size) # reshape to  (N, num_groups, group_size)
    grouped_softmax = torch.nn.functional.softmax(logits, dim=2)
    argmax_result = grouped_softmax.argmax(dim=2).view(logits.shape[0], -1)

    if output_softmax:
        softmax_result = grouped_softmax.view(logits.shape[0], -1) # reshape back to (N, M)
        return argmax_result, softmax_result
    return argmax_result, None

class CLIPLoRABenchmarkRunner:

    def __init__(self, 
                 data_root_dir=DEFAULT_DATA_ROOT_DIR,
                 img_embed_dir=dirname(abspath(__file__)),
                 baseline_model=DEFAULT_BASE_MODEL_ID,
                 models=None,
                 device=DEFAULT_DEVICE):
        self.device = device
        self.baseline_model = baseline_model
        self.processor = CLIPProcessor.from_pretrained(baseline_model)
        self.models = models
        self.model = CLIPModel.from_pretrained(baseline_model)
        self.model.to(self.device)
        self.context_length = self.model.text_model.config.max_position_embeddings
        print("Context Length:", self.context_length)

        benchmark_dataset = EmbedDataset(data_root_dir, 
                                                img_embed_dir=img_embed_dir,
                                                csv_file="val", 
                                                preprocessor=self.processor,
                                                context_length=self.context_length)
        self.benchmark_size = len(benchmark_dataset)
        self.bench_loader = torch.utils.data.DataLoader(benchmark_dataset,
                                batch_size=self.benchmark_size, # Load all at once
                                shuffle=False)
        self.text_eval_ground_truth = None
        print("Benchmark Dataset Size:", self.benchmark_size)

    def run_baseline(self):
        # Reload baseline model
        del self.model
        self.model = CLIPModel.from_pretrained(self.baseline_model)
        self.model.to(self.device)
        results = self.evaluate_with_img_embeds(self.model, 
                                             model_name="baseline",
                                             eval_loader=self.bench_loader,
                                             save_dir=dirname(abspath(__file__)))
        

        if self.text_eval_ground_truth is None:
            qn_eval_results = self.evaluate_with_textual_questions(self.model, 
                                                            model_name="baseline",
                                                            questions=DEFAULT_QUESTIONS,
                                                            input_mode="input_ids",
                                                            eval_loader=self.bench_loader,
                                                            question_preprocessor=self.processor,
                                                            save_dir=dirname(abspath(__file__)))
            self.text_eval_ground_truth = qn_eval_results["text_eval_ground_truth"]
        else:
            qn_eval_results = self.evaluate_with_textual_questions(self.model, 
                                                            model_name="baseline",
                                                            questions=DEFAULT_QUESTIONS,
                                                            input_mode="input_ids",
                                                            eval_loader=self.bench_loader,
                                                            question_preprocessor=self.processor,
                                                            ground_truth=self.text_eval_ground_truth,
                                                            save_dir=dirname(abspath(__file__)))
        qn_eval_results.pop("text_eval_ground_truth")

        results.update(qn_eval_results)
        results.update({"model_name": "baseline"})
        print("Baseline Results:", results)
        return results
    
    def run_adapters(self):
        all_results = []
        for model_id in self.models:
            self.swap_adapter(model_id)
            adapter_name = model_id.split("/")[-1]
            results = self.evaluate_with_img_embeds(self.model, 
                                                 model_name=adapter_name,
                                                 eval_loader=self.bench_loader,
                                                 save_dir=dirname(abspath(__file__)))
            
            if self.text_eval_ground_truth is None:
                qn_eval_results = self.evaluate_with_textual_questions(self.model, 
                                                                model_name=adapter_name,
                                                                input_mode="input_ids",
                                                                questions=DEFAULT_QUESTIONS,
                                                                ground_truth=self.text_eval_ground_truth,
                                                                eval_loader=self.bench_loader,
                                                                question_preprocessor=self.processor,
                                                                save_dir=dirname(abspath(__file__)))
                self.text_eval_ground_truth = qn_eval_results["text_eval_ground_truth"]
            else:
                qn_eval_results = self.evaluate_with_textual_questions(self.model, 
                                                                model_name=adapter_name,
                                                                input_mode="input_ids",
                                                                questions=DEFAULT_QUESTIONS,
                                                                ground_truth=self.text_eval_ground_truth,
                                                                eval_loader=self.bench_loader,
                                                                question_preprocessor=self.processor,
                                                                save_dir=dirname(abspath(__file__)))
            qn_eval_results.pop("text_eval_ground_truth")

            results.update(qn_eval_results)
            results.update({"model_name": adapter_name})
            print(f"{adapter_name} Results:", results)
            all_results.append(results)
        return all_results
            
    
    def swap_adapter(self, adapter_id: str | PathLike):
        # Reload baseline model
        del self.model
        self.model = CLIPModel.from_pretrained(self.baseline_model)
        self.model.to(self.device)
        text_model_adapter_id = join(adapter_id, "text_model")
        if exists(text_model_adapter_id):
            adapted_text_model = PeftModel.from_pretrained(model=self.model.text_model, 
                                                           model_id=text_model_adapter_id)
            self.model.text_model = adapted_text_model
        text_projection_id = join(adapter_id, "text_projection.pt")

        if exists(text_projection_id):
            loaded_dict = torch.load(text_projection_id, map_location=self.device, weights_only=True)
            for name, param in self.model.text_projection.named_parameters():
                param.data = loaded_dict[name]
        

    @staticmethod
    def evaluate_with_img_embeds(model: CLIPModel, 
                              model_name: str,
                              eval_loader: torch.utils.data.DataLoader, 
                              top_k: int = 1,
                              save_dir: str = None):
        '''
            Evaluate the model using image retrieval.
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

                # Here we are looking the the logits ROWWISE, each row for a sensor input
                # and we want to see which image is retrieved by the model
                logits_per_text = torch.matmul(text_embeds, 
                                                image_embeds.t().to(text_embeds.device)) \
                                                * model.logit_scale.exp().to(text_embeds.device)
                metric_acc[0] += clip_loss(logits_per_text)
                metric_acc[1] += recall_at_k(logits_per_text, k=top_k, dim=1)
                metric_acc[2] += recall_at_k(logits_per_text, k=top_k, dim=0)
                probs = torch.nn.functional.softmax(logits_per_text, dim=1)
                assert torch.allclose(probs.sum(dim=1), 
                                      torch.tensor(1.0, device=probs.device)), \
                        "Probabilities should sum to 1" 

                # Boost the value for top-k to 1.0 for better visualization
                probs = boost_top_k(probs, k=top_k, dim=1)
 
                if save_dir is not None:

                    torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)
                    
                    # print(probs)
                    # Ensure the loader is of type EmbedDataset
                    images, texts = eval_loader.dataset.get_dataset_representation()
                    plot_heatmap(probs.cpu().round(decimals=2), 
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
    
    @staticmethod
    def evaluate_with_textual_questions(model: CLIPModel, 
                                    model_name: str,
                                    input_mode: ["image_embeds", "input_ids"],
                                    eval_loader: torch.utils.data.DataLoader, 
                                    questions: list[str] = DEFAULT_QUESTIONS,
                                    question_preprocessor: CLIPProcessor = \
                                        CLIPProcessor.from_pretrained(DEFAULT_BASE_MODEL_ID),
                                    ground_truth: torch.Tensor = None,
                                    save_dir: str = None):
        '''
            Evaluate the model in the form of question answering.
            eval_loader: __getitem__ should return (image_embedding, tokenized_caption)
            questions: List of textual questions to evaluate the model. Each question should consists of 
                        4 mutually exclusive options (Blank question is fine). It is eseentially classification 
                        with prompt engineering applied to class labels.
            ground_truth: Ground truth option for each question. It should be a tensor of size (len(questions),len(dataset))
                            with each entry being the index of the correct option for each question (0 to 3).
        '''
        assert type(eval_loader.dataset).__name__ == "EmbedDataset", "Loader must be of type EmbedDataset"
        assert len(eval_loader) == 1, "Batch size of the benchmark dataset must be equal to the dataset size"
        assert input_mode in ["image_embeds", "input_ids"], "Input mode must be either image_embeds or input_ids"
        assert len(questions) % 4 == 0, "Each question should have 4 options"
        model.eval()
        
        with torch.no_grad():

            metric_acc = None

            # Precompute all questions embeddings
            inputs = question_preprocessor(text=questions, images=None, return_tensors="pt", padding=True, do_rescale=False)
            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

            text_embeds = model.get_text_features(input_ids=inputs["input_ids"])
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            for i, (image_embeds, input_ids) in enumerate(tqdm(eval_loader)):

                if ground_truth is None:
                    logits_per_image = torch.matmul(image_embeds, 
                                text_embeds.t().to(image_embeds.device)) \
                                * model.logit_scale.exp().to(image_embeds.device)
                    # Perform softmax on each group of rows in the logits tensor.
                    ground_truth, softmax_result = col_partial_argmax(logits_per_image, group_size=4)

                    if save_dir is not None:
                   
                        # The image is used as the input for better visualization and since it should be entrywise 
                        # aligned to each set of sensor readings in the dataset anyway.
                        images, _ = eval_loader.dataset.get_dataset_representation()
                        plot_heatmap(softmax_result.t().cpu().round(decimals=2), 
                                    images, 
                                    questions,
                                    title_class="Question Answering Logits(Grount Truth)",
                                    model_name=model_name,
                                    output_mode="save",
                                    save_dir=save_dir)
                    

                if input_mode == "image_embeds":
                    input_embeds = image_embeds
                    input_embeds.to(model.device)
                    del input_ids # Using image embeddings only, ignore sensor readings
                elif input_mode == "input_ids":
                    input_ids = input_ids.to(model.device)
                    input_embeds = model.get_text_features(input_ids=input_ids)
                    input_embeds = input_embeds / input_embeds.norm(p=2, dim=-1, keepdim=True)
                    del image_embeds # Using sensor readings only, ignore image embeddings

                # Here we are looking at the logits ROWWISE, each row for a sensor input
                # and we want to see which text is picked by the model
                logits_per_reading = torch.matmul(input_embeds, 
                                                text_embeds.t().to(input_embeds.device)) \
                                                * model.logit_scale.exp().to(input_embeds.device)
                
                predictions, softmax_result = col_partial_argmax(logits_per_reading, group_size=4)

                assert ground_truth.shape == predictions.shape, "Ground truth should have the same shape as predictions"
                metric_acc = torch.sum(predictions == ground_truth, dim=0).float() / predictions.shape[0]
 
                if save_dir is not None:
                   
                    # The image is used as the input for better visualization and since it should be entrywise 
                    # aligned to each set of sensor readings in the dataset anyway.
                    images, _ = eval_loader.dataset.get_dataset_representation()
                    plot_heatmap(softmax_result.t().cpu().round(decimals=2), 
                                 images, 
                                 questions,
                                 title_class="Question Answering Logits",
                                 model_name=model_name,
                                 output_mode="save",
                                 save_dir=save_dir)
                    
            res = {f"text_eval_Q{i}": acc for i, acc in enumerate(metric_acc.tolist())}
            res.update({"text_eval_ground_truth": ground_truth})
            return res


DEFAULT_CHECKPOINT_DIR = "./checkpoints"

lora_finetune = CLIPModelModifier(train_projection=True)
lora_finetune.setupTrainer()
lora_finetune.train()
del lora_finetune

lora_finetune = CLIPModelModifier(train_projection=False)
lora_finetune.setupTrainer()
lora_finetune.train()
del lora_finetune

# models = ["text_lora-finetuned", "text_projection_lora-finetuned"]
# models = [join(DEFAULT_CHECKPOINT_DIR, model) for model in models]

# runner = CLIPLoRABenchmarkRunner(models=models)
# final_results = runner.run_adapters()
# baseline_results = runner.run_baseline()
# final_results.append(baseline_results)

# df = pd.DataFrame(final_results)
# columns = [df.columns[-1]] + list(df.columns[:-1])
# df = df[columns]
# del df["probs"]
# pd.set_option("display.max_columns", None)
# print(df)

