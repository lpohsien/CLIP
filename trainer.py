import torch
from os.path import dirname, abspath, join, exists
from os import mkdir
from peft import PeftModel
from tqdm import tqdm
import wandb
from safetensors.torch import save_file

from utils import recall_at_k, check_memory_usage, clip_loss


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIPTrainLogger:
    def __init__(self, use_wandb=False):
        self.minibatch_losses = 0
        self.epoch_losses = 0
        self.minibatch_count = 0
        self.use_wandb = use_wandb

    def update_minibatch(self, outputs):
        if self.use_wandb:
            wandb.log({key: value.item() for key, value in outputs.items()})
        self.minibatch_count += 1
        self.minibatch_losses += outputs["loss"].item()

    def update_epoch(self, epoch_idx, eval_metrics=None):
        epoch_avg_training_loss = self.minibatch_losses / self.minibatch_count
        print(f"Epoch {epoch_idx} training_loss: {epoch_avg_training_loss}", end=" ")
        if eval_metrics is not None:
            for key, value in eval_metrics.items():
                print(f"{key}: {value}", end=" ")
        print()

        self.minibatch_losses = 0
        self.minibatch_count = 0

def best_eval_loss(results: dict) -> float:
    if "eval_loss" not in results:
        raise KeyError("eval_loss is used as the metric for saving model ",
                       "but it is not part of the evaluation metrics.")
    return results["eval_loss"]

def best_eval_recall(results: dict) -> float:
    if "eval_per_text_accuracy" not in results:
        raise KeyError("eval_per_text_accuracy is used as the metric for saving model ",
                       "but it is not part of the evaluation metrics.")
    if "eval_per_image_accuracy" not in results:
        raise KeyError("eval_per_image_accuracy is used as the metric for saving model",
                       " but it is not part of the evaluation metrics.")
    return - results["eval_per_text_accuracy"] - results["eval_per_image_accuracy"]

class CLIPTrainer:
    def __init__(self, 
                 model, 
                 optimizer,
                 train_loader, 
                 eval_loader,
                 save_dir,
                 output_model_name,
                 train_projection=False,
                 args=None, 
                 overwrite_exisintg_checkpoint=True,                
                 loss_fn=clip_loss,
                 compute_metrics=recall_at_k,
                 metric_eval_func=best_eval_loss,
                 num_epochs=10,
                 use_wandb=False,
                 device=DEFAULT_DEVICE):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.args = args
        self.compute_metrics = compute_metrics
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.num_epochs = num_epochs
        self.use_wandb = use_wandb
        self.train_logger = CLIPTrainLogger()
        self.device = device
        self.compute_metrics = compute_metrics
        self.output_model_name = output_model_name
        self.save_dir = join(save_dir, self.output_model_name)
        self.overwrite_exisintg_checkpoint = overwrite_exisintg_checkpoint
        self.train_projection = train_projection
        self.eval_metric = metric_eval_func

        # Internal variables
        self.best_metric = torch.tensor(float("inf"), device=self.device)

        # Create directory for saving checkpoint
        if not exists(self.save_dir):
            print(f"Creating directory {self.save_dir}")
            mkdir(self.save_dir)
        else:
            if self.overwrite_exisintg_checkpoint:
                print("Overwriting existing checkpoint...")
            else:
                raise FileExistsError("Checkpoint already exists.",
                                      "Set overwrite_exisintg_checkpoint=True to overwrite it.")



    def train_from_img_embeds_one_epoch(self):
        '''
            Train the model for one epoch.
            train_loader: __getitem__ should return (image_embedding, tokenized_caption)
        '''
        self.model.train()
        for i, (image_embeds, input_ids) in enumerate(tqdm(self.train_loader)):

            self.optimizer.zero_grad()
            image_embeds = image_embeds.to(self.device)
            # image_embeds are already computed
            
            input_ids = input_ids.to(self.device)
            text_embeds = self.model.get_text_features(input_ids=input_ids)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)


            logits_per_text = torch.matmul(text_embeds, 
                                            image_embeds.t().to(text_embeds.device)) \
                                            * self.model.logit_scale.exp().to(text_embeds.device)
            
            loss = self.loss_fn(logits_per_text)
            loss.backward()
            self.optimizer.step()
            self.train_logger.update_minibatch({"loss": loss, "logits_per_text": logits_per_text})
            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            torch.clamp(self.model.logit_scale, 0, 4.6052)

        return 0

    def eval_from_img_embeds(self):
        '''
            Evaluate the model.
            eval_loader: __getitem__ should return (image_embedding, tokenized_caption)
        '''
        self.model.eval()
        with torch.no_grad():
            metric_acc = torch.zeros(3).to(self.device)
            for i, (image_embeds, input_ids) in enumerate(tqdm(self.eval_loader)):

                image_embeds = image_embeds.to(self.device)
                # image_embeds are already computed
            
                input_ids = input_ids.to(self.device)
                text_embeds = self.model.get_text_features(input_ids=input_ids)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

                logits_per_text = torch.matmul(text_embeds, 
                                                image_embeds.t().to(text_embeds.device)) \
                                                * self.model.logit_scale.exp().to(text_embeds.device)
                metric_acc[0] += self.loss_fn(logits_per_text)
                metric_acc[1] += self.compute_metrics(logits_per_text)
                metric_acc[2] += self.compute_metrics(logits_per_text.t())
            metric_acc /= len(self.eval_loader)
        metric_acc = metric_acc.tolist()
        return {"eval_loss": metric_acc[0], 
                "eval_per_text_accuracy": metric_acc[1], 
                "eval_per_image_accuracy": metric_acc[2]}
            
    
    def train_from_img_embeds(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}")
            self.train_from_img_embeds_one_epoch()
            eval_metrics = self.eval_from_img_embeds()
            self.train_logger.update_epoch(epoch, eval_metrics)
            if self.use_wandb:
                wandb.log(eval_metrics)

            curr_metric = self.eval_metric(eval_metrics)
            if curr_metric < self.best_metric:
                self.best_metric = curr_metric
                self.save_text_model_adpater()
                if self.train_projection:
                    self.save_text_projection_adapter()

            # Guard against memory leaks and/or running on the wrong device
            ram_free, free_ram_free = check_memory_usage(15)
        
        print(f"Training complete. Best eval metric: {self.best_metric}")

    def save_text_model_adpater(self):
        text_model_dir = join(self.save_dir, "text_model")
        self.model.text_model.save_pretrained(text_model_dir)
        print(f"Saved {self.output_model_name} to {text_model_dir}")

    def save_text_projection_adapter(self):
        torch.save(self.model.text_projection.state_dict(), 
                   join(self.save_dir, "text_projection.pt"))
        print(f"Saved {self.output_model_name} to {self.save_dir}")