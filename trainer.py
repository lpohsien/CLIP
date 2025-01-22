from transformers import CLIPTextModel, CLIPVisionModel
import torch

from tqdm import tqdm
import wandb

from utils import recall_at_k, check_memory_usage, clip_loss


default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIPTrainLogger:
    def __init__(self, use_wandb=False):
        self.minibatch_losses = torch.zeros(1)
        self.epoch_losses = torch.zeros(1)
        self.minibatch_count = 0
        self.use_wandb = use_wandb

    def update_minibatch(self, outputs):
        if self.use_wandb:
            wandb.log({key: value.item() for key, value in outputs.items()})
        self.minibatch_count += 1
        self.minibatch_losses += outputs["loss"]

    def update_epoch(self, epoch_idx, eval_metrics=None):
        epoch_avg_training_loss = self.minibatch_losses / self.minibatch_count
        print(f"Epoch {epoch_idx} training_loss: {epoch_avg_training_loss}", end=" ")
        if eval_metrics is not None:
            for key, value in eval_metrics.items():
                print(f"{key}: {value}", end=" ")
        print()
        self.minibatch_count = 0


class CLIPTrainer:
    def __init__(self, 
                 model, 
                 optimizer,
                 train_loader, 
                 eval_loader,
                 args=None, 
                 loss_fn=clip_loss,
                 compute_metrics=recall_at_k,
                 use_wandb=False,
                 device=default_device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.args = args
        self.compute_metrics = compute_metrics
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.num_epochs = 10
        self.batch_size = 40
        self.use_wandb = use_wandb
        self.train_logger = CLIPTrainLogger()
        self.device = device
        self.compute_metrics = compute_metrics


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
            # self.train_logger.update_minibatch({"loss": loss, "logits_per_text": logits_per_text})
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

            # Guard against memory leaks and/or running on the wrong device
            ram_free, free_ram_free = check_memory_usage(15)


            