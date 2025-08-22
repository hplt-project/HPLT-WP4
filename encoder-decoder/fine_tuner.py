# from https://share.google/Xr3qC66s92x5ueFGN
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW

from ner_dataset import WikiAnnDataset

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparam, tokenizer, dataset):
        super(T5FineTuner, self).__init__()
        self.hparam = hparam
        self.dataset = dataset
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hparam.model_path, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.training_step_outputs = []   # save outputs in each batch to compute metric overall epoch
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch

    def is_logger(self):
        return True

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # https://stackoverflow.com/questions/73111496/pytorch-lightning-misconfiguration-exception-the-closure-hasnt-been-executed
        self.manual_backward(loss)
        
        optimizer = self.optimizers()

        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

        self.training_step_outputs.append({"loss":loss})
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def on_train_epoch_end(self):
        # https://stackoverflow.com/questions/70790473/pytorch-lightning-epoch-end-validation-epoch-end
        avg_train_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.val_step_outputs.append({"val_loss": loss})
        self.log('val_loss', loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.val_step_outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparam.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparam.learning_rate, eps=self.hparam.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = self.get_dataset(
            type_path="train", args=self.hparam
            )
        dataloader = DataLoader(train_dataset, batch_size=self.hparam.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=2)
        t_total = (
            (len(dataloader.dataset) //
             (self.hparam.train_batch_size * max(1, self.hparam.devices)))
            // self.hparam.gradient_accumulation_steps
            * float(self.hparam.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparam.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = self.get_dataset(
            type_path="validation", args=self.hparam,
            )
        return DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=2)

    def get_dataset(self, type_path, args):
        self.tokenizer.max_length = args.max_seq_length
        self.tokenizer.model_max_length = args.max_seq_length
        
        return WikiAnnDataset(tokenizer=self.tokenizer, dataset=self.dataset, type_path=type_path)