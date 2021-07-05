#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : BQ_trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/1/21 14:45
@version: 1.0
@desc  : code for BQ task
"""
import argparse
import json
import os
import random
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

from datasets.collate_functions import collate_to_max_length
from datasets.spm_dataset import SPMDataset
from models.modeling_glycebert import GlyceBertForSequenceClassification
from utils.random_seed import set_random_seed

set_random_seed(2333)


class BQTask(pl.LightningModule):

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=False)
        self.model = GlyceBertForSequenceClassification.from_pretrained(self.bert_dir)

        self.loss_fn = CrossEntropyLoss()
        self.acc = pl.metrics.Accuracy(num_classes=self.bert_config.num_labels)
        gpus_string = self.args.gpus if not self.args.gpus.endswith(',') else self.args.gpus[:-1]
        self.num_gpus = len(gpus_string.split(","))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids):
        """"""
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, pinyin_ids, attention_mask=attention_mask)

    def compute_loss_and_acc(self, batch):
        input_ids, pinyin_ids, labels = batch
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        y = labels.view(-1)
        y_hat = self.forward(
            input_ids=input_ids,
            pinyin_ids=pinyin_ids
        )
        # compute loss
        loss = self.loss_fn(y_hat[0], y)
        # compute acc
        predict_scores = F.softmax(y_hat[0], dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        acc = self.acc(predict_labels, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        return {'val_loss': loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean() / self.num_gpus
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print(avg_loss, avg_acc)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        dataset = SPMDataset(data_path=os.path.join(self.args.data_dir, prefix + '.tsv'),
                             chinese_bert_path=self.args.bert_path,
                             max_length=self.args.max_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
            drop_last=False
        )
        return dataloader

    def test_dataloader(self):
        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        return {'test_loss': loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = torch.stack([x['test_acc'] for x in outputs]).mean() / self.num_gpus
        tensorboard_logs = {'test_loss': test_loss, 'test_acc': test_acc}
        print(test_loss, test_acc)
        return {'test_loss': test_loss, 'log': tensorboard_logs}


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=3, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--checkpoint_path", type=str, help="train checkpoint")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--mode", default='train', type=str, help="train or evaluate")
    parser.add_argument("--warmup_proporation", default=0.01, type=float, help="warmup proporation")
    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = BQTask(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, 'checkpoint', '{epoch}-{val_loss:.4f}-{val_acc:.4f}'),
        save_top_k=args.save_topk,
        save_last=False,
        monitor="val_acc",
        mode="max",
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log'
    )

    # save args
    with open(os.path.join(args.save_path, 'checkpoint', "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         distributed_backend="ddp",
                                         logger=logger)

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
