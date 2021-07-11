#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : cmrc_trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/5/20 15:03
@version: 1.0
@desc  : 
"""
import argparse
import collections
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tokenizers import BertWordPieceTokenizer
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

from datasets.cmrc_2018_dataset import CMRC2018Dataset
from models.modeling_glycebert import GlyceBertForQuestionAnswering
from utils.random_seed import set_random_seed

set_random_seed(2333)
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


class CMRCTask(pl.LightningModule):

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
        self.model = GlyceBertForQuestionAnswering.from_pretrained(self.bert_dir)
        self.tokenizer = BertWordPieceTokenizer(os.path.join(self.args.bert_path, "vocab.txt"))

        gpus_string = self.args.gpus if not self.args.gpus.endswith(',') else self.args.gpus[:-1]
        self.num_gpus = len(gpus_string.split(","))
        self.query_map = {}
        self.result = {}

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

    def compute_loss_and_acc(self, batch):
        input_ids, pinyin_ids, input_mask, span_mask, segment_ids, start, end = batch
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        # attention mask
        attention_mask = (input_ids != 0).long()
        output = self.model(input_ids, pinyin_ids, attention_mask=attention_mask,
                            token_type_ids=segment_ids, start_positions=start, end_positions=end)
        return output

    def training_step(self, batch, batch_idx):
        """"""
        output = self.compute_loss_and_acc(batch)
        loss = output[0]
        tf_board_logs = {
            "train_loss": loss,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""
        output = self.compute_loss_and_acc(batch)
        loss = output[0]
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        print(avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        dataset = CMRC2018Dataset(directory=self.args.data_dir, prefix=prefix)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True
        )
        return dataloader


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=4, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--warmup_proporation", default=0.01, type=float, help="warmup proporation")
    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = CMRCTask(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, 'checkpoint', '{epoch}-{val_loss:.4f}'),
        save_top_k=args.save_topk,
        save_last=False,
        monitor="val_loss",
        mode="min",
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


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
