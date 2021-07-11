#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : cmrc_evaluate.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/5/30 11:34
@version: 1.0
@desc  : 
"""
import argparse
import collections
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from tokenizers import BertWordPieceTokenizer
from torch.utils.data.dataloader import DataLoader
from transformers import BertConfig

from datasets.cmrc_2018_dataset import CMRC2018EvalDataset
from models.modeling_glycebert import GlyceBertForQuestionAnswering
from tasks.CMRC.processor import write_predictions
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
        self.all_results = []

    def compute_loss_and_acc(self, batch):
        input_ids, pinyin_ids, input_mask, span_mask, segment_ids, unique_ids, indexes = batch
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        # attention mask
        attention_mask = (input_ids != 0).long()
        output = self.model(input_ids, pinyin_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        return output, unique_ids, indexes

    def get_dataloader(self) -> DataLoader:
        """get training dataloader"""
        self.dataset = CMRC2018EvalDataset(bert_path=self.args.bert_path,
                                           test_file=self.args.test_file)
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True
        )
        return dataloader

    def test_dataloader(self):
        return self.get_dataloader()

    def test_step(self, batch, batch_idx):
        output, unique_ids, indexes = self.compute_loss_and_acc(batch)
        start_logits, end_logits = output[0], output[1]

        input_ids, pinyin_ids, input_mask, span_mask, segment_ids, unique_ids, indexes = batch
        for i in range(input_ids.shape[0]):
            self.all_results.append(
                RawResult(
                    unique_id=int(unique_ids[i][0]),
                    start_logits=start_logits[i].cpu().tolist(),
                    end_logits=end_logits[i].cpu().tolist()))

        return {'test_loss': 0}

    def test_epoch_end(self, outputs):
        eval_examples = self.dataset.examples
        eval_features = self.dataset.samples
        all_results = self.all_results
        n_best_size = 20
        max_answer_length = 30
        do_lower_case = False
        output_prediction_file = os.path.join(self.args.save_path, "test_predictions.json")
        output_nbest_file = os.path.join(self.args.save_path, "test_nbest_predictions.json")
        write_predictions(eval_examples, eval_features, all_results,
                          n_best_size, max_answer_length,
                          do_lower_case, output_prediction_file,
                          output_nbest_file)

        return {'test_loss': 0}


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--test_file", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--task", default='cmrc', type=str, help="checkpoint path")
    parser.add_argument("--pretrain_checkpoint", default="", type=str, help="train data path")
    parser.add_argument("--warmup_proporation", default=0.01, type=float, help="warmup proporation")
    return parser


def evaluate():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = CMRCTask(args)
    checkpoint = torch.load(args.pretrain_checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    trainer = Trainer.from_argparse_args(args, distributed_backend="ddp")
    trainer.test(model)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    evaluate()
