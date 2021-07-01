#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : OntoNotes_trainer.py
@author: xiaoya li
@contact : xiaoya_li@shannonai.com
@date  : 2021/06/30 17:28
@version: 1.0
@desc  :
"""

import os
import re
import json
import argparse
import logging
from functools import partial
from collections import namedtuple

from datasets.collate_functions import collate_to_max_length
from datasets.ontonotes_ner_dataset import OntoNotesNERDataset
from models.modeling_glycebert import GlyceBertForTokenClassification
from utils.random_seed import set_random_seed
from metrics.ner import SpanF1ForNER

# enable reproducibility
# https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
set_random_seed(2333)

import torch
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class OntoNotesTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.entity_labels = OntoNotesNERDataset.get_labels()
        self.bert_dir = args.bert_path
        self.num_labels = len(self.entity_labels)
        self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=False,
                                                      num_labels=self.num_labels,
                                                      hidden_dropout_prob=self.args.hidden_dropout_prob)
        self.model = GlyceBertForTokenClassification.from_pretrained(self.bert_dir,
                                                                     config=self.bert_config,
                                                                     mlp=False if self.args.classifier=="single" else True)

        self.ner_evaluation_metric = SpanF1ForNER(entity_labels=self.entity_labels, save_prediction=self.args.save_ner_prediction)

        format = '%(asctime)s - %(name)s - %(message)s'
        logging.basicConfig(format=format, filename=os.path.join(self.args.save_path, "eval_result_log.txt"), level=logging.INFO)
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.args.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), lr=self.args.lr, eps=self.args.adam_epsilon, )
        elif self.args.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            raise ValueError("Please import the Optimizer first. ")
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (
                self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.no_lr_scheduler:
            return [optimizer]
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids):
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, pinyin_ids, attention_mask=attention_mask)

    def compute_loss(self, logits, labels, loss_mask=None):
        """
        Desc:
            compute cross entropy loss
        Args:
            logits: FloatTensor, shape of [batch_size, sequence_len, num_labels]
            labels: LongTensor, shape of [batch_size, sequence_len, num_labels]
            loss_mask: Optional[LongTensor], shape of [batch_size, sequence_len].
                1 for non-PAD tokens, 0 for PAD tokens.
        """
        loss_fct = CrossEntropyLoss()
        if loss_mask is not None:
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, pinyin_ids, labels = batch
        loss_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        sequence_logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids,)[0]
        loss = self.compute_loss(sequence_logits, labels, loss_mask=loss_mask)

        tf_board_logs = {
            "train_loss": loss,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"]
        }
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, pinyin_ids, gold_labels = batch
        batch_size, seq_len = input_ids.shape
        loss_mask = (input_ids != 0).long()
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        sequence_logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids,)[0]
        loss = self.compute_loss(sequence_logits, gold_labels, loss_mask=loss_mask)
        probabilities, argmax_labels = self.postprocess_logits_to_labels(sequence_logits.view(batch_size, seq_len, -1))
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=loss_mask)
        return {"val_loss": loss, "confusion_matrix": confusion_matrix}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        confusion_matrix = torch.stack([x[f"confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> valid_f1 is: {f1}")
        tensorboard_logs = {"val_loss": avg_loss, "val_f1": f1, }
        return {"val_loss": avg_loss, "val_log": tensorboard_logs, "val_f1": f1, "val_precision": precision, "val_recall": recall}

    def train_dataloader(self,) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self, ) -> DataLoader:
        return self.get_dataloader("dev")

    def _load_dataset(self, prefix="test"):
        dataset = OntoNotesNERDataset(directory=self.args.data_dir, prefix=prefix,
                                      vocab_file=os.path.join(self.args.bert_path, "vocab.txt"),
                                      max_length=self.args.max_length,
                                      config_path=os.path.join(self.args.bert_path, "config"),
                                      file_name = self.args.train_file_name if len(self.args.train_file_name) != 0 and prefix == "train" else "char.bmes")

        return dataset

    def get_dataloader(self, prefix="train", limit=None) -> DataLoader:
        """return {train/dev/test} dataloader"""
        dataset = self._load_dataset(prefix=prefix)

        if prefix == "train":
            batch_size = self.args.train_batch_size
            # small dataset like weibo ner, define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            batch_size = self.args.eval_batch_size
            data_sampler = SequentialSampler(dataset)

        # sampler option is mutually exclusive with shuffle
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                                drop_last=False)

        return dataloader

    def test_dataloader(self, ) -> DataLoader:
        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        input_ids, pinyin_ids, gold_labels = batch
        sequence_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        sequence_logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids,)[0]
        probabilities, argmax_labels = self.postprocess_logits_to_labels(sequence_logits.view(batch_size, seq_len, -1))
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=sequence_mask)
        return {"confusion_matrix": confusion_matrix}

    def test_epoch_end(self, outputs):
        confusion_matrix = torch.stack([x[f"confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        if self.args.save_ner_prediction:
            precision, recall, f1, entity_tuple = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative, prefix="test")
            gold_entity_lst, pred_entity_lst = entity_tuple
            self.save_predictions_to_file(gold_entity_lst, pred_entity_lst)
        else:
            precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        tensorboard_logs = {"test_f1": f1,}
        self.result_logger.info(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} ")
        return {"test_log": tensorboard_logs, "test_f1": f1, "test_precision": precision, "test_recall": recall}

    def postprocess_logits_to_labels(self, logits):
        """input logits should in the shape [batch_size, seq_len, num_labels]"""
        probabilities = F.softmax(logits, dim=2) # shape of [batch_size, seq_len, num_labels]
        argmax_labels = torch.argmax(probabilities, 2, keepdim=False) # shape of [batch_size, seq_len]
        return probabilities, argmax_labels

    def save_save_predictions_to_file(self, gold_entity_lst, pred_entity_lst, prefix="test"):
        dataset = self._load_dataset(prefix=prefix)
        data_items = dataset.data_items

        save_file_path = os.path.join(self.args.save_path, "test_predictions.txt")
        print(f"INFO -> write predictions to {save_file_path}")
        with open(save_file_path, "w") as f:
            for gold_label_item, pred_label_item, data_item in zip(gold_entity_lst, pred_entity_lst, data_items):
                data_tokens = data_item[0]
                f.write("=!"* 20+"\n")
                f.write("".join(data_tokens)+"\n")
                f.write(gold_label_item+"\n")
                f.write(pred_label_item+"\n")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", type=str, help="bert config file")
    parser.add_argument("--train_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, )
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--classifier", type=str, default="single")
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--train_file_name", default="", type=str, help="use for truncated train sets.")
    parser.add_argument("--save_ner_prediction", action="store_true", help="only work for test.")
    parser.add_argument("--path_to_model_hparams_file", default="", type=str, help="use for evaluation")
    parser.add_argument("--checkpoint_path", default="", type=str, help="use for evaluation.")

    return parser


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = OntoNotesTask(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, "checkpoint", "{epoch}",),
        save_top_k=args.save_topk,
        save_last=False,
        monitor="val_f1",
        mode="max",
        verbose=True,
        period=-1,
    )

    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log')

    # save args
    with open(os.path.join(args.save_path, "checkpoint", "args.json"), "w") as f:
        args_dict = args.__dict__
        del args_dict["tpu_cores"]
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger,
                                         deterministic=True)
    trainer.fit(model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.save_path)
    model.result_logger.info("=&"*20)
    model.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    checkpoint = torch.load(path_to_best_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    trainer.test(model)
    model.result_logger.info("=&"*20)


def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt"):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN=re.compile(r"val_f1 reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN=re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/glyce/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_f1_on_dev = 0
    best_checkpoint_on_dev = 0
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("val_f1 reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(" as top", "")

        if current_f1 >= best_f1_on_dev:
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def evaluate():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = OntoNotesTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                            hparams_file=args.path_to_model_hparams_file,
                                                            map_location=None,
                                                            batch_size=1)
    trainer = Trainer.from_argparse_args(args, deterministic=True)

    trainer.test(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()



