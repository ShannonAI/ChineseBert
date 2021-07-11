#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : cmrc_2018_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/5/29 16:46
@version: 1.0
@desc  : 
"""
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tasks.CMRC.processor import read_squad_examples, convert_examples_to_features


class CMRC2018Dataset(Dataset):

    def __init__(self, directory, prefix):
        super().__init__()
        file = os.path.join(directory, prefix + '.json')
        with open(file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.data['input_ids'][idx])
        pinyin_ids = torch.LongTensor(self.data['pinyin_ids'][idx]).view(-1)
        input_mask = torch.LongTensor(self.data['input_mask'][idx])
        span_mask = torch.LongTensor(self.data['span_mask'][idx])
        segment_ids = torch.LongTensor(self.data['segment_ids'][idx])
        start = torch.LongTensor([self.data['start'][idx]])
        end = torch.LongTensor([self.data['end'][idx]])

        return input_ids, pinyin_ids, input_mask, span_mask, segment_ids, start, end


class CMRC2018EvalDataset(Dataset):

    def __init__(self, bert_path, test_file):
        super().__init__()
        self.examples = read_squad_examples(input_file=test_file, is_training=False)
        vocab_file = os.path.join(bert_path, 'vocab.txt')
        self.samples = convert_examples_to_features(
            bert_path=bert_path,
            examples=self.examples,
            max_seq_length=512,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            vocab_file=vocab_file,
            do_lower_case=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.samples[idx].input_ids)
        pinyin_ids = torch.LongTensor(self.samples[idx].pinyin_ids).view(-1)
        input_mask = torch.LongTensor(self.samples[idx].input_mask)
        span_mask = torch.LongTensor(self.samples[idx].input_span_mask)
        segment_ids = torch.LongTensor(self.samples[idx].segment_ids)
        unique_ids = torch.LongTensor([self.samples[idx].unique_id])
        indexes = torch.LongTensor([idx])
        return input_ids, pinyin_ids, input_mask, span_mask, segment_ids, unique_ids, indexes


def unit_test():
    root_path = "/home/sunzijun/cmrc"
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"
    prefix = "train"
    dataset = CMRC2018Dataset(directory=root_path, prefix=prefix)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=True
    )
    for input_ids, pinyin_ids, input_mask, span_mask, segment_ids, start, end in tqdm(dataloader):
        bs, length = input_ids.shape
        # print(input_ids.shape)
        # print(pinyin_ids.reshape(bs, length, -1).shape)
        # print(start.view(-1).shape)
        # print(end.view(-1).shape)
        # print()


def eval():
    root_path = "/data/nfsdata2/sunzijun/squad-style-data"
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"
    prefix = "dev"
    dataset = CMRC2018EvalDataset(directory=root_path, prefix=prefix, vocab_file=vocab_file)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=True
    )
    for input_ids, pinyin_ids, input_mask, span_mask, segment_ids, unique_ids, indexes in tqdm(dataloader):
        bs, length = input_ids.shape
        # print(input_ids.shape)
        # print(pinyin_ids.reshape(bs, length, -1).shape)
        # print(start.view(-1).shape)
        # print(end.view(-1).shape)
        # print()


if __name__ == '__main__':
    eval()
    unit_test()
