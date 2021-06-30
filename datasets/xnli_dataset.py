#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : xnli_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/6/29 23:59
@version: 1.0
@desc  :  Dataset for natural language inference
"""

from functools import partial

import torch
from torch.utils.data import DataLoader

from datasets.chinese_bert_dataset import ChineseBertDataset
from datasets.collate_functions import collate_to_max_length


class XNLIDataset(ChineseBertDataset):

    def __init__(self, data_path, chinese_bert_path, max_length: int = 512):
        super().__init__(data_path, chinese_bert_path, max_length)
        self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2, "contradictory": 2}

    def get_lines(self):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        return lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        first, second, third = line.strip().split('\t', 2)
        first_output = self.tokenizer.encode(first, add_special_tokens=False)
        first_pinyin_tokens = self.convert_sentence_to_pinyin_ids(first, first_output)
        second_output = self.tokenizer.encode(second, add_special_tokens=False)
        second_pinyin_tokens = self.convert_sentence_to_pinyin_ids(second, second_output)
        label = self.label_map[third]
        # convert sentence to id
        bert_tokens = first_output.ids + [102] + second_output.ids
        pinyin_tokens = first_pinyin_tokens + [[0] * 8] + second_pinyin_tokens
        if len(bert_tokens) > self.max_length - 2:
            bert_tokens = bert_tokens[:self.max_length - 2]
            pinyin_tokens = pinyin_tokens[:self.max_length - 2]

        # assert
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # 转化list为tensor
        input_ids = torch.LongTensor([101] + bert_tokens + [102])
        pinyin_ids = torch.LongTensor([[0] * 8] + pinyin_tokens + [[0] * 8]).view(-1)
        label = torch.LongTensor([int(label)])
        return input_ids, pinyin_ids, label


def unit_test():
    data_path = "/data/nfsdata2/sunzijun/glyce/tasks/XNLI/xnli_train"
    model_path = "/data/nfsdata2/sunzijun/glyce/best/ChineseBERT-base"
    dataset = XNLIDataset(data_path=data_path,
                          chinese_bert_path=model_path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0])
    )
    for input_ids, pinyin_ids, label in dataloader:
        bs, length = input_ids.shape
        print(input_ids.shape)
        print(pinyin_ids.reshape(bs, length, -1).shape)
        print(label.view(-1).shape)
        print()


if __name__ == '__main__':
    unit_test()
