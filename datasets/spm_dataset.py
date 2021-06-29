#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : spm_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/1/21 15:00
@version: 1.0
@desc  : Dataset for sentence pair matching tasks
"""

from functools import partial

import torch
from torch.utils.data import DataLoader

from datasets.chinese_bert_dataset import ChineseBertDataset
from datasets.collate_functions import collate_to_max_length


class SPMDataset(ChineseBertDataset):

    def get_lines(self):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        return lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        third, first, second, _ = line.split('\t')
        first = first.replace(" ", "")
        second = second.replace(" ", "")
        first_output = self.tokenizer.encode(first, add_special_tokens=False)
        first_pinyin_tokens = self.convert_sentence_to_pinyin_ids(first, first_output)
        second_output = self.tokenizer.encode(second, add_special_tokens=False)
        second_pinyin_tokens = self.convert_sentence_to_pinyin_ids(second, second_output)
        label = third
        # convert sentence to id
        bert_tokens = first_output.ids + [102] + second_output.ids
        pinyin_tokens = first_pinyin_tokens + [[0] * 8] + second_pinyin_tokens
        if len(bert_tokens) > self.max_length - 2:
            bert_tokens = bert_tokens[:self.max_length - 2]
            pinyin_tokens = pinyin_tokens[:self.max_length - 2]

        # id nums should be same
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)

        # convert list to tensor
        input_ids = torch.LongTensor([101] + bert_tokens + [102])
        pinyin_ids = torch.LongTensor([[0] * 8] + pinyin_tokens + [[0] * 8]).view(-1)
        label = torch.LongTensor([int(label)])
        return input_ids, pinyin_ids, label


def unit_test():
    data_path = "/data/nfsdata2/sunzijun/glyce/tasks/BQ/dev.tsv"
    chinese_bert_path = "/data/nfsdata2/sunzijun/glyce/best/ChineseBERT-base"
    dataset = SPMDataset(data_path=data_path, chinese_bert_path=chinese_bert_path)

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
