#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : bert_weibo_ner_dataset.py
@author: xiaoya li
@contact : xiaoya_li@shannonai.com
@date  : 2021/01/20 15:50
@version: 1.0
@desc  :
"""

import os
import torch
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer


class BertWeiboNERDataset(Dataset):
    def __init__(self, data_dir, prefix, vocab_file, max_length=512, file_name="all.bmes"):
        """
        Args:
            data_dir: str, path to data directory
            prefix: str, one of [train, dev, test]
            vocab_file: str, path to the vocab file when pre-training
            max_length: int,
        """
        super().__init__()
        self.max_length = max_length
        data_file_path = os.path.join(data_dir, "{}.{}".format(prefix, file_name))
        self.data_items = BertWeiboNERDataset.read_conll(data_file_path)
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.label_to_idx = {label_item: label_idx for label_idx, label_item in enumerate(BertWeiboNERDataset.get_labels())}

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        # tokenizer will process [CLS] + <sentence> + [SEP]
        data_item = self.data_items[idx]
        token_sequence, label_sequence = data_item[0], data_item[1]
        label_sequence = [self.label_to_idx[label_item] for label_item in label_sequence]
        token_sequence = "".join(token_sequence[: self.max_length - 2])
        label_sequence = label_sequence[: self.max_length - 2]
        #
        tokenizer_output = self.tokenizer.encode(token_sequence)
        # example of tokenizer_output ->
        # Encoding(num_tokens=77, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
        bert_tokens = tokenizer_output.ids
        label_sequence = self._update_labels_using_tokenize_offsets(tokenizer_output.offsets, label_sequence)
        assert len(bert_tokens) == len(label_sequence)

        input_ids = torch.LongTensor(bert_tokens)
        # token_type_ids are used to indicate whether this is the first sequence or the second sequence.
        # for single sequence, token_type_ids should equal to 0.
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        label_sequence = torch.LongTensor(label_sequence)
        return input_ids, attention_mask, token_type_ids, label_sequence

    def _update_labels_using_tokenize_offsets(self, offsets, original_sequence_labels):
        """part of offset sequence [(51, 52), (52, 54)] -> (token index after tokenized, original token index)"""
        update_sequence_labels = []
        for offset_idx, offset_item in enumerate(offsets):
            if offset_idx == 0 or offset_idx == (len(offsets) - 1) :
                continue
            update_index, origin_index = offset_item
            current_label = original_sequence_labels[origin_index-1]
            update_sequence_labels.append(current_label)
        update_sequence_labels = [self.label_to_idx["O"]] + update_sequence_labels + [self.label_to_idx["O"]]
        return update_sequence_labels

    @classmethod
    def get_labels(cls, ):
        """gets the list of labels for this data set."""
        return ["O", "B-GPE.NAM", "M-GPE.NAM", "E-GPE.NAM", "B-GPE.NOM", "E-GPE.NOM", \
                "B-LOC.NAM", "M-LOC.NAM", "E-LOC.NAM", "B-LOC.NOM", "M-LOC.NOM", "E-LOC.NOM", \
                "B-ORG.NAM", "M-ORG.NAM", "E-ORG.NAM", "B-ORG.NOM", "M-ORG.NOM", "E-ORG.NOM", \
                "B-PER.NAM", "M-PER.NAM", "E-PER.NAM", "B-PER.NOM", "M-PER.NOM", "E-PER.NOM", "S-GPE.NAM", \
                "S-LOC.NOM", "S-PER.NAM", "S-PER.NOM"]

    @staticmethod
    def read_conll(input_file, delimiter=" "):
        """load ner dataset from CoNLL-format files."""
        dataset_item_lst = []
        with open(input_file, "r", encoding="utf-8") as r_f:
            datalines = r_f.readlines()

        cached_token, cached_label = [], []
        for idx, data_line in enumerate(datalines):
            data_line = data_line.strip()
            if idx != 0 and len(data_line) == 0:
                dataset_item_lst.append([cached_token, cached_label])
                cached_token, cached_label = [], []
            else:
                token_label = data_line.split(delimiter)
                token_data_line, label_data_line = token_label[0], token_label[1]
                cached_token.append(token_data_line)
                cached_label.append(label_data_line)
        return dataset_item_lst
