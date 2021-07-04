#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : data_generate.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/5/29 22:34
@version: 1.0
@desc  : 
"""
import argparse
import json
import os

from tasks.CMRC.processor import read_squad_examples, convert_examples_to_features


def extract_data(samples):
    data = {
        'input_ids': [],
        'pinyin_ids': [],
        'input_mask': [],
        'span_mask': [],
        'segment_ids': [],
        'start': [],
        'end': []
    }
    for sample in samples:
        data['input_ids'].append(sample.input_ids)
        data['pinyin_ids'].append(sample.pinyin_ids)
        data['input_mask'].append(sample.input_mask)
        data['span_mask'].append(sample.input_span_mask)
        data['segment_ids'].append(sample.segment_ids)
        data['start'].append(sample.start_position)
        data['end'].append(sample.end_position)
    return data


def generate_data(bert_path, data_dir, output_dir):
    vocab_file = os.path.join(bert_path, "vocab.txt")
    do_lower_case = False

    train_file = os.path.join(data_dir, "train.json")
    dev_file = os.path.join(data_dir, "dev.json")
    test_file = os.path.join(data_dir, "test.json")

    # train data
    train_examples = read_squad_examples(input_file=train_file, is_training=True)
    train_samples = convert_examples_to_features(
        bert_path=bert_path,
        examples=train_examples,
        max_seq_length=512,
        doc_stride=128,
        max_query_length=64,
        is_training=True,
        vocab_file=vocab_file,
        do_lower_case=do_lower_case)
    train_data = extract_data(train_samples)
    print(train_examples[:3])
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f)

    # dev data
    dev_examples = read_squad_examples(input_file=dev_file, is_training=True)
    dev_samples = convert_examples_to_features(
        bert_path=bert_path,
        examples=dev_examples,
        max_seq_length=512,
        doc_stride=128,
        max_query_length=64,
        is_training=True,
        vocab_file=vocab_file,
        do_lower_case=do_lower_case)
    dev_data = extract_data(dev_samples)
    print(dev_examples[:3])
    with open(os.path.join(output_dir, 'dev.json'), 'w') as f:
        json.dump(dev_data, f)


def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert path")
    parser.add_argument("--data_dir", required=True, type=str, help="input data dir")
    parser.add_argument("--output_dir", required=True, type=str, help="output data dir")
    args = parser.parse_args()
    generate_data(args.bert_path, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
