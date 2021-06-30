#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2021.01.25
# files: nn_modules.py
#

import torch.nn as nn

class BertMLP(nn.Module):
    def __init__(self, config,):
        super().__init__()
        self.dense_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_to_labels_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.activation = nn.Tanh()

    def forward(self, sequence_hidden_states):
        sequence_output = self.dense_layer(sequence_hidden_states)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dense_to_labels_layer(sequence_output)
        return sequence_output
