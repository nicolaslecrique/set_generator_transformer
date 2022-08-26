import random

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F, Module


# generate sequence following a fitted language model
# see https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
class SequenceGenerator:

    nb_tokens: int
    seq_to_seq_model: Module
    end_of_sequence_token_id: int
    start_of_sequence_token_ids: [int]

    def __init__(
            self,
            seq_to_seq_model: nn.Module,
            start_of_sequence_token_ids: [int],
            end_of_sequence_token_id: int,
            nb_tokens: int
    ):
        seq_to_seq_model.eval()
        self.end_of_sequence_token_id = end_of_sequence_token_id
        self.start_of_sequence_token_ids = start_of_sequence_token_ids
        self.seq_to_seq_model = seq_to_seq_model
        self.nb_tokens = nb_tokens

    def generate(self, max_size: int, topk: int, sequence_beginning: [int]):
        self.seq_to_seq_model.eval()

        # init with <sos> token
        if len(sequence_beginning) == 0:
            sequence_beginning = [random.choice(self.start_of_sequence_token_ids)]

        # format [size_sequence, 1=batch_size]
        current_seq = torch.from_numpy(np.asarray(sequence_beginning)).view(-1, 1)

        while current_seq[-1, 0] != self.end_of_sequence_token_id and len(current_seq) < max_size:
            #seq_output [size_sequence, 1=batch_size, vocab_size]
            seq_output = self.seq_to_seq_model.forward(current_seq)
            # get model output for last token
            next_token_linear_output = seq_output[-1, 0, :]
            # transform model output into proba
            next_token_proba = F.softmax(next_token_linear_output)
            # restrict possible tokens to topk tokens
            topk_proba, topk_idx = torch.topk(next_token_proba, self.nb_tokens if len(current_seq) == 1 else topk)

            # choose one value following multinomial proba, https://arxiv.org/pdf/1805.04833.pdf
            next_token_generated_idx_in_topk = torch.multinomial(topk_proba, num_samples=1)  # return one-dim tensor with one element
            next_token_generated = topk_idx[next_token_generated_idx_in_topk]

            # prevent token duplication
            if next_token_generated in current_seq:
                continue

            # set next_token_generated to same format as current_seq to call torch.cat [1=added_size_in_seq,1=batch_size]
            next_token_generated_seq_format = next_token_generated.unsqueeze(0)
            # concat generated token to previous token
            current_seq = torch.cat([current_seq, next_token_generated_seq_format])

        # flatten final seq
        final_seq = current_seq.view(1, -1).squeeze()
        return final_seq
