# -*- coding: utf-8 -*-

"""Embed numerical values with a single linear layer e(x) = ReLU(x.w + b).
The implementation is slightly different from linear.py. We learn two embeddings
(e1, e2)  and sum them with weights x and (1- x), then applies the ReLU activation:
e(x) = ReLU( x.e1 + (1 - x).e2 ). This formulation is equivalent to the first but
allows us to initialize the embeddings (e1, e2) in the same way than discrete
entity embeddings. We thus use this one in our experiments."""

from pykeen.nn.init import xavier_uniform_
import torch
from torch import nn, FloatTensor
from torch.nn.init import normal_, xavier_normal_
from typing import Union

from .base import NumericalEmbeddingModel


class LinearEmbedding2(NumericalEmbeddingModel):
    def __init__(self, activation_function: Union[str, None], **kwargs) -> None:

        super().__init__(**kwargs)
        self.activation_function = activation_function
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            None: nn.Identity(),
        }
        torch.manual_seed(self.random_state)
        self.models = nn.ModuleList(
            [
                nn.EmbeddingBag(
                    num_embeddings=2,
                    embedding_dim=self.embedding_dim,
                    mode="sum",
                    device=self.device,
                )
                for _ in range(self.n_num_rel)
            ]
        )
        # If needed, init weight and bias
        if self.initializer == "TransE":
            for k in range(self.n_num_rel):
                xavier_uniform_(self.models[k].weight)
        if self.initializer == "DistMult":
            for k in range(self.n_num_rel):
                xavier_normal_(self.models[k].weight)
        elif self.initializer == "MuRE":
            for k in range(self.n_num_rel):
                normal_(self.models[k].weight, std=1e-3)
        return

    def compute_embeddings(self, x: FloatTensor, rel_idx: int) -> FloatTensor:
        inputs = torch.zeros((x.shape[0], 2), dtype=torch.long, device=self.device)
        inputs[:, 1] = 1
        weights = torch.zeros_like(inputs, dtype=torch.float32, device=self.device)
        weights[:, 0] = x[:, 0]
        weights[:, 1] = 1 - x[:, 0]
        output = self.models[rel_idx](inputs, per_sample_weights=weights)
        if self.activation_function == "relu":
            output = nn.functional.relu(output)
        return output
