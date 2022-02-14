# -*- coding: utf-8 -*-

"""Embed numerical values with a single linear layer."""

from pykeen.nn.init import xavier_uniform_
import torch
from torch import nn, FloatTensor
from torch.nn.init import normal_
from typing import Union

from .base import NumericalEmbeddingModel


class LinearEmbedding(NumericalEmbeddingModel):
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
                nn.Sequential(
                    nn.Linear(
                        in_features=1,
                        out_features=self.embedding_dim,
                        device=self.device,
                    ),
                    activations[activation_function],
                )
                for _ in range(self.n_num_rel)
            ]
        )
        # If needed, init weight and bias
        if self.initializer == "xavier_uniform":
            for k in range(self.n_num_rel):
                xavier_uniform_(self.models[k][0].weight)
                xavier_uniform_(self.models[k][0].bias)
        elif self.initializer == "MuRE":
            for k in range(self.n_num_rel):
                normal_(self.models[k][0].bias, std=1e-3)
                e1 = self.models[k][0].bias.data
                e2 = torch.randn(self.embedding_dim, device=self.device) * 1e-3
                self.models[k][0].weight.data[:, 0] = e2 - e1
        return

    def compute_embeddings(self, x: FloatTensor, rel_idx: int) -> FloatTensor:
        return self.models[rel_idx](x)
