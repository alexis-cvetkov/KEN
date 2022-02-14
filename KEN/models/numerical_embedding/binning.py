# -*- coding: utf-8 -*-

"""A class that maps numerical values to bins, and learn an embedding for each bin."""

from pykeen.nn.init import xavier_uniform_
import torch
from torch import nn, FloatTensor
from torch.nn.init import normal_

from .base import NumericalEmbeddingModel


class BinnedEmbedding(NumericalEmbeddingModel):
    def __init__(self, n_bins: int, **kwargs) -> None:

        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.boundaries = torch.linspace(0, 1, self.n_bins + 1, device=self.device)[
            1:-1
        ]
        torch.manual_seed(self.random_state)
        self.num_embeddings = nn.Embedding(
            num_embeddings=self.n_bins * self.n_num_rel,
            embedding_dim=self.embedding_dim,
        )
        # If needed, init embeddings
        if self.initializer == "TransE":
            xavier_uniform_(self.num_embeddings.weight)
        elif self.initializer == "MuRE":
            normal_(self.num_embeddings.weight, std=1e-3)
        return

    def compute_embeddings(self, x: FloatTensor, rel_idx: int) -> FloatTensor:
        indices = rel_idx * self.n_bins + torch.bucketize(x[:, 0], self.boundaries)
        return self.num_embeddings(indices)
