# -*- coding: utf-8 -*-

"""Base class for all numerical embedding models."""

from abc import ABC, abstractmethod
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, FunctionTransformer
from torch import nn, FloatTensor
from typing import Union

from pykeen.triples import TriplesFactory


class NumericalEmbeddingModel(nn.Module, ABC):
    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int,
        initializer: Union[str, None],
        input_transformer: Union[str, None],
        p_norm: int,
        random_state: int,
        device: str,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.initializer = initializer
        self.input_transformer = input_transformer
        self.p_norm = p_norm
        self.n_num_rel = triples_factory.n_num_rel
        self.random_state = random_state
        self.device = device
        self.fit_input_transformers(triples_factory)
        return

    def fit_input_transformers(self, triples_factory: TriplesFactory) -> None:
        self.input_transformers = []
        r, t = triples_factory.mapped_triples.T[1:]
        # Loop over numerical relations
        for rel_idx in range(self.n_num_rel):
            # Init transformer
            if self.input_transformer == "quantile":
                transformer = QuantileTransformer(
                    n_quantiles=20, subsample=2 ** 100, random_state=self.random_state
                )
            elif self.input_transformer == "min_max":
                transformer = MinMaxScaler(clip=True)
            elif self.input_transformer == None:
                transformer = FunctionTransformer(func=None)
            # Fit transformer on data
            transformer.fit(t[r == rel_idx][:, None])
            self.input_transformers.append(transformer)
        return

    def forward(self, x: FloatTensor, rel_idx: int) -> FloatTensor:
        x = self.input_transformers[rel_idx].transform(x.cpu())
        x = FloatTensor(x).to(self.device)
        x = self.compute_embeddings(x, rel_idx)
        if self.p_norm != None:
            x = nn.functional.normalize(x, p=self.p_norm, dim=1)
        return x

    @abstractmethod
    def compute_embeddings(self, x: FloatTensor) -> FloatTensor:
        raise NotImplementedError
