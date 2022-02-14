# -*- coding: utf-8 -*-

"""Implementation of MuRE."""

import torch
from torch import LongTensor, DoubleTensor, FloatTensor
from typing import Union, Tuple, cast
from pykeen.models import MuRE as BaseModel
from pykeen.typing import HeadRepresentation, RelationRepresentation, TailRepresentation

from ..numerical_embedding import NumericalEmbeddingModel


class MuRE(BaseModel):
    def __init__(
        self,
        numerical_embedding_model: Union[NumericalEmbeddingModel, None],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_emb_model = numerical_embedding_model
        # Learn a fixed bias for each numerical relation, initialized with zeros.
        if self.num_emb_model != None:
            self.num_rel_bias = torch.nn.Parameter(
                torch.zeros(self.num_emb_model.n_num_rel, device=self.device)
            )
        return

    def _get_representations(self, h_indices, r_indices, t_indices):
        """Get representations for head, relation and tails, in canonical shape."""
        if self.num_emb_model == None:
            h, r, t = [
                [
                    representation.get_in_more_canonical_shape(dim=dim, indices=indices)
                    for representation in representations
                ]
                for dim, indices, representations in (
                    ("h", h_indices, self.entity_representations),
                    ("r", r_indices, self.relation_representations),
                    ("t", t_indices, self.entity_representations),
                )
            ]
        else:
            h_indices, r_indices = h_indices.long(), r_indices.long()
            h, r = [
                [
                    representation.get_in_more_canonical_shape(dim=dim, indices=indices)
                    for representation in representations
                ]
                for dim, indices, representations in (
                    ("h", h_indices, self.entity_representations),
                    ("r", r_indices, self.relation_representations),
                )
            ]
            # Get representations for tails
            t = [torch.empty(hk.shape, device=self.device) for hk in h]
            mask_num = r_indices < self.num_emb_model.n_num_rel
            idx_num, idx_cat = torch.where(mask_num)[0], torch.where(~mask_num)[0]
            t_indices_cat = t_indices[idx_cat].long()
            t_cat = [
                representation.get_in_more_canonical_shape(
                    dim="t", indices=t_indices_cat
                )
                for representation in self.entity_representations
            ]
            for k in range(3):
                t[k][idx_cat] = t_cat[k]
            r_idx_num, t_idx_num = r_indices[mask_num], t_indices[mask_num]
            for num_rel_idx in range(self.num_emb_model.n_num_rel):
                # Add numerical embeddings
                mask_num_k = r_idx_num == num_rel_idx
                if mask_num_k.any():
                    idx_num_k, val_num_k = idx_num[mask_num_k], t_idx_num[mask_num_k]
                    t[0][idx_num_k] = self.num_emb_model(
                        val_num_k[:, None].float(), num_rel_idx
                    ).view(-1, 1, 1, 1, self.num_emb_model.embedding_dim)
                    # Add biases
                    t[1][idx_num_k] = self.num_rel_bias[num_rel_idx]
                    t[2][idx_num_k] = self.num_rel_bias[num_rel_idx]
        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (h, r, t)),
        )
