# -*- coding: utf-8 -*-

"""Implementation of TransE."""

import torch
from torch import LongTensor, DoubleTensor, FloatTensor
from typing import Union
from pykeen.models import TransE as BaseModel

from ..numerical_embedding import NumericalEmbeddingModel


class TransE(BaseModel):
    def __init__(
        self,
        numerical_embedding_model: Union[NumericalEmbeddingModel, None],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_emb_model = numerical_embedding_model
        return

    def score_hrt(self, hrt_batch: Union[LongTensor, DoubleTensor]) -> FloatTensor:
        # Get entity embeddings
        if self.num_emb_model == None:
            h = self.entity_embeddings(indices=hrt_batch[:, 0])
            r = self.relation_embeddings(indices=hrt_batch[:, 1])
            t = self.entity_embeddings(indices=hrt_batch[:, 2])
        # Get embeddings for entities and numerical values
        else:
            h = self.entity_embeddings(indices=hrt_batch[:, 0].long())
            r = self.relation_embeddings(indices=hrt_batch[:, 1].long())
            t = torch.empty(h.shape, device=self.device)
            r_idx, t_idx = hrt_batch[:, 1], hrt_batch[:, 2]
            mask_num = r_idx < self.num_emb_model.n_num_rel
            idx_num, idx_cat = torch.where(mask_num)[0], torch.where(~mask_num)[0]
            t[idx_cat] = self.entity_embeddings(t_idx[idx_cat].long())
            r_idx_num, t_idx_num = r_idx[mask_num], t_idx[mask_num]
            for num_rel_idx in range(self.num_emb_model.n_num_rel):
                mask_num_k = r_idx_num == num_rel_idx
                if mask_num_k.any():
                    idx_num_k, val_num_k = idx_num[mask_num_k], t_idx_num[mask_num_k]
                    t[idx_num_k] = self.num_emb_model(
                        val_num_k[:, None].float(), num_rel_idx
                    )
        # Compute scores
        # TODO: Use torch.dist
        return -torch.norm(h + r - t, dim=-1, p=self.scoring_fct_norm, keepdim=True)
