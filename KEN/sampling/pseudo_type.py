# -*- coding: utf-8 -*-

"""
Minor modification of PyKEEN PseudoTypedNegativeSampler to allow for numerical values.
"""

import torch

from pykeen.sampling import PseudoTypedNegativeSampler as BaseSampler
from pykeen.triples import CoreTriplesFactory

class PseudoTypedNegativeSampler(BaseSampler):

    def __init__(self, *, triples_factory: CoreTriplesFactory, **kwargs):
        super().__init__(triples_factory=triples_factory, **kwargs)
        return
    
    def corrupt_batch(self, positive_batch: torch.LongTensor):  # noqa: D102
        batch_size = positive_batch.shape[0]

        # shape: (batch_size, num_neg_per_pos, 3)
        negative_batch = positive_batch.unsqueeze(dim=1).repeat(1, self.num_negs_per_pos, 1)

        # Uniformly sample from head/tail offsets
        r = positive_batch[:, 1].long()
        start_heads = self.offsets[2 * r].unsqueeze(dim=-1)
        start_tails = self.offsets[2 * r + 1].unsqueeze(dim=-1)
        end = self.offsets[2 * r + 2].unsqueeze(dim=-1)
        num_choices = end - start_heads
        negative_ids = start_heads + (torch.rand(size=(batch_size, self.num_negs_per_pos)) * num_choices).long()

        # get corresponding entity
        entity_id = self.data[negative_ids]

        # and position within triple (0: head, 2: tail)
        triple_position = 2 * (negative_ids >= start_tails).long()

        # write into negative batch
        negative_batch[
            torch.arange(batch_size, device=negative_batch.device).unsqueeze(dim=-1),
            torch.arange(self.num_negs_per_pos, device=negative_batch.device).unsqueeze(dim=0),
            triple_position,
        ] = entity_id.type(negative_batch.dtype)

        return negative_batch