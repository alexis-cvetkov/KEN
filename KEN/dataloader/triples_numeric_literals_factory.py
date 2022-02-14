# -*- coding: utf-8 -*-

"""Modification of PyKEEN original TriplesNumericLiteralsFactory class."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

from pykeen.triples import TriplesNumericLiteralsFactory as BaseClass
from pykeen.triples import TriplesFactory
from sqlalchemy import literal


def create_matrix_of_literals(
    numeric_triples: pd.DataFrame,
    entity_to_id: dict,
    literals_to_id: dict,
) -> np.ndarray:
    """Create matrix of literals where each row corresponds to an entity
    and each column to a literal."""
    # Prepare literal matrix, set every literal to zero, and afterwards fill
    # in the corresponding value if available
    num_literals = np.zeros([len(entity_to_id), len(literals_to_id)], dtype=np.float32)
    # TODO vectorize code
    entity_idx, relation_idx = numeric_triples["head"], numeric_triples["rel"]
    literal_values = numeric_triples["tail"].astype(np.float32)
    num_literals[entity_idx, relation_idx] = literal_values
    return num_literals


class TriplesNumericLiteralsFactory(BaseClass, TriplesFactory):
    def __init__(
        self,
        *,
        triples_factory: TriplesFactory,
        numeric_triples: np.ndarray,
        num_rel_to_idx: dict,
        **kwargs,
    ) -> None:
        """Initialize the multi-modal triples factory."""
        base = triples_factory
        # Init TriplesFactory
        super(BaseClass, self).__init__(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
        )
        # LiteralE doesn't handle 1-to-N relations, e.g. a project that received
        # multiple donations (KDD14). We thus make groups of (entity, relation) and compute
        # the mean value.
        df = pd.DataFrame(numeric_triples, columns=["head", "rel", "tail"])
        df[["head", "rel"]] = df[["head", "rel"]].astype(np.int64)
        df_gb = df.groupby(["head", "rel"], as_index=False).mean()
        self.numeric_triples = df_gb
        self.literals_to_id = num_rel_to_idx
        assert self.entity_to_id is not None
        self.numeric_literals = create_matrix_of_literals(
            numeric_triples=self.numeric_triples,
            entity_to_id=self.entity_to_id,
            literals_to_id=self.literals_to_id,
        )
        return
