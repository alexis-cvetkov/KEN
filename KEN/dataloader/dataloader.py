# -*- coding: utf-8 -*-

"""A class to load training triples and convert them to a TriplesFactory object that
can be used by PyKEEN."""

from typing import Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from pykeen.triples import TriplesFactory

from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory


class DataLoader:
    def __init__(self, triples_dir: Path, use_features: str) -> None:
        self.triples_dir = triples_dir
        self.use_features = use_features
        self.load_metadata()
        return

    def load_metadata(self) -> None:
        with open(f"{self.triples_dir}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        for key, value in metadata.items():
            setattr(self, key, value)
        return

    def get_triples_factory(self) -> TriplesFactory:
        df = pd.DataFrame(np.load(Path(self.triples_dir) / "triplets.npy"))
        if self.use_features == "categorical":
            mask = df["rel"] >= self.n_num_attr
            df = df[mask]
            df = df.astype(np.int64)
        elif self.use_features == "numerical":
            mask = df["rel"] < self.n_num_attr
            df = df[mask]
        tf = TriplesFactory(df.to_numpy(), self.entity_to_idx, self.rel_to_idx)
        tf.n_num_rel = self.n_num_attr
        return tf

    def get_numeric_triples_factory(self) -> TriplesNumericLiteralsFactory:
        """Only used with DistMult + LiteralE"""
        df = pd.DataFrame(np.load(Path(self.triples_dir) / "triplets.npy"))
        assert self.use_features == "all"
        mask = df["rel"] >= self.n_num_attr
        triples, num_triples = df[mask].to_numpy(dtype=np.int64), df[~mask].to_numpy()
        triples[:, 1] -= self.n_num_attr
        non_num_rels = list(self.rel_to_idx.keys())[self.n_num_attr :]
        num_rels = list(self.rel_to_idx.keys())[:self.n_num_attr]
        non_num_rel_to_idx = {rel: idx for idx, rel in enumerate(non_num_rels)}
        num_rel_to_idx = {rel: idx for idx, rel in enumerate(num_rels)}
        tf = TriplesFactory(triples, self.entity_to_idx, non_num_rel_to_idx)
        num_tf = TriplesNumericLiteralsFactory(
            triples_factory=tf,
            numeric_triples=num_triples,
            num_rel_to_idx=num_rel_to_idx,
        )
        return num_tf

    def get_hrt(self) -> Tuple[np.array]:
        df = np.load(Path(self.triples_dir) / "triplets.npy")
        return df["head"], df["rel"], df["tail"]

    def get_dataframe(self) -> pd.DataFrame:
        df = np.load(Path(self.triples_dir) / "triplets.npy")
        df = pd.DataFrame.from_records(df)
        if self.use_features == "categorical":
            mask = df["rel"] >= self.n_num_attr
            df = df[mask]
            df = df.astype(np.int64)
        return df
