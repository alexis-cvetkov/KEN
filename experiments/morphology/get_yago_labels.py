# -*- coding: utf-8 -*-

"""
Code to compute prediction scores with Nearest Neighbor matching.
Target entity names are encoded using a CountVectorizer, and then matched
to the closest entity in YAGO based on their cosine similiarity.
"""

import numpy as np
from pathlib import Path
import pandas as pd

from KEN.dataloader import DataLoader

exp_dir = Path(__file__).absolute().parents[1]

if __name__ == "__main__":
    # Load YAGO3 labels
    labels = pd.read_csv(
        exp_dir / "datasets/yago3/raw/yagoLabels.tsv", sep="\t",
        skiprows=1,
        names=["C0", "entity", "relation", "label", "C4"],
        usecols=["entity", "relation", "label"]
    )
    # Drop missing values
    labels.dropna(inplace=True)
    # Keep only english labels
    mask = labels["label"].str.contains("@eng")
    labels = labels[mask]
    # Load entities for us_elections
    df = pd.read_parquet(exp_dir / "datasets/us_elections/target_log.parquet")
    entities = df["col_to_embed"].unique()
    mask = labels["entity"].isin(entities)
    ent_labels = labels[mask].copy()
    del ent_labels["relation"]
    ent_labels = ent_labels.groupby("entity")["label"].unique().reset_index()
    ent_labels["label"][6]
