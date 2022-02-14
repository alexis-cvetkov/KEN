# -*- coding: utf-8 -*-

"""
Code to train RDF2Vec embeddings.
"""


from memory_profiler import memory_usage
import numpy as np
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm
from typing import Union, List
import uuid

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker

from KEN.dataloader import DataLoader


exp_dir = Path(__file__).absolute().parents[1]


def train_model(
    triples_dir: Path,
    target_entities: Union[Path, str, None],
    add_frequent_entities: float,
    max_depth: int,
    max_walks: Union[int, None],
    n_epoch_list: List[int],
    n_jobs: int,
    random_state: int,
):
    # Load data
    dl = DataLoader(triples_dir, use_features="categorical")
    triples = dl.get_dataframe()
    triples["rel"] = "r" + triples["rel"].astype(str)
    head, rel, tail = triples.to_numpy(dtype=str).T
    # Construct graph
    print("# Construct graph")
    kg = KG()
    for k in tqdm(range(len(head))):
        subj = Vertex(head[k])
        obj = Vertex(tail[k])
        pred = Vertex(rel[k], predicate=True, vprev=subj, vnext=obj)
        kg.add_walk(subj, pred, obj)
    # Init walker
    walker = RandomWalker(
        max_depth=max_depth,
        max_walks=max_walks,
        with_reverse=True,
        n_jobs=n_jobs,
        random_state=random_state,
        md5_bytes=None,
    )
    ### Define entities from which the random walks depart
    if target_entities == None:  # Only use most frequent entities
        assert add_frequent_entities > 0
        ent_to_embed = np.array(list(dl.entity_to_idx.values()), dtype=str)
        if add_frequent_entities == 1:  # Add all entities
            start_ent = ent_to_embed
        else:  # Add a fraction of the most common entities
            start_ent = pd.Series(np.concatenate([head, tail]))
            ent_counts = start_ent.value_counts(normalize=True)
            n_entities = int(len(ent_counts) * add_frequent_entities)
            start_ent = ent_counts.iloc[:n_entities].index.values
    else:  # Use target entities
        df = pd.read_parquet(target_entities)
        start_ent = df["col_to_embed"].map(dl.entity_to_idx).dropna()
        start_ent = start_ent.astype(np.int64).astype(str).unique()
        ent_to_embed = start_ent.copy()
        assert add_frequent_entities < 1
        if add_frequent_entities > 0:  # Add a fraction of the most common entities
            start_ent2 = pd.Series(np.concatenate([head, tail]))
            ent2_counts = start_ent2.value_counts()
            n_entities = int(len(ent2_counts) * add_frequent_entities)
            # 10% most common entities correspond to 63% of triplets
            start_ent2 = ent2_counts.iloc[:n_entities].index.values
            start_ent = np.concatenate([start_ent, start_ent2])
    # Only keep start entities that are in the KG
    mask = pd.Series(start_ent).isin(head) | pd.Series(start_ent).isin(tail)
    start_ent = start_ent[mask].copy()
    ### Get walks
    transformer = RDF2VecTransformer(Word2Vec(), walkers=[walker], verbose=1)
    t0 = time.perf_counter()
    walks = transformer.get_walks(kg, start_ent)
    duration_walks = time.perf_counter() - t0
    ### Fit the model on walks for different epochs
    config_list = []
    for n_epoch in n_epoch_list:
        transformer = RDF2VecTransformer(
            Word2Vec(epochs=n_epoch, vector_size=200),
            walkers=[walker],
            verbose=1,
        )
        t0 = time.perf_counter()
        transformer.fit(walks, is_update=False)
        duration_fit = time.perf_counter() - t0
        # Get embeddings for entities in the vocabulary, otherwise assign them
        # a specific vector
        vocab = list(transformer.embedder._model.wv.key_to_index.keys())
        mask = pd.Series(ent_to_embed).isin(vocab)
        embeddings, _ = transformer.transform(kg, ent_to_embed[mask])
        missing_embeddings = -np.ones(((~mask).sum(), 200), dtype=np.float32)
        embeddings = np.vstack([np.array(embeddings), missing_embeddings])
        df = pd.DataFrame(embeddings, columns=[f"X{k}" for k in range(200)])
        df["entity"] = np.concatenate([ent_to_embed[mask], ent_to_embed[~mask]]).astype(
            np.int64
        )
        filename = (
            exp_dir / f"model_training/rdf2vec_vectors/{str(uuid.uuid4())}.parquet"
        )
        df.to_parquet(filename, index=False)
        config = {
            "triples_dir": str(triples_dir),
            "target_entities": str(target_entities),
            "max_depth": max_depth,
            "max_walks": max_walks,
            "n_epochs": n_epoch,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "duration_walks": duration_walks,
            "duration_fit": duration_fit,
            "filename": str(filename),
            "add_frequent_entities": add_frequent_entities,
        }
        config_list.append(config)
    return config_list


def train_all_models(
    add_target_entities,
    add_frequent_entities,
    max_depth,
    max_walks,
    n_epoch_list,
    n_jobs,
    random_state,
):
    data_dir = exp_dir / "datasets"
    config_file = exp_dir / "model_training/rdf2vec_config.parquet"
    if add_target_entities:
        L = [
            ("yago3/triplets/full", "us_elections/target_log.parquet"),
            ("yago3/triplets/full", "housing_prices/target_log.parquet"),
            ("yago3/triplets/full", "us_accidents/counts.parquet"),
            ("yago3/triplets/full", "movie_revenues/target.parquet"),
            ("yago3/triplets/full", "company_employees/target.parquet"),
            ("kdd14/triplets", "kdd14/target.parquet"),
            ("kdd15/triplets", "kdd15/target.parquet"),
        ]
    else:
        L = [
            ("yago3/triplets/full", None),
            ("kdd14/triplets", None),
            ("kdd15/triplets", None),
        ]
    for triples_dir, target_entities in L:
        triples_dir = data_dir / triples_dir
        if add_target_entities:
            target_entities = data_dir / target_entities
        print("Train model on target:", target_entities)

        peak_memory_usage, config_list = memory_usage(
            (
                train_model,
                (
                    triples_dir,
                    target_entities,
                    add_frequent_entities,
                    max_depth,
                    max_walks,
                    n_epoch_list,
                    n_jobs,
                    random_state,
                ),
            ),
            max_usage=True,
            include_children=True,
            max_iterations=1,
            retval=True,
        )
        config_df = pd.DataFrame(config_list)
        config_df["peak_memory"] = peak_memory_usage
        if Path(config_file).is_file():
            old_configs = pd.read_parquet(config_file)
            config_df = config_df.append(old_configs).reset_index(drop=True)
        config_df.to_parquet(config_file, index=False)
    return

if __name__ == "__main__":
    # Train RDF2Vec on target entities only
    train_all_models(
        add_target_entities=True,
        add_frequent_entities=0,
        max_depth=2,
        max_walks=10,
        n_epoch_list=[5, 10, 20, 40, 80],
        n_jobs=40,
        random_state=0,
    )
    #  Train RDF2Vec on the 1% most frequent entities
    train_all_models(
        add_target_entities=False,
        add_frequent_entities=0.01,
        max_depth=2,
        max_walks=10,
        n_epoch_list=[5, 10, 20, 40, 80],
        n_jobs=40,
        random_state=0,
    )
