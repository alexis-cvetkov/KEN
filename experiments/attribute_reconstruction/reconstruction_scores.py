# -*- coding: utf-8 -*-

"""
Compute the cross-validation scores when predicting entities numerical attributes
(e.g. county population) from their embeddings.
We evaluate these scores on the models that obtained the highest cv scores on the
target dataset (see Table VII of the paper).

We use simple regression models: K-Nearest-Neighbors, with K being tuned
to maximize the scores.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import torch
from time import time

from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor

from KEN.dataloader import DataLoader
from KEN.evaluation import compute_literalE_embeddings

exp_dir = Path(__file__).absolute().parents[1]


def make_X_y(
    triples_dir,
    target_file,
    target_attr,
    ckpt_info,
    model: str,
    use_KEN: bool,
    epoch: int,
    random_state: int,
    target_stat: str,
):
    ckpt_params = {
        "emb_model_name": model,
        "triples_dir": str(triples_dir).replace("data/parietal", "storage"),
        "random_state": random_state,
    }
    if use_KEN:
        ckpt_params.update(
            {
                "num_emb_model_name": "LinearEmbedding2",
                "num_emb_model_input_transformer": "quantile",
                "num_emb_model_activation_function": "relu",
            }
        )
    else:
        ckpt_params.update({"num_emb_model_name": "None"})
    # Get triples for target_attr
    dl = DataLoader(triples_dir, use_features="all")
    idx = dl.rel_to_idx[target_attr]
    triples = dl.get_dataframe()
    triples = triples[triples["rel"] == idx]
    del triples["rel"]
    # Load dataframe
    df = pd.read_parquet(target_file)
    mask = df["col_to_embed"].isin(dl.entity_to_idx.keys())
    df = df[mask]
    df["head"] = df["col_to_embed"].map(dl.entity_to_idx)
    df = df[["head"]]
    df = df.drop_duplicates()
    # Groupby + mean
    if target_stat == "mean":
        triples = triples.groupby("head").mean()
    elif target_stat == "Q1":
        triples = triples.groupby("head").filter(lambda x: len(x) > 30)
        triples = triples.groupby("head").quantile(q=0.25)
    elif target_stat == "Q3":
        triples = triples.groupby("head").filter(lambda x: len(x) > 30)
        triples = triples.groupby("head").quantile(q=0.75)
    # Merge
    df = df.merge(triples, on="head")
    df = df.sample(frac=1)
    df = df.iloc[:10000]
    print(len(df))
    # Find ckpt_file with ckpt_params
    df_ckpt_info = pd.read_parquet(ckpt_info)
    mask = np.ones(len(df_ckpt_info), dtype=bool)
    for key, value in ckpt_params.items():
        mask *= df_ckpt_info[key] == value
    mask *= np.array([epoch in epochs for epochs in df_ckpt_info["epochs"]])
    # Add embeddings
    assert len(df_ckpt_info[mask]) == 1, "checkpoint is not uniquely defined"
    ckpt_id = df_ckpt_info[mask]["id"].values[0]
    ckpt_file = exp_dir / f"model_training/checkpoints/{ckpt_id}_{epoch}"
    checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"))
    if model == "MuRE":
        embeddings = checkpoint["model_state_dict"][
            "entity_representations.0._embeddings.weight"
        ].numpy()
    elif model in ["TransE", "DistMult"]:
        embeddings = checkpoint["model_state_dict"][
            "entity_embeddings._embeddings.weight"
        ].numpy()
    elif model == "DistMultLiteralGated":
        embeddings = compute_literalE_embeddings(checkpoint["model_state_dict"])
    X = embeddings[df["head"]]
    y = df["tail"]
    return X, y


def cv_scores(X, y):
    param_grid = {
        "n_neighbors": [1, 2, 3, 4, 8, 16],
        "p": [1, 2],
        "weights": ["uniform", "distance"],
    }
    model = GridSearchCV(
        KNeighborsRegressor(), param_grid=param_grid, scoring="r2", cv=3, n_jobs=3
    )
    cv = RepeatedKFold(n_repeats=5, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=25)
    return scores


def reconstruction_scores(
    triples_dir,
    target_file,
    target_attr,
    ckpt_info,
    model,
    use_KEN,
    epoch,
    random_state,
    results_file,
    target_stat=None,
):
    X, y = make_X_y(
        triples_dir,
        target_file,
        target_attr,
        ckpt_info,
        model,
        use_KEN,
        epoch,
        random_state,
        target_stat,
    )
    scores = cv_scores(X, y)
    results = {
        "triples_dir": str(triples_dir),
        "target_file": str(target_file),
        "target_attr": target_attr,
        "random_state": random_state,
        "model": model,
        "KEN": use_KEN,
        "scores": scores,
        "target_stat": target_stat,
    }
    df = pd.DataFrame([results])
    if Path(results_file).is_file():
        old_df = pd.read_parquet(results_file)
        df = df.append(old_df).reset_index(drop=True)
    df.to_parquet(results_file, index=False)
    return


def kdd14():
    for target_stat in ["mean", "Q1", "Q3"]:
        epochs = [40, 16, 40, 16, 8]
        params = [
            ("MuRE", True),  # model, use_KEN
            ("MuRE", False),
            ("DistMult", True),
            ("DistMult", False),
            ("DistMultLiteralGated", False),
        ]
        for epoch, (model, use_KEN) in zip(epochs, params):
            print(f"# {model}, KEN = {use_KEN}")
            t0 = time()
            reconstruction_scores(
                triples_dir=exp_dir / "datasets/kdd14/triplets",
                target_file=exp_dir / "datasets/kdd14/target.parquet",
                target_attr="projectid -> donation_to_project",
                ckpt_info=exp_dir / "model_training/checkpoints_info.parquet",
                model=model,
                use_KEN=use_KEN,
                epoch=epoch,
                random_state=0,
                results_file=exp_dir / "attribute_reconstruction/results.parquet",
                target_stat=target_stat,
            )
            print("Time ellapsed: ", time() - t0)
    return


def kdd15():
    for target_stat in ["mean", "Q1", "Q3"]:
        epochs = [40, 40, 40, 40, 40]
        params = [
            ("MuRE", True),
            ("MuRE", False),
            ("DistMult", True),
            ("DistMult", False),
            ("DistMultLiteralGated", False),
        ]
        for epoch, (model, use_KEN) in zip(epochs, params):
            print(f"# {model}, KEN = {use_KEN}")
            t0 = time()
            reconstruction_scores(
                triples_dir=exp_dir / "datasets/kdd15/triplets",
                target_file=exp_dir / "datasets/kdd15/target.parquet",
                target_attr="enrollment_id -> time",
                ckpt_info=exp_dir / "model_training/checkpoints_info.parquet",
                model=model,
                use_KEN=use_KEN,
                epoch=epoch,
                random_state=0,
                results_file=exp_dir / "attribute_reconstruction/results.parquet",
                target_stat=target_stat,
            )
            print("Time ellapsed: ", time() - t0)
    return


def yago3():
    for target_attr in ["<hasNumberOfPeople>", "<hasLatitude>", "<hasLongitude>"]:
        epochs = [8, 8, 40, 40, 40]
        params = [
            ("MuRE", True),
            ("MuRE", False),
            ("DistMult", True),
            ("DistMult", False),
            ("DistMultLiteralGated", False),
        ]
        for epoch, (model, use_KEN) in zip(epochs, params):
            print(f"# {model}, KEN = {use_KEN}")
            t0 = time()
            reconstruction_scores(
                triples_dir=exp_dir / "datasets/yago3/triplets/full",
                target_file=exp_dir / "datasets/us_elections/target.parquet",
                target_attr=target_attr,
                ckpt_info=exp_dir / "model_training/checkpoints_info.parquet",
                model=model,
                use_KEN=use_KEN,
                epoch=epoch,
                random_state=0,
                results_file=exp_dir / "attribute_reconstruction/results.parquet",
            )
            print("Time ellapsed: ", time() - t0)
    return


if __name__ == "__main__":
    kdd14()
    kdd15()
    yago3()
