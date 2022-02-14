# -*- coding: utf-8 -*-

"""
Functions to plot/print the results of the paper.
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Any, List, Tuple

exp_dir = Path(__file__).absolute().parents[1]

pd.set_option("display.max_rows", 500)


def plot_scores(
    df: pd.DataFrame,
    dataset: str,
    mask_values: List[Tuple[str, Any]] = [],
    label_cols: List[str] = [],
) -> None:
    df.fillna(value="None", inplace=True)
    mask = df["target_dataset"] == dataset
    for col, val in mask_values:
        if isinstance(val, list):
            mask2 = np.zeros_like(mask, dtype=bool)
            for v in val:
                mask2 = np.logical_or(mask2, df[col] == v)
            mask *= mask2
        else:
            mask *= df[col] == val
    df = df[mask]
    plt.figure(figsize=(4, 3), dpi=150)
    unq_vals = []
    for col in label_cols:
        unq_vals.append(df[col].unique().tolist())
    for vals in itertools.product(*unq_vals):
        str_vals = [str(val) for val in vals]
        label = ", ".join(str_vals)
        mask = np.ones(len(df), dtype=bool)
        for k in range(len(vals)):
            mask *= df[label_cols[k]] == vals[k]
        if mask.any():
            df2 = df[["epoch", "mean_scores", "std_scores", "random_state"]][mask]
            df2_max = df2.groupby("random_state").max("mean_scores")
            y_max, std = df2_max["mean_scores"].mean(), df2_max["std_scores"].mean()
            print(label, f"--- {y_max:.3f} +/- {std:.3f}")
            df2 = df2.groupby("epoch").mean()
            x, y = df2.index.to_numpy(), df2["mean_scores"].to_numpy()
            sorted_idx = np.argsort(x)
            plt.plot(x[sorted_idx], y[sorted_idx], ".-", label=label)
    plt.xlabel("Number of epoch")
    plt.ylabel("Mean cross-validation score")
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.show()
    return


dataset_names = {
    str("datasets/kdd14/target.parquet"): "kdd14",
    str("datasets/kdd15/target.parquet"): "kdd15",
    str("datasets/us_elections/target.parquet"): "us_elections (old)",
    str("datasets/housing_prices/target.parquet"): "housing_prices (old)",
    str("datasets/us_elections/target_log.parquet"): "us_elections",
    str("datasets/housing_prices/target_log.parquet"): "housing_prices",
    # str("datasets/housing_prices/target_yago4.parquet"): "housing_prices",
    str("datasets/us_accidents/counts.parquet"): "us_accidents",
    # str("datasets/us_accidents/duration.parquet"): "us_accidents_duration",
    str("datasets/movie_revenues/target.parquet"): "movie_revenues",
    str("datasets/company_employees/target.parquet"): "company_employees",
    "all": "all",
}


triples_dir_names = {
    str("datasets/kdd14/triplets"): "kdd14",
    str("datasets/kdd14/triplets_uniform"): "kdd14_uniform",
    str("datasets/kdd14/triplets_cardinality"): "kdd14_cardinality",
    str("datasets/kdd14/triplets_types"): "kdd14_types",
    str("datasets/kdd15/triplets"): "kdd15",
    str("datasets/kdd15/triplets_uniform"): "kdd15_uniform",
    str("datasets/kdd15/triplets_cardinality"): "kdd15_cardinality",
    str("datasets/kdd15/triplets_types"): "kdd15_types",
    str("datasets/yago3/triplets/full"): "yago3 full",
    str("datasets/yago3/triplets/only_cities"): "subset_housing_prices",
    str("datasets/yago3/triplets/only_counties"): "subset_us_elections",
    str("datasets/yago3/triplets/subset_us_accidents"): "subset_us_accidents",
    str("datasets/yago3/triplets/subset_movie_revenues"): "subset_movie_revenues",
    str("datasets/yago3/triplets/subset_company_employees"): "subset_company_employees",
    str("datasets/yago4/triplets"): "yago4",
}


def plot_scores_embeddings():
    """
    Plot mean cross-validation scores obtained by embedding models
    on each dataset at different epochs.
    The legend of the plot indicates (in order):
        - the embedding algorithm
        - the approach to embed numerical value (binning or KEN = LinearEmbedding2)
        - the activation function applied to embeddings of numerical values
          (ReLU or None)
        - the normalization of numerical values before embedding them
          (quantile or min-max)

    We also print the best scores obtained by each approach, with their
    standard deviations.

    Finally, we print the time needed to run each cross-validated evaluation.
    """
    scores_emb = pd.read_parquet(
        exp_dir / "prediction_scores/scores_embeddings.parquet"
    )
    scores_emb["id"] = scores_emb["checkpoint_id"]
    scores_emb["target_file"] = (
        scores_emb["target_file"].str.split("experiments/").str[-1]
    )
    scores_emb["triples_dir"] = (
        scores_emb["triples_dir"].str.split("experiments/").str[-1]
    )
    checkpoints = pd.read_parquet(exp_dir / "model_training/checkpoints_info.parquet")
    scores_emb = scores_emb.merge(checkpoints, on="id", how="left", suffixes=("", "_x"))
    scores_emb["target_dataset"] = scores_emb["target_file"].map(dataset_names)
    scores_emb["source_dataset"] = scores_emb["triples_dir"].map(triples_dir_names)
    scores_emb["mean_scores"] = [arr.mean() for arr in scores_emb["scores"]]
    scores_emb["std_scores"] = [arr.std() for arr in scores_emb["scores"]]

    for target_dataset, source_dataset in (
        ("kdd14", "kdd14"),
        ("kdd14", "kdd14_uniform"),
        ("kdd14", "kdd14_cardinality"),
        # ("kdd14", "kdd14_types"),
        ("kdd15", "kdd15"),
        ("kdd15", "kdd15_uniform"),
        ("kdd15", "kdd15_cardinality"),
        # ("kdd15", "kdd15_types"),
        ("us_elections", "yago3 full"),
        ("housing_prices", "yago3 full"),
        ("us_accidents", "yago3 full"),
        ("movie_revenues", "yago3 full"),
        ("company_employees", "yago3 full"),
        # Subset of YAGO3, to see if embeddings capture "deep features"
        ("us_elections", "subset_us_elections"),
        ("housing_prices", "subset_housing_prices"),
        ("us_accidents", "subset_us_accidents"),
        ("movie_revenues", "subset_movie_revenues"),
        ("company_employees", "subset_company_employees"),
    ):
        print("### --- ", target_dataset, " <- ", source_dataset)
        plot_scores(
            df=scores_emb,
            dataset=target_dataset,
            mask_values=[
                ("source_dataset", source_dataset),
                ("prediction_model", "HistGB"),
            ],
            label_cols=[
                "emb_model_name",
                "num_emb_model_name",
                "num_emb_model_activation_function",
                "num_emb_model_input_transformer",
            ],
        )
    ### Print duration of cross-validated evaluation with embeddings
    keep_cols = [
        "target_dataset",
        "duration",
        "n_repeats",
    ]
    df = scores_emb[keep_cols].copy()
    # Divide durations by 4 when we did 20 repeats instead of 5,
    # in order to compare with DFS.
    mask = df["n_repeats"] == 20
    df.loc[mask, "duration"] /= 4
    del df["n_repeats"]
    df = df.groupby(keep_cols[0]).mean()
    df.columns = ["Time for cross-validated evaluation (s)"]
    print(df.to_markdown())
    return


def print_scores_rdf2vec():

    scores_rdf = pd.read_parquet(exp_dir / "prediction_scores/scores_rdf2vec.parquet")
    configs = pd.read_parquet(exp_dir / "model_training/rdf2vec_config.parquet")
    scores_rdf = scores_rdf.merge(
        configs, on="filename", how="left", suffixes=("", "_x")
    )
    scores_rdf["target_file"] = (
        scores_rdf["target_file"].str.split("experiments/").str[-1]
    )
    scores_rdf["dataset"] = scores_rdf["target_file"].map(dataset_names)
    scores_rdf["mean_scores"] = [arr.mean() for arr in scores_rdf["scores"]]
    scores = pd.pivot_table(
        scores_rdf,
        values="mean_scores",
        index=["dataset"],
        columns="n_epochs",
    )
    print(scores.to_markdown())
    return


def print_scores_dfs():
    """
    Print the cross-validation scores (mean and std) obtained by deep features
    on each dataset.
    We also indicate the duration of this cross-validated evaluation.

    """
    scores_dfs = pd.read_parquet(exp_dir / "prediction_scores/scores_dfs.parquet")
    scores_dfs["target_file"] = (
        scores_dfs["target_file"].str.split("experiments/").str[-1]
    )
    scores_dfs["dataset"] = scores_dfs["target_file"].map(dataset_names)
    scores_dfs["mean_scores"] = [arr.mean() for arr in scores_dfs["scores"]]
    scores_dfs["std_scores"] = [arr.std() for arr in scores_dfs["scores"]]
    keep_cols = ["dataset", "depth", "mean_scores", "std_scores", "duration"]
    # Filter old metric on kdd15
    mask = (scores_dfs["dataset"] == "kdd15") * (
        scores_dfs["scoring"] == "average_precision"
    )
    scores = scores_dfs[keep_cols][~mask]
    # Remove old scores for us_elections and housing_prices on raw target (not log)
    print(scores.sort_values("dataset").to_markdown(index=False))
    return


def print_scores_FE():
    """
    Print the cross-validation scores for manual feature engineering.
    """
    scores_FE = pd.read_parquet(exp_dir / "prediction_scores/scores_FE.parquet")
    scores_FE["dataset"] = (
        scores_FE["target_file"].str.split("/").str[-1].str.replace(".parquet", "")
    )
    scores_FE["mean_scores"] = [arr.mean() for arr in scores_FE["scores"]]
    scores_FE["std_scores"] = [arr.std() for arr in scores_FE["scores"]]
    keep_cols = ["dataset", "mean_scores", "std_scores"]
    scores = scores_FE[keep_cols]
    print(scores.sort_values("dataset").to_markdown(index=False))
    return


def print_complexity_dfs():
    """
    Print for each dataset and depth the time/memory needed to run DFS on all entities,
    as well as the number of generated features.
    The number of features is given after encoding categorical ones.
    """
    dfs_info = pd.read_parquet(exp_dir / "deep_feature_synthesis/feature_info.parquet")
    dfs_info["target_entities"] = (
        dfs_info["target_entities"].str.split("experiments/").str[-1]
    )
    dfs_info["triples_dir"] = dfs_info["triples_dir"].str.split("experiments/").str[-1]
    dfs_info["target_entities"] = dfs_info["target_entities"].map(dataset_names)
    dfs_info["dataset"] = dfs_info["triples_dir"].map(triples_dir_names)
    dfs_info["duration"] = dfs_info["build_duration"] + dfs_info["dfs_duration"]
    keep_cols = [
        "dataset",
        "depth",
        "peak_memory",
        "duration",
        "number_of_features",
    ]
    mask = dfs_info["target_entities"] == "all"
    df = dfs_info[mask][keep_cols]
    print(df.sort_values("dataset").to_markdown(index=False))
    return


def print_complexity_embeddings():
    """
    Print the time and memory needed to learn embeddings for each
    dataset and embedding model (LinearEmbedding2 = KEN).
    """
    ### Time/Memory to learn embeddings
    checkpoints = pd.read_parquet(exp_dir / "model_training/checkpoints_info.parquet")
    checkpoints["triples_dir"] = (
        checkpoints["triples_dir"].str.split("experiments/").str[-1]
    )
    checkpoints["dataset"] = checkpoints["triples_dir"].map(triples_dir_names)
    mask = checkpoints["dataset"].isin(["kdd14", "kdd15", "yago3 full"])
    mask *= checkpoints["num_emb_model_name"].isin(["None", "LinearEmbedding2"])
    keep_cols = [
        "dataset",
        "emb_model_name",
        "num_emb_model_name",
        "duration",
        "peak_memory_usage",
    ]
    df = checkpoints[keep_cols][mask]
    df = df.groupby(keep_cols[:3]).min()
    print(df.to_markdown())
    return


def print_reconstruction_scores():
    """
    Print the reconstruction scores of embedding models.
    """
    df = pd.read_parquet(exp_dir / "attribute_reconstruction/results.parquet")
    df["target_stat"].fillna("mean", inplace=True)
    df["target_file"] = df["target_file"].str.split("experiments/").str[-1]
    df["dataset"] = df["target_file"].map(dataset_names)
    df["mean_scores"] = [arr.mean() for arr in df["scores"]]
    df["std_scores"] = [arr.std() for arr in df["scores"]]
    df["model"] = df["model"] + " KEN = " + df["KEN"].astype(str)
    keep_cols = [
        "dataset",
        "model",
        "target_attr",
        "target_stat",
        "mean_scores",
        "std_scores",
    ]
    print(df[keep_cols].sort_values("model").to_markdown(index=False))
    return


def nn_matching_scores():
    """
    Print scores for nearest neighbor matching.
    """
    df = pd.read_parquet(exp_dir / "morphology/scores_nn_matching.parquet")
    df["dataset"] = df["target_file"].map(dataset_names)
    df["mean_scores"] = [arr.mean() for arr in df["scores"]]
    df["std_scores"] = [arr.std() for arr in df["scores"]]
    df = df.groupby(["dataset", "embedding_model", "use_KEN"], as_index=False).mean()
    keep_cols = ["dataset", "embedding_model", "use_KEN", "mean_scores", "std_scores"]
    print(df[keep_cols].sort_values("dataset").to_markdown(index=False))
    return


def fasttext_scores():
    """
    Print scores for fasttext embeddings.
    """
    df = pd.read_parquet(exp_dir / "morphology/scores_fasttext.parquet")
    df["dataset"] = df["target_file"].map(dataset_names)
    df["mean_scores"] = [arr.mean() for arr in df["scores"]]
    df["std_scores"] = [arr.std() for arr in df["scores"]]
    df = df.groupby(["dataset"], as_index=False).mean()
    keep_cols = ["dataset", "mean_scores", "std_scores"]
    print(df[keep_cols].sort_values("dataset").to_markdown(index=False))
    return
