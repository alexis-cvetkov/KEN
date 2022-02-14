# -*- coding: utf-8 -*-

"""
Code to compute prediction scores with Nearest Neighbor matching.
Target entity names are encoded using a CountVectorizer, and then matched
to the closest entity in YAGO based on their cosine similiarity.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import Union, Tuple

from KEN.dataloader import DataLoader
from KEN.evaluation.prediction_scores import compute_prediction_scores

exp_dir = Path(__file__).absolute().parents[1]


def make_X_y(
    ckpt_file: Union[str, Path],
    embedding_model: str,
    df: pd.DataFrame,
    matched_idx: np.array,
):
    """Return X with entity embeddings and y with target"""
    # Load embeddings from checkpoint
    checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"))
    if embedding_model == "MuRE":
        embeddings = checkpoint["model_state_dict"][
            "entity_representations.0._embeddings.weight"
        ].numpy()
    elif embedding_model == "TransE":
        embeddings = checkpoint["model_state_dict"][
            "entity_embeddings._embeddings.weight"
        ].numpy()

    X_emb = embeddings[matched_idx]
    # Add party column (only for us_elections)
    if "party" in df.columns:
        enc_col = pd.get_dummies(df["party"], prefix="party")
        X_emb = np.hstack([X_emb, enc_col.to_numpy()])
    y = df["target"]
    return X_emb, y


def make_nn_matching(
    target_file: Union[str, Path],
    triples_dir: Union[str, Path],
) -> Tuple[pd.DataFrame, np.array]:
    """
    Match raw entity names to embedding indices using a CountVectorizer
    on n-grams and by pairing entities with highest cosine similarity.
    """
    # Load target dataframe
    df = pd.read_parquet(target_file)
    # Init dataloader
    dataloader = DataLoader(triples_dir, use_features="all")
    # Remove target entities that could not be manually matched
    mask = df["col_to_embed"].isin(list(dataloader.entity_to_idx.keys()))
    df = df[mask]
    # Compute CountVectorizer on raw entity names
    target_ent = df["raw_entities"]
    unq_target_ent, inverse = np.unique(target_ent, return_inverse=True)
    emb_ent = list(dataloader.entity_to_idx.keys())
    if "yago3" in str(triples_dir):
        emb_ent = [
            x.replace("_", " ").replace("<", "").replace(">", "") for x in emb_ent
        ]
    cv = CountVectorizer(analyzer="char", ngram_range=(3, 3))
    cv.fit(unq_target_ent)
    target_ent_enc = cv.transform(unq_target_ent)
    emb_ent_enc = cv.transform(emb_ent)
    sim_scores = cosine_similarity(target_ent_enc, emb_ent_enc, dense_output=False)
    matched_idx = csr_matrix.argmax(sim_scores, axis=1)
    matched_idx = np.array(matched_idx)[inverse, 0]
    return df, matched_idx


def prediction_scores(
    ckpt_info: Union[Path, str],
    target_file: Union[Path, str],
    triples_dir: Union[Path, str],
    embedding_model: str,
    use_KEN: bool,
    n_repeats: int,
    tune_hyperparameters: bool,
    is_regression_task: bool,
    prediction_model: str,
    scoring: str,
    results_file: Union[Path, str],
    matching_scores_file: Union[Path, str],
):
    # Load checkpoints for the embedding model with/without KEN
    checkpoints = pd.read_parquet(ckpt_info)
    mask = (checkpoints["emb_model_name"] == embedding_model) * (
        checkpoints["triples_dir"] == str(triples_dir)
    )
    if use_KEN:
        mask *= (
            (checkpoints["num_emb_model_name"] == "LinearEmbedding2")
            * (checkpoints["num_emb_model_activation_function"] == "relu")
            * (checkpoints["num_emb_model_input_transformer"] == "quantile")
        )
    else:
        mask *= checkpoints["use_features"] == "categorical"
    checkpoints = checkpoints[mask].reset_index(drop=True)
    # Load prediction scores (with manual matching) for the selected checkpoints
    # For each seed, select the epoch with highest score.
    matching_scores = pd.read_parquet(matching_scores_file)
    mask = (
        matching_scores["checkpoint_id"].isin(checkpoints["id"])
        * (matching_scores["target_file"] == str(target_file))
        * (matching_scores["scoring"] == scoring)
        * (matching_scores["prediction_model"] == prediction_model)
        * (matching_scores["tune_hyperparameters"] == tune_hyperparameters)
    )
    matching_scores = matching_scores[mask]
    matching_scores["mean_score"] = [np.mean(x) for x in matching_scores["scores"]]
    best_ckpt = (
        matching_scores.groupby(["checkpoint_id", "random_state"])
        .max("mean_score")
        .reset_index()
    )
    # Merge checkpoint_dir column
    best_ckpt = best_ckpt.rename({"checkpoint_id": "id"}, axis=1)
    best_ckpt = best_ckpt.merge(checkpoints[["id", "checkpoint_dir"]], on="id")
    # Load previously stored results
    if Path(results_file).is_file():
        df_res = pd.read_parquet(results_file)
    else:
        df_res = None
    # Match raw entity names to embedding indices
    print("Start nearest neighbor matching on", str(target_file))
    target_df, matched_idx = make_nn_matching(target_file, triples_dir)
    print("Start computing cross-validation scores")
    for _, ckpt in tqdm(best_ckpt.iterrows()):
        ckpt_id, epoch, random_state, ckpt_dir = (
            ckpt["id"],
            ckpt["epoch"],
            ckpt["random_state"],
            ckpt["checkpoint_dir"],
        )
        # Check if results for the same config are in df_res already
        if df_res is not None:
            mask = df_res["checkpoint_id"] == ckpt_id
            mask *= df_res["epoch"] == epoch
            mask *= df_res["target_file"] == str(target_file)
            mask *= df_res["scoring"] == scoring
            mask *= df_res["prediction_model"] == prediction_model
            mask *= df_res["tune_hyperparameters"] == tune_hyperparameters
            # If not, compute prediction scores
        if df_res is None or (~mask).all():
            ckpt_file = Path(ckpt_dir) / f"{ckpt_id}_{epoch}"
            X, y = make_X_y(ckpt_file, embedding_model, target_df, matched_idx)
            cv_scores, _, _ = compute_prediction_scores(
                X,
                y,
                n_repeats,
                tune_hyperparameters,
                is_regression_task,
                prediction_model,
                scoring,
                random_state,
            )
            # Save results to a dataframe
            results = {
                "triples_dir": str(triples_dir),
                "target_file": str(target_file),
                "checkpoint_id": ckpt_id,
                "embedding_model": embedding_model,
                "use_KEN": use_KEN,
                "epoch": epoch,
                "n_repeats": n_repeats,
                "tune_hyperparameters": tune_hyperparameters,
                "prediction_model": prediction_model,
                "scoring": scoring,
                "random_state": random_state,
                "scores": cv_scores,
            }
            new_df_res = pd.DataFrame([results])
            if Path(results_file).is_file():
                df_res = pd.read_parquet(results_file)
                df_res = df_res.append(new_df_res).reset_index(drop=True)
            else:
                df_res = new_df_res
            df_res.to_parquet(results_file, index=False)
    return


def scores_MuRE_KEN():
    for target_file in [
        # exp_dir / "datasets/us_elections/target_log.parquet",
        # exp_dir / "datasets/housing_prices/target_log.parquet",
        # exp_dir / "datasets/us_accidents/counts.parquet",
        # exp_dir / "datasets/movie_revenues/target.parquet",
        exp_dir / "datasets/company_employees/target.parquet",
    ]:
        prediction_scores(
            ckpt_info=exp_dir / "model_training/checkpoints_info.parquet",
            target_file=target_file,
            triples_dir=exp_dir / "datasets/yago3/triplets/full",
            embedding_model="MuRE",
            use_KEN=True,
            n_repeats=5,
            tune_hyperparameters=True,
            is_regression_task=True,
            prediction_model="HistGB",
            scoring="r2",
            results_file=exp_dir / "morphology/scores_nn_matching.parquet",
            matching_scores_file=exp_dir
            / "prediction_scores/scores_embeddings.parquet",
        )


if __name__ == "__main__":
    scores_MuRE_KEN()
