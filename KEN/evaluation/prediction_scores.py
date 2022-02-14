# -*- coding: utf-8 -*-

"""
Function to compute cross-validated prediction scores, using deep features
or embeddings. Also profiles the duration/memory and save results to a dataframe.
"""

import featuretools as ft
from memory_profiler import memory_usage
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, Union, List

from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    cross_val_score,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
)

from KEN.dataloader import DataLoader


def compute_literalE_embeddings(model_state):
    with torch.no_grad():
        # Load embeddings and literals
        x = model_state["entity_representations.0._embeddings.weight"]
        literal = model_state["entity_representations.1._embeddings.weight"]
        combination = torch.cat([x, literal], -1)
        # Compute h
        linlayer_weight = model_state[
            "interaction.combination.combination_linear_layer.weight"
        ]
        linlayer_bias = model_state[
            "interaction.combination.combination_linear_layer.bias"
        ]
        h = torch.tanh(F.linear(combination, linlayer_weight, linlayer_bias))
        # Compute z
        gate_entity = model_state["interaction.combination.gate_entity_layer.weight"]
        gate_literal = model_state["interaction.combination.gate_literal_layer.weight"]
        gate_bias = model_state["interaction.combination.bias"]
        z = F.linear(x, gate_entity) + F.linear(literal, gate_literal) + gate_bias
        z = torch.sigmoid(z)
    return (z * h + (1 - z) * x).numpy()


def make_X_y_dfs(
    triples_dir: Union[Path, str],
    target_entities: Union[Path, str],
    feature_info: Union[Path, str],
    depth: int,
):
    ### Load and preprocess target dataframe
    dataloader = DataLoader(triples_dir, use_features="all")
    df = pd.read_parquet(target_entities)
    if "raw_entities" in df.columns:
        del df["raw_entities"]
    df.rename({"col_to_embed": "head"}, axis=1, inplace=True)
    # Remove entities that are not in dataloader
    mask = df["head"].isin(list(dataloader.entity_to_idx.keys()))
    df = df[mask]
    # Replace entity names by their idx
    df["head"] = df["head"].map(dataloader.entity_to_idx).astype(int)
    # Encode the column "party" if it exists (only for us elections)
    if "party" in df.columns:
        enc_col = pd.get_dummies(df["party"], prefix="party")
        df = pd.concat([df, enc_col], axis=1)
        del df["party"]
    ### Load features info and feature_matrix
    info = pd.read_parquet(feature_info)
    idx = np.where(
        (info["triples_dir"] == str(triples_dir))
        * (info["target_entities"] == str(target_entities))
        * (info["depth"] == depth)
    )[0][0]
    feature_path = info["feature_path"][idx]
    feature_matrix_path = info["feature_matrix_path"][idx]
    features = ft.load_features(feature_path)
    feature_matrix = pd.read_parquet(feature_matrix_path)
    feature_matrix.ww.init()
    # Encode categorical features
    feature_matrix, features = ft.encode_features(feature_matrix, features, top_n=10)
    feature_matrix = feature_matrix.reset_index()

    # Merge features into the target dataframe
    df = df.merge(feature_matrix, on="head", how="left")
    del df["head"]
    y = df.pop("target").to_numpy()
    X = df.astype(float).to_numpy()
    return X, y


def make_X_y_embeddings(
    ckpt_file: Union[Path, str], model: str, dataloader: DataLoader, df: pd.DataFrame
):
    # Remove raw entity names
    if "raw_entities" in df.columns:
        del df["raw_entities"]
    # Load checkpoint and extract embeddings
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
    # Remove entities that are not in dataloader
    mask = df["col_to_embed"].isin(list(dataloader.entity_to_idx.keys()))
    df = df[mask]
    # Replace entity names by their idx
    df["col_to_embed"] = df["col_to_embed"].map(dataloader.entity_to_idx).astype(int)
    X_emb = embeddings[df["col_to_embed"]]
    y = df["target"]
    # Add party column (only for us_elections)
    if "party" in df.columns:
        enc_col = pd.get_dummies(df["party"], prefix="party")
        X_emb = np.hstack([X_emb, enc_col.to_numpy()])
    return X_emb, y


def make_X_y_rdf2vec(filename, dataloader, df):
    if "raw_entities" in df.columns:
        del df["raw_entities"]
    # Load embeddings
    embeddings = pd.read_parquet(filename)
    mask = df["col_to_embed"].isin(list(dataloader.entity_to_idx.keys()))
    df = df[mask].copy()
    # Replace entity names by their idx
    df["col_to_embed"] = df["col_to_embed"].map(dataloader.entity_to_idx).astype(int)
    df.rename({"col_to_embed": "entity"}, axis=1, inplace=True)
    df = df.merge(embeddings, on="entity", how="left")
    y = df["target"]
    X_emb = df[[f"X{k}" for k in range(200)]].to_numpy()
    # Add party column (only for us_elections)
    if "party" in df.columns:
        enc_col = pd.get_dummies(df["party"], prefix="party")
        X_emb = np.hstack([X_emb, enc_col.to_numpy()])
    return X_emb, y


def compute_prediction_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int,
    tune_hyperparameters: bool,
    is_regression_task: bool,
    prediction_model: str,
    scoring: str,
    random_state: int,
):
    if is_regression_task:
        if prediction_model == "linear":
            model = LinearRegression()
        elif prediction_model == "HistGB":
            model = HGBR(random_state=random_state)
        cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=random_state)
    else:
        model = HGBC(random_state=random_state)
        cv = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=n_repeats, random_state=random_state
        )
    if tune_hyperparameters and prediction_model == "HistGB":
        model = GridSearchCV(
            model,
            param_grid={
                "max_depth": [2, 4, 6, None],
                "min_samples_leaf": [4, 6, 10, 20],
            },
            scoring=scoring,
            cv=3,
        )
    start_time = time()
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=15)
    duration = time() - start_time
    return cv_scores, duration, X.shape


def prediction_scores_dfs(
    triples_dir: Union[Path, str],
    target_file: Union[Path, str],
    feature_info: Union[Path, str],
    max_depth: int,
    prediction_model: str,
    n_repeats: int,
    tune_hyperparameters: bool,
    is_regression_task: bool,
    scoring: str,
    random_state: int,
    results_file: Union[Path, str],
):
    print("DFS scores on target: ", str(target_file))
    for depth in tqdm(range(max_depth + 1)):
        # Make X, y
        X, y = make_X_y_dfs(triples_dir, target_file, feature_info, depth)
        # Compute prediction scores
        peak_memory_usage, (cv_scores, duration, X_shape) = memory_usage(
            (
                compute_prediction_scores,
                (
                    X,
                    y,
                    n_repeats,
                    tune_hyperparameters,
                    is_regression_task,
                    prediction_model,
                    scoring,
                    random_state,
                ),
            ),
            max_usage=True,
            include_children=True,
            max_iterations=1,
            retval=True,
        )
        # Save results to a dataframe
        results = {
            "triples_dir": str(triples_dir),
            "target_file": str(target_file),
            "depth": depth,
            "n_repeats": n_repeats,
            "prediction_model": prediction_model,
            "tune_hyperparameters": tune_hyperparameters,
            "scoring": scoring,
            "random_state": random_state,
            "duration": duration,
            "peak_memory": peak_memory_usage,
            "n_samples": X_shape[0],
            "n_features": X_shape[1],
            "scores": cv_scores,
        }
        df = pd.DataFrame([results])
        if Path(results_file).is_file():
            old_df = pd.read_parquet(results_file)
            df = df.append(old_df).reset_index(drop=True)
        df.to_parquet(results_file, index=False)
    return


def prediction_scores_FE(
    target_file: Union[Path, str],
    prediction_model: str,
    n_repeats: int,
    tune_hyperparameters: bool,
    is_regression_task: bool,
    scoring: str,
    random_state: int,
    results_file: Union[Path, str],
):
    """Compute and save prediction scores with handcrafted features."""
    print("FE scores on target: ", str(target_file))
    df = pd.read_parquet(target_file)
    if "raw_entities" in df.columns:
        del df["raw_entities"]
    y = df.pop("target").to_numpy()
    X = df.to_numpy()
    cv_scores, duration, X_shape = compute_prediction_scores(
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
        "target_file": str(target_file),
        "prediction_model": prediction_model,
        "n_repeats": n_repeats,
        "tune_hyperparameters": tune_hyperparameters,
        "scoring": scoring,
        "random_state": random_state,
        "duration": duration,
        "n_samples": X_shape[0],
        "n_features": X_shape[1],
        "scores": cv_scores,
    }
    df = pd.DataFrame([results])
    if Path(results_file).is_file():
        old_df = pd.read_parquet(results_file)
        df = df.append(old_df).reset_index(drop=True)
    df.to_parquet(results_file, index=False)
    return


def prediction_scores_rdf2vec(
    config_file: Union[Path, str],
    target_file: Union[Path, str],
    triples_dir: Union[Path, str],
    n_repeats: int,
    tune_hyperparameters: bool,
    is_regression_task: bool,
    prediction_model: str,
    scoring: str,
    results_file: Union[Path, str],
):
    configs = pd.read_parquet(config_file)
    dataloader = DataLoader(triples_dir, use_features="all")
    target_df = pd.read_parquet(target_file)
    mask = configs["triples_dir"] == str(triples_dir)
    mask *= configs["target_entities"].isin([str(target_file), "None"])
    configs = configs[mask].reset_index()
    for k in tqdm(range(len(configs))):
        filename = configs["filename"][k]
        random_state = configs["random_state"][k]
        if Path(results_file).is_file():
            df_res = pd.read_parquet(results_file)
        else:
            df_res = None
        if df_res is None or filename not in list(df_res["filename"]):
            X, y = make_X_y_rdf2vec(filename, dataloader, target_df)
            cv_scores, duration, X_shape = compute_prediction_scores(
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
                "filename": filename,
                "target_file": str(target_file),
                "prediction_model": prediction_model,
                "n_repeats": n_repeats,
                "tune_hyperparameters": tune_hyperparameters,
                "scoring": scoring,
                "random_state": random_state,
                "duration": duration,
                "n_samples": X_shape[0],
                "n_features": X_shape[1],
                "scores": cv_scores,
            }
            df = pd.DataFrame([results])
            if Path(results_file).is_file():
                old_df = pd.read_parquet(results_file)
                df = df.append(old_df).reset_index(drop=True)
            df.to_parquet(results_file, index=False)
    return


def prediction_scores_embeddings(
    ckpt_info: Union[Path, str],
    target_file: Union[Path, str],
    triples_dir: Union[Path, str],
    n_repeats: int,
    tune_hyperparameters: bool,
    is_regression_task: bool,
    prediction_model: str,
    scoring: str,
    results_file: Union[Path, str],
):
    # Load checkpoint and target dataframes
    checkpoints = pd.read_parquet(ckpt_info)
    mask_triples_dir = checkpoints["triples_dir"].isin(
        [
            str(triples_dir).replace("data/parietal", "storage"),
            str(triples_dir).replace("storage", "data/parietal"),
        ]
    )
    checkpoints = checkpoints[mask_triples_dir]
    checkpoints = checkpoints.reset_index(drop=True)
    target = pd.read_parquet(target_file)
    # Init dataloader
    dataloader = DataLoader(triples_dir, use_features="all")
    # Load previously stored results
    if Path(results_file).is_file():
        df_res = pd.read_parquet(results_file)
    else:
        df_res = None
    print("Embeddings scores on target: ", str(target_file))
    for k in tqdm(range(len(checkpoints))):
        model = checkpoints["emb_model_name"][k]
        ckpt_id = checkpoints["id"][k]
        random_state = checkpoints["random_state"][k]
        for epoch in checkpoints["epochs"][k]:
            if df_res is not None:
                mask = df_res["checkpoint_id"] == ckpt_id
                mask *= df_res["epoch"] == epoch
                mask *= df_res["target_file"].isin(
                    [
                        str(target_file).replace("data/parietal", "storage"),
                        str(target_file).replace("storage", "data/parietal"),
                    ]
                )
                mask *= df_res["scoring"] == scoring
                mask *= df_res["prediction_model"] == prediction_model
            if df_res is None or (~mask).all():
                print(f"### {model}, epoch {epoch}")
                ckpt_file = (
                    Path(checkpoints["checkpoint_dir"][k]) / f"{ckpt_id}_{epoch}"
                )
                X, y = make_X_y_embeddings(ckpt_file, model, dataloader, target.copy())
                peak_memory_usage, (cv_scores, duration, X_shape) = memory_usage(
                    (
                        compute_prediction_scores,
                        (
                            X,
                            y,
                            n_repeats,
                            tune_hyperparameters,
                            is_regression_task,
                            prediction_model,
                            scoring,
                            random_state,
                        ),
                    ),
                    max_usage=True,
                    include_children=True,
                    max_iterations=1,
                    retval=True,
                )
                # Save results to a dataframe
                results = {
                    "triples_dir": str(triples_dir),
                    "target_file": str(target_file),
                    "checkpoint_id": ckpt_id,
                    "epoch": epoch,
                    "n_repeats": n_repeats,
                    "tune_hyperparameters": tune_hyperparameters,
                    "prediction_model": prediction_model,
                    "scoring": scoring,
                    "random_state": random_state,
                    "duration": duration,
                    "peak_memory": peak_memory_usage,
                    "n_samples": X_shape[0],
                    "n_features": X_shape[1],
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
