# -*- coding: utf-8 -*-

"""
Compute cross-validation scores on the target, using DFS, manual feature engineering or 
embeddings (TransE, DistMult, MuRE, RDF2Vec) as features for the entities of interest.
"""

from pathlib import Path
import pandas as pd
from typing import Union
import warnings

from KEN.evaluation import (
    prediction_scores_dfs,
    prediction_scores_embeddings,
    prediction_scores_FE,
    prediction_scores_rdf2vec,
)

exp_dir = Path(__file__).absolute().parents[1]
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def compute_prediction_scores(
    target_file: Union[Path, str],
    target_file_FE: Union[Path, str],
    triples_dir: Union[Path, str],
    feature_info: Union[Path, str],
    checkpoint_info: Union[Path, str],
    rdf2vec_config: Union[Path, str],
    prediction_model: str,
    n_repeats: int,
    tune_hyperparameters: bool,
    is_regression_task: bool,
    scoring: str,
    random_state: int,
    max_depth_dfs: int,
    mode: str,
):
    if mode in ["DFS", "all"]:
        # Deep Feature Synthesis
        prediction_scores_dfs(
            triples_dir,
            target_file,
            feature_info,
            max_depth_dfs,
            prediction_model,
            n_repeats,
            tune_hyperparameters,
            is_regression_task,
            scoring,
            random_state,
            results_file=exp_dir / "prediction_scores/scores_dfs.parquet",
        )
    if mode in ["embedding", "all"]:
        # KG Embeddings
        prediction_scores_embeddings(
            checkpoint_info,
            target_file,
            triples_dir,
            n_repeats,
            tune_hyperparameters,
            is_regression_task,
            prediction_model,
            scoring,
            results_file=exp_dir / "prediction_scores/scores_embeddings.parquet",
        )
    if mode in ["FE", "all"]:
        # Manual feature engineering
        prediction_scores_FE(
            target_file_FE,
            prediction_model,
            n_repeats,
            tune_hyperparameters,
            is_regression_task,
            scoring,
            random_state,
            results_file=exp_dir / "prediction_scores/scores_FE.parquet",
        )
    if mode in ["rdf2vec", "all"]:
        prediction_scores_rdf2vec(
            rdf2vec_config,
            target_file,
            triples_dir,
            n_repeats,
            tune_hyperparameters,
            is_regression_task,
            prediction_model,
            scoring,
            results_file=exp_dir / "prediction_scores/scores_rdf2vec.parquet",
        )
    return


def kdd14(mode="all"):
    for triples_dir in [
        "datasets/kdd14/triplets",
        "datasets/kdd14/triplets_uniform",
        "datasets/kdd14/triplets_cardinality",
        #"datasets/kdd14/triplets_types",
    ]:
        print("Triples dir: ", triples_dir)
        compute_prediction_scores(
            target_file=exp_dir / "datasets/kdd14/target.parquet",
            target_file_FE=exp_dir / "manual_feature_engineering/kdd14.parquet",
            triples_dir=exp_dir / triples_dir,
            feature_info=exp_dir / "deep_feature_synthesis/feature_info.parquet",
            checkpoint_info=exp_dir / "model_training/checkpoints_info.parquet",
            rdf2vec_config=exp_dir / "model_training/rdf2vec_config.parquet",
            prediction_model="HistGB",
            n_repeats=3,
            tune_hyperparameters=False,
            is_regression_task=False,
            scoring="average_precision",
            random_state=0,
            max_depth_dfs=3,
            mode=mode,
        )
    return


def kdd15(mode="all"):
    for triples_dir in [
        "datasets/kdd15/triplets",
        "datasets/kdd15/triplets_uniform",
        "datasets/kdd15/triplets_cardinality",
        #"datasets/kdd15/triplets_types",
    ]:
        print("Triples dir: ", triples_dir)
        compute_prediction_scores(
            target_file=exp_dir / "datasets/kdd15/target.parquet",
            target_file_FE=exp_dir / "manual_feature_engineering/kdd15.parquet",
            triples_dir=exp_dir / triples_dir,
            feature_info=exp_dir / "deep_feature_synthesis/feature_info.parquet",
            checkpoint_info=exp_dir / "model_training/checkpoints_info.parquet",
            rdf2vec_config=exp_dir / "model_training/rdf2vec_config.parquet",
            prediction_model="HistGB",
            n_repeats=3,
            tune_hyperparameters=False,
            is_regression_task=False,
            scoring="roc_auc",
            random_state=0,
            max_depth_dfs=3,
            mode=mode,
        )
    return


def us_elections(mode="all", yago_subset=False):
    if yago_subset:
        triples_dir = exp_dir / "datasets/yago3/triplets/only_counties"
        mode = "embedding"
    else:
        triples_dir = exp_dir / "datasets/yago3/triplets/full"
    compute_prediction_scores(
        target_file=exp_dir / "datasets/us_elections/target_log.parquet",
        target_file_FE=exp_dir / "manual_feature_engineering/us_elections.parquet",
        triples_dir=triples_dir,
        feature_info=exp_dir / "deep_feature_synthesis/feature_info.parquet",
        checkpoint_info=exp_dir / "model_training/checkpoints_info.parquet",
        rdf2vec_config=exp_dir / "model_training/rdf2vec_config.parquet",
        prediction_model="HistGB",
        n_repeats=5,
        tune_hyperparameters=True,
        is_regression_task=True,
        scoring="r2",
        random_state=0,
        max_depth_dfs=3,
        mode=mode,
    )
    return


def housing_prices(mode="all", yago_subset=False):
    if yago_subset:
        triples_dir = exp_dir / "datasets/yago3/triplets/only_cities"
        mode = "embedding"
    else:
        triples_dir = exp_dir / "datasets/yago3/triplets/full"
    compute_prediction_scores(
        target_file=exp_dir / "datasets/housing_prices/target_log.parquet",
        target_file_FE=exp_dir / "manual_feature_engineering/housing_prices.parquet",
        triples_dir=triples_dir,
        feature_info=exp_dir / "deep_feature_synthesis/feature_info.parquet",
        checkpoint_info=exp_dir / "model_training/checkpoints_info.parquet",
        rdf2vec_config=exp_dir / "model_training/rdf2vec_config.parquet",
        prediction_model="HistGB",
        n_repeats=5,
        tune_hyperparameters=True,
        is_regression_task=True,
        scoring="r2",
        random_state=0,
        max_depth_dfs=3,
        mode=mode,
    )
    return


def us_accidents(mode="all", target="counts", yago_subset=False):
    if yago_subset:
        triples_dir = exp_dir / "datasets/yago3/triplets/subset_us_accidents"
        mode = "embedding"
    else:
        triples_dir = exp_dir / "datasets/yago3/triplets/full"
    if target == "counts":
        target_file = exp_dir / "datasets/us_accidents/counts.parquet"
    elif target == "duration":
        target_file = exp_dir / "datasets/us_accidents/duration.parquet"
    compute_prediction_scores(
        target_file=target_file,
        target_file_FE=exp_dir / "manual_feature_engineering/us_accidents.parquet",
        triples_dir=triples_dir,
        feature_info=exp_dir / "deep_feature_synthesis/feature_info.parquet",
        checkpoint_info=exp_dir / "model_training/checkpoints_info.parquet",
        rdf2vec_config=exp_dir / "model_training/rdf2vec_config.parquet",
        prediction_model="HistGB",
        n_repeats=5,
        tune_hyperparameters=True,
        is_regression_task=True,
        scoring="r2",
        random_state=0,
        max_depth_dfs=3,
        mode=mode,
    )
    return


def movie_revenues(mode="all", yago_subset=False):
    if yago_subset:
        triples_dir = exp_dir / "datasets/yago3/triplets/subset_movie_revenues"
        mode = "embedding"
    else:
        triples_dir = exp_dir / "datasets/yago3/triplets/full"
    compute_prediction_scores(
        target_file=exp_dir / "datasets/movie_revenues/target.parquet",
        target_file_FE=exp_dir / "manual_feature_engineering/movie_revenues.parquet",
        triples_dir=triples_dir,
        feature_info=exp_dir / "deep_feature_synthesis/feature_info.parquet",
        checkpoint_info=exp_dir / "model_training/checkpoints_info.parquet",
        rdf2vec_config=exp_dir / "model_training/rdf2vec_config.parquet",
        prediction_model="HistGB",
        n_repeats=5,
        tune_hyperparameters=True,
        is_regression_task=True,
        scoring="r2",
        random_state=0,
        max_depth_dfs=3,
        mode=mode,
    )
    return


def company_employees(mode="all", yago_subset=False):
    if yago_subset:
        triples_dir = exp_dir / "datasets/yago3/triplets/subset_company_employees"
        mode = "embedding"
    else:
        triples_dir = exp_dir / "datasets/yago3/triplets/full"
    compute_prediction_scores(
        target_file=exp_dir / "datasets/company_employees/target.parquet",
        target_file_FE=exp_dir / "manual_feature_engineering/company_employees.parquet",
        triples_dir=triples_dir,
        feature_info=exp_dir / "deep_feature_synthesis/feature_info.parquet",
        checkpoint_info=exp_dir / "model_training/checkpoints_info.parquet",
        rdf2vec_config=exp_dir / "model_training/rdf2vec_config.parquet",
        prediction_model="HistGB",
        n_repeats=5,
        tune_hyperparameters=True,
        is_regression_task=True,
        scoring="r2",
        random_state=0,
        max_depth_dfs=3,
        mode=mode,
    )
    return


if __name__ == "__main__":
    # Compute prediction scores for all methods
    kdd14(mode="all")
    kdd15(mode="all")
    us_elections(mode="all")
    housing_prices(mode="all")
    us_accidents(mode="all")
    movie_revenues(mode="all")
    company_employees(mode="all")
    # Compute prediction scores on yago3 subsets
    us_elections(yago_subset=True)
    housing_prices(yago_subset=True)
    us_accidents(yago_subset=True)
    movie_revenues(yago_subset=True)
    company_employees(yago_subset=True)

