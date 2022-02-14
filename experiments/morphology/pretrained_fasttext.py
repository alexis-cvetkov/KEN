# -*- coding: utf-8 -*-

"""
Code to compute prediction scores with fuzzy matching.
Target entity names are encoded using a CountVectorizer, and then matched
to the closest entity in YAGO based on their cosine similiarity.
"""

import fasttext
import fasttext.util
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Union, List

from KEN.evaluation.prediction_scores import compute_prediction_scores

exp_dir = Path(__file__).absolute().parents[1]


def make_X_y_fasttext(
    target_file: Union[str, Path], ft_model: fasttext.FastText._FastText
):
    """Load target dataframe and encode raw entity names with a pretrained
    fasttext model."""
    # Load dataframe
    df = pd.read_parquet(target_file)
    # Encode raw entity names. We use the title method because fasttext is case sensitive.
    # We also remove potential "\n" since they change the embeddings.
    X_emb = np.zeros((len(df), 200))
    raw_entities = df["raw_entities"].str.title().str.replace("\n", "")
    for k, x in enumerate(raw_entities):
        X_emb[k] = ft_model.get_sentence_vector(x)
    # Add party column (only for us_elections)
    if "party" in df.columns:
        enc_col = pd.get_dummies(df["party"], prefix="party")
        X_emb = np.hstack([X_emb, enc_col.to_numpy()])
    y = df["target"]
    return X_emb, y


def prediction_scores(
    target_files: List[Union[Path, str]],
    n_repeats: int,
    tune_hyperparameters: bool,
    is_regression_task: bool,
    prediction_model: str,
    scoring: str,
    results_file: Union[Path, str],
):
    # Load previously stored results
    if Path(results_file).is_file():
        df_res = pd.read_parquet(results_file)
    else:
        df_res = None
    # Load fasttext model and reduce it to 200-dimensional embeddings
    print("Load fasttext model")
    ft_model = fasttext.load_model(str(exp_dir / "morphology/cc.en.300.bin"))
    fasttext.util.reduce_model(ft_model, 200)
    for target_file in target_files:
        # Make X, y with fasttext embeddings
        X, y = make_X_y_fasttext(target_file, ft_model)
        print("Start computing cross-validation scores for", str(target_file))
        for random_state in range(3):
            # Check if results for the same config are in df_res already
            if df_res is not None:
                mask = df_res["target_file"] == str(target_file)
                mask *= df_res["scoring"] == scoring
                mask *= df_res["prediction_model"] == prediction_model
                mask *= df_res["tune_hyperparameters"] == tune_hyperparameters
                mask *= df_res["random_state"] == random_state
                # If not, compute prediction scores
            if df_res is None or (~mask).all():
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
                    "target_file": str(target_file),
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


if __name__ == "__main__":

    target_files = [
        exp_dir / "datasets/us_elections/target_log.parquet",
        exp_dir / "datasets/housing_prices/target_log.parquet",
        exp_dir / "datasets/us_accidents/counts.parquet",
        exp_dir / "datasets/movie_revenues/target.parquet",
        exp_dir / "datasets/company_employees/target.parquet",
    ]
    prediction_scores(
        target_files=target_files,
        n_repeats=5,
        tune_hyperparameters=True,
        is_regression_task=True,
        prediction_model="HistGB",
        scoring="r2",
        results_file=exp_dir / "morphology/scores_fasttext.parquet",
    )
