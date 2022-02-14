# -*- coding: utf-8 -*-

"""
Functions to train and save embedding models on each dataset.
"""

from pathlib import Path
from torch.optim import Adam
import torch.nn as nn
import resource

from pykeen.losses import MarginRankingLoss, SoftplusLoss
from pykeen.models import DistMultLiteralGated

from KEN.models.entity_embedding import DistMult, TransE, MuRE
from KEN.models.numerical_embedding import (
    BinnedEmbedding,
    LinearEmbedding2,
)
from KEN.sampling import PseudoTypedNegativeSampler
from KEN.training import HPPTrainer

exp_dir = Path(__file__).absolute().parents[1]


def mure(gpu):
    print("----- TRAIN MuRE -----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd14/triplets_uniform", [0]),
        ("datasets/kdd14/triplets_cardinality", [0]),
        ("datasets/kdd14/triplets_types", [0]),

        # ("datasets/kdd15/triplets", [0]),
        ("datasets/kdd15/triplets_uniform", [0]),
        ("datasets/kdd15/triplets_cardinality", [0]),
        ("datasets/kdd15/triplets_types", [0]),

        ("datasets/yago3/triplets/full", [0, 1, 2]),

        ("datasets/yago3/triplets/only_counties", [0, 1, 2]),
        ("datasets/yago3/triplets/only_cities", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_us_accidents", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_movie_revenues", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_company_employees", [0, 1, 2]),
        # ("datasets/yago4/triplets", [0]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="categorical",
                emb_model=MuRE,
                emb_params={"embedding_dim": 200},
                num_emb_model=None,
                num_emb_params={},
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=SoftplusLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": [],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[2, 4, 8, 16, 24, 32, 40],
                sub_batch_size=None,
                random_state=seed,
                device=f"cuda:{gpu}",
                # model_device=f"cpu",
                # forward_device=f"cpu",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def mure_ken(gpu):
    print("----- TRAIN MuRE + KEN without ReLU -----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd15/triplets", [0]),
        ("datasets/yago3/triplets/full", [0, 1, 2]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="all",
                emb_model=MuRE,
                emb_params={"embedding_dim": 200},
                num_emb_model=LinearEmbedding2,
                num_emb_params={
                    "activation_function": None,
                    "embedding_dim": 200,
                    "initializer": "MuRE",
                    "input_transformer": "quantile",
                    "p_norm": None,
                },
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=SoftplusLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": [],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[2, 4, 8, 16, 24, 32, 40],
                sub_batch_size=None,
                random_state=seed,
                device=f"cuda:{gpu}",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def mure_ken_relu(gpu):
    print("----- TRAIN MuRE + KEN with ReLU -----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd14/triplets_uniform", [0]),
        ("datasets/kdd14/triplets_cardinality", [0]),
        # ("datasets/kdd14/triplets_types", [0]),
        
        ("datasets/kdd15/triplets", [0]),
        ("datasets/kdd15/triplets_uniform", [0]),
        ("datasets/kdd15/triplets_cardinality", [0]),
        # ("datasets/kdd15/triplets_types", [0]),

        ("datasets/yago3/triplets/full", [0, 1, 2]),

        ("datasets/yago3/triplets/only_counties", [0, 1, 2]),
        ("datasets/yago3/triplets/only_cities", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_us_accidents", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_movie_revenues", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_company_employees", [0, 1, 2]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="all",
                emb_model=MuRE,
                emb_params={"embedding_dim": 200},
                num_emb_model=LinearEmbedding2,
                num_emb_params={
                    "activation_function": "relu",
                    "embedding_dim": 200,
                    "initializer": "MuRE",
                    "input_transformer": "quantile",
                    "p_norm": None,
                },
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=SoftplusLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": [],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[2, 4, 8, 16, 24, 32, 40],
                sub_batch_size=None,
                random_state=seed,
                device=f"cuda:{gpu}",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def mure_ken_relu_no_quantile(gpu):
    print("----- TRAIN MuRE + KEN with ReLU no quantile-----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd15/triplets", [0]),
        ("datasets/yago3/triplets/full", [0, 1, 2]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="all",
                emb_model=MuRE,
                emb_params={"embedding_dim": 200},
                num_emb_model=LinearEmbedding2,
                num_emb_params={
                    "activation_function": "relu",
                    "embedding_dim": 200,
                    "initializer": "MuRE",
                    "input_transformer": "min_max",
                    "p_norm": None,
                },
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=SoftplusLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": [],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[2, 4, 8, 16, 24, 32, 40],
                sub_batch_size=None,
                random_state=seed,
                device=f"cuda:{gpu}",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def mure_binning(gpu):
    print("----- TRAIN MuRE + Binning -----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd15/triplets", [0]),
        ("datasets/yago3/triplets/full", [0, 1, 2]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="all",
                emb_model=MuRE,
                emb_params={"embedding_dim": 200},
                num_emb_model=BinnedEmbedding,
                num_emb_params={
                    "n_bins": 20,
                    "embedding_dim": 200,
                    "initializer": "MuRE",
                    "input_transformer": "quantile",
                    "p_norm": None,
                },
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=SoftplusLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": [],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[2, 4, 8, 16, 24, 32, 40],
                sub_batch_size=None,
                random_state=seed,
                device=f"cuda:{gpu}",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def transe(gpu):
    print("----- TRAIN TransE -----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd15/triplets", [0]),
        ("datasets/yago3/triplets/full", [0, 1, 2]),
        ("datasets/yago3/triplets/only_counties", [0, 1, 2]),
        ("datasets/yago3/triplets/only_cities", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_us_accidents", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_movie_revenues", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_company_employees", [0, 1, 2]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="categorical",
                emb_model=TransE,
                emb_params={"embedding_dim": 200},
                num_emb_model=None,
                num_emb_params={},
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=MarginRankingLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                    "margin": [4],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": ["margin"],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[2, 4, 8, 16, 24, 32, 40],
                sub_batch_size=None,
                random_state=seed,
                device=f"cuda:{gpu}",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def transe_ken_relu(gpu):
    print("----- TRAIN TransE + KEN with ReLU -----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd15/triplets", [0]),
        ("datasets/yago3/triplets/full", [0, 1, 2]),
        ("datasets/yago3/triplets/only_counties", [0, 1, 2]),
        ("datasets/yago3/triplets/only_cities", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_us_accidents", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_movie_revenues", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_company_employees", [0, 1, 2]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="all",
                emb_model=TransE,
                emb_params={"embedding_dim": 200},
                num_emb_model=LinearEmbedding2,
                num_emb_params={
                    "activation_function": "relu",
                    "embedding_dim": 200,
                    "initializer": "TransE",
                    "input_transformer": "quantile",
                    "p_norm": None,
                },
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=MarginRankingLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                    "margin": [4],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": ["margin"],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[2, 4, 8, 16, 24, 32, 40],
                sub_batch_size=10000,
                random_state=seed,
                device=f"cuda:{gpu}",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def distmult_literalE(gpu):
    print("----- TRAIN DistMult + LiteralE -----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd15/triplets", [0]),
        ("datasets/yago3/triplets/full", [0, 1, 2]),
        ("datasets/yago3/triplets/only_counties", [0, 1, 2]),
        ("datasets/yago3/triplets/only_cities", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_us_accidents", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_movie_revenues", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_company_employees", [0, 1, 2]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="all",
                emb_model=DistMultLiteralGated,
                emb_params={"embedding_dim": 200},
                num_emb_model=None,
                num_emb_params={},
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=SoftplusLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": [],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[2, 4, 8, 16, 24, 32, 40],
                sub_batch_size=None,
                random_state=seed,
                device=f"cuda:{gpu}",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def distmult(gpu):
    print("----- TRAIN DistMult -----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd15/triplets", [0]),
        ("datasets/yago3/triplets/full", [0, 1, 2]),
        ("datasets/yago3/triplets/only_counties", [0, 1, 2]),
        ("datasets/yago3/triplets/only_cities", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_us_accidents", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_movie_revenues", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_company_employees", [0, 1, 2]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="categorical",
                emb_model=DistMult,
                emb_params={"embedding_dim": 200},
                num_emb_model=None,
                num_emb_params={},
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=SoftplusLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": [],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[4, 8, 16, 24, 32, 40],
                sub_batch_size=None,
                random_state=seed,
                device=f"cuda:{gpu}",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def distmult_ken(gpu):
    print("----- TRAIN DistMult + KEN -----")
    for triples_dir, seeds in (
        ("datasets/kdd14/triplets", [0]),
        ("datasets/kdd15/triplets", [0]),
        ("datasets/yago3/triplets/full", [0, 1, 2]),
        ("datasets/yago3/triplets/only_counties", [0, 1, 2]),
        ("datasets/yago3/triplets/only_cities", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_us_accidents", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_movie_revenues", [0, 1, 2]),
        ("datasets/yago3/triplets/subset_company_employees", [0, 1, 2]),
    ):
        for seed in seeds:
            print("### ", triples_dir, "seed = ", seed)
            trainer = HPPTrainer(
                triples_dir=exp_dir / triples_dir,
                use_features="all",
                emb_model=DistMult,
                emb_params={"embedding_dim": 200},
                num_emb_model=LinearEmbedding2,
                num_emb_params={
                    "activation_function": "relu",
                    "embedding_dim": 200,
                    "initializer": "DistMult",
                    "input_transformer": "quantile",
                    "p_norm": None,
                },
                negative_sampler=PseudoTypedNegativeSampler,
                sampler_params={},
                loss=SoftplusLoss,
                loss_params={},
                optimizer=Adam,
                optimizer_params={},
                param_distributions={
                    "lr": [1e-3],
                    "num_negs_per_pos": [10],
                    "batch_size": [100000],
                },
                param_mapping={
                    "emb_model": [],
                    "num_emb_model": [],
                    "negative_sampler": ["num_negs_per_pos"],
                    "loss": [],
                    "optimizer": ["lr"],
                },
                n_param_samples=1,
                num_epoch_list=[4, 8, 16, 24, 32, 40],
                sub_batch_size=None,
                random_state=seed,
                device=f"cuda:{gpu}",
                checkpoint_dir=exp_dir / "model_training/checkpoints",
                saved_dataframe=exp_dir / "model_training/checkpoints_info.parquet",
            )
            trainer.run()
    return


def set_memory_limit(max_memory: int):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory * 1024 ** 3, hard))
    return


if __name__ == "__main__":
    set_memory_limit(400)
    ### MuRE
    mure(gpu=0)
    mure_ken_relu(gpu=0)
    mure_ken(gpu=0)
    mure_ken_relu_no_quantile(gpu=0)
    mure_binning(gpu=0)
    ### TransE
    transe(gpu=0)
    transe_ken_relu(gpu=0)
    ### DistMult
    distmult(gpu=0)
    distmult_ken(gpu=0)
    distmult_literalE(gpu=0)
