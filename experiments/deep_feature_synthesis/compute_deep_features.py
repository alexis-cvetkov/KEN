# -*- coding: utf-8 -*-

""" Functions to run Deep Feature Synthesis on each dataset. We compute deep features
either on a subset of all entities (for instance only counties) or on all entities.
When the dataset is too big to run DFS, we can still get the number of generated
features using the `features_only` parameter. """

import gc
from pathlib import Path
import resource
from typing import Union, List

from KEN.baselines import DFS

exp_dir = Path(__file__).absolute().parents[1]


def set_memory_limit(max_memory: int):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory * 1024 ** 3, hard))
    return


def compute_deep_features(
    triples_dir: Union[Path, str],
    target_entities: Union[Path, str],
    depth_range: List[int],
    features_only: bool,
    chunk_size_list: List[Union[int, None]],
    n_jobs_list: List[int],
):
    # Init chunk_size to None (no chunks)
    for k, depth in enumerate(depth_range):
        print(f"### Depth {depth}")
        dfs = DFS(
            triples_dir=triples_dir,
            target_entities=target_entities,
            depth=depth,
            features_only=features_only,
            chunk_size=chunk_size_list[k],
            n_jobs=n_jobs_list[k],
            save_dir=exp_dir / "deep_feature_synthesis/saved_features",
            metadata_file=exp_dir / "deep_feature_synthesis/feature_info.parquet",
        )
        dfs.run()
        # Free memory
        del dfs
        gc.collect()
    return


def kdd14(all_entities=False, features_only=False):
    print("Compute deep features for KDD14")
    entities = "all" if all_entities else exp_dir / "datasets/kdd14/target.parquet"
    compute_deep_features(
        triples_dir=exp_dir / "datasets/kdd14/triplets",
        target_entities=entities,
        depth_range=[0, 1, 2, 3],
        features_only=features_only,
        chunk_size_list=[None, None, None, 0.001],
        n_jobs_list=[1, 1, 1, 1],
    )
    return


def kdd15(all_entities=False, features_only=False):
    print("Compute deep features for KDD15")
    entities = "all" if all_entities else exp_dir / "datasets/kdd15/target.parquet"
    compute_deep_features(
        triples_dir=exp_dir / "datasets/kdd15/triplets",
        target_entities=entities,
        depth_range=[0, 1, 2, 3],
        features_only=features_only,
        chunk_size_list=[None, None, None, 0.001],
        n_jobs_list=[1, 1, 1, 1],
    )
    return


def yago3_all_entities(features_only=False):
    print("Compute deep features for all entities in YAGO3")
    compute_deep_features(
        triples_dir=exp_dir / "datasets/yago3/triplets/full",
        target_entities="all",
        depth_range=[0, 1, 2, 3],
        features_only=features_only,
        chunk_size_list=[None, None, 0.001, 0.001],
        n_jobs_list=[1, 1, 1, 1],
    )
    return


def yago3_us_elections(features_only=False):
    print("Compute deep features for counties from YAGO3")
    compute_deep_features(
        triples_dir=exp_dir / "datasets/yago3/triplets/full",
        target_entities=exp_dir / "datasets/us_elections/target.parquet",
        depth_range=[0, 1, 2, 3],
        features_only=features_only,
        chunk_size_list=[None, None, None, 0.001],
        n_jobs_list=[1, 1, 1, 1],
    )
    return


def yago3_housing_prices(features_only=False):
    print("Compute deep features for cities from YAGO3")
    compute_deep_features(
        triples_dir=exp_dir / "datasets/yago3/triplets/full",
        target_entities=exp_dir / "datasets/housing_prices/target.parquet",
        depth_range=[0, 1, 2, 3],
        features_only=features_only,
        chunk_size_list=[None, None, None, 0.001],
        n_jobs_list=[1, 1, 1, 1],
    )
    return


def yago3_us_accidents(features_only=False):
    print("Compute deep features for us_accidents cities from YAGO3")
    compute_deep_features(
        triples_dir=exp_dir / "datasets/yago3/triplets/full",
        target_entities=exp_dir / "datasets/us_accidents/counts.parquet",
        depth_range=[0, 1, 2],
        features_only=features_only,
        chunk_size_list=[None, None, None],
        n_jobs_list=[1, 1, 1],
    )
    return


def yago3_movie_revenues(features_only=False):
    print("Compute deep features for movies from YAGO3")
    compute_deep_features(
        triples_dir=exp_dir / "datasets/yago3/triplets/full",
        target_entities=exp_dir / "datasets/movie_revenues/target.parquet",
        depth_range=[0, 1, 2],
        features_only=features_only,
        chunk_size_list=[None, None, None],
        n_jobs_list=[1, 1, 1],
    )
    return


def yago3_company_employees(features_only=False):
    print("Compute deep features for companies from YAGO3")
    compute_deep_features(
        triples_dir=exp_dir / "datasets/yago3/triplets/full",
        target_entities=exp_dir / "datasets/company_employees/target.parquet",
        depth_range=[0, 1, 2, 3],
        features_only=features_only,
        chunk_size_list=[None, None, None, 0.001],
        n_jobs_list=[1, 1, 1, 1],
    )
    return


def yago4_all_entities(features_only=False):
    print("Compute deep features for companies from YAGO3")
    compute_deep_features(
        triples_dir=exp_dir / "datasets/yago4/triplets",
        target_entities="all",
        depth_range=[1],
        features_only=features_only,
        chunk_size_list=[0.001],
        n_jobs_list=[1],
    )
    return


if __name__ == "__main__":
    set_memory_limit(400)  # Set memory limit to 400 Gb
    # Compute deep features for target entities only (e.g. US counties)
    kdd14(all_entities=False)
    kdd15(all_entities=False)
    yago3_us_elections()
    yago3_housing_prices()
    yago3_us_accidents()
    yago3_movie_revenues()
    yago3_company_employees()
    # Compute deep features for all entities
    kdd14(all_entities=True)
    kdd15(all_entities=True)
    yago3_all_entities(features_only=False)
    # yago4_all_entities()
    # Compute the number of features only
    yago3_all_entities(features_only=True)

# def yago4_housing_prices(features_only=False):
#     print("Compute deep features for cities from YAGO4")
#     compute_deep_features(
#         triples_dir=exp_dir / "datasets/yago4/triplets",
#         target_entities=exp_dir / "datasets/housing_prices/target_yago4.parquet",
#         depth_range=[0, 1, 2, 3],
#         features_only=features_only,
#         chunk_size_list=[None, None, None, 0.001],
#         n_jobs_list=[1, 1, 1, 1],
#     )
#     return
