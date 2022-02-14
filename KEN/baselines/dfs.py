# -*- coding: utf-8 -*-

"""A class to perform and profile Deep Feature Synthesis."""

import featuretools as ft
from memory_profiler import memory_usage
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from tqdm import tqdm
from typing import Union, List
from uuid import uuid4
from woodwork.logical_types import Categorical, Double

from KEN.dataloader import DataLoader


class DFS:
    def __init__(
        self,
        triples_dir: Union[str, Path],
        target_entities: Union[Path, str, None],
        depth: int,
        features_only: bool,
        chunk_size: int,
        n_jobs: int,
        save_dir: Union[str, Path],
        metadata_file: Union[str, Path],
    ) -> None:
        self.triples_dir = triples_dir
        self.target_entities = target_entities
        self.depth = depth
        self.features_only = features_only
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.save_dir = save_dir
        self.metadata_file = metadata_file
        return

    def run(self) -> None:
        """Run and profile DFS, then save the generated feature vectors."""
        # Run the experiment while profiling the memory usage
        peak_memory_usage = memory_usage(
            (self._run,), max_usage=True, include_children=True, max_iterations=1
        )
        if self.features_only:
            return
        # Save features and feature_matrix
        base_path = Path(self.save_dir) / str(uuid4())
        feature_path = base_path.with_suffix(".json").as_posix()
        feature_matrix_path = base_path.with_suffix(".parquet").as_posix()
        ft.save_features(self.features, feature_path)
        self.feature_matrix.to_parquet(feature_matrix_path)
        # Save metadata
        metadata = {
            "triples_dir": str(self.triples_dir),
            "target_entities": str(self.target_entities),
            "depth": self.depth,
            "chunk_size": self.chunk_size,
            "n_jobs": self.n_jobs,
            "peak_memory": peak_memory_usage,
            "build_duration": self.build_duration,
            "dfs_duration": self.dfs_duration,
            "number_of_features": self.n_features_enc,
            "feature_path": str(feature_path),
            "feature_matrix_path": str(feature_matrix_path),
        }
        df = pd.DataFrame([metadata])
        if Path(self.metadata_file).is_file():
            old_df = pd.read_parquet(self.metadata_file)
            df = df.append(old_df).reset_index(drop=True)
        df.to_parquet(self.metadata_file, index=False)
        return

    def _run(self) -> None:
        """Build EntitySet from the KG, and perform DFS."""
        # Init dataloader and target entities
        dataloader = DataLoader(self.triples_dir, use_features="all")
        if self.target_entities == "all":
            target_entities = np.arange(dataloader.n_entities)
        else:
            df = pd.read_parquet(self.target_entities)
            target_entities = (
                df["col_to_embed"].map(dataloader.entity_to_idx).unique().astype(int)
            )
        print("Build the EntitySet")
        start_time = time()
        ### Depth 0: 1-to-1 and N-to-1 relationships
        # Init dataframes
        target_entities = pd.DataFrame(target_entities, columns=["head"])
        all_entities = pd.DataFrame(np.arange(dataloader.n_entities), columns=["head"])
        n_num_rel = dataloader.n_num_attr
        df = dataloader.get_dataframe()
        df_gb = df.groupby("rel")
        rel_df_list = [df_gb.get_group(x).drop("rel", axis=1) for x in df_gb.groups]
        num_rel_df_list = rel_df_list[:n_num_rel]
        cat_rel_df_list = rel_df_list[n_num_rel:]
        cat_rel_df_list += [
            rel_df.rename({"head": "tail", "tail": "head"}, axis=1)
            for rel_df in cat_rel_df_list
        ]
        # Merge numerical relationships
        dtypes = {"head": Categorical}
        for k, num_df in enumerate(num_rel_df_list):
            if num_df["head"].is_unique:
                num_df = num_df.rename({"tail": f"N{k}"}, axis=1)
                target_entities = target_entities.merge(num_df, on="head", how="left")
                all_entities = all_entities.merge(num_df, on="head", how="left")
                dtypes[f"N{k}"] = Double
        # Merge categorical relationships
        for k, cat_df in enumerate(cat_rel_df_list):
            if cat_df["head"].is_unique:
                cat_df = cat_df.rename({"tail": f"C{k}"}, axis=1)
                target_entities = target_entities.merge(cat_df, on="head", how="left")
                all_entities = all_entities.merge(cat_df, on="head", how="left")
                dtypes[f"C{k}"] = Categorical
        # Init EntitySet
        es = ft.EntitySet("dfs")
        es = es.add_dataframe(
            dataframe_name="target_entities",
            dataframe=target_entities,
            index="head",
            logical_types=dtypes,
        )
        ### Depth 1: 1-to-N and N-to-N relationships
        if self.depth >= 1:
            # Add numerical dataframes
            for k, num_df in enumerate(num_rel_df_list):
                if not num_df["head"].is_unique:
                    es.add_dataframe(
                        dataframe_name=f"N{k}",
                        dataframe=num_df,
                        index="index",
                        make_index=True,
                        logical_types={"head": Categorical, "tail": Double},
                    )
                    es.add_relationship("target_entities", "head", f"N{k}", "head")
            # Add categorical dataframes
            prev_df_names = []
            for k, cat_df in enumerate(cat_rel_df_list):
                if not cat_df["head"].is_unique:
                    es.add_dataframe(
                        dataframe_name=f"C{k}",
                        dataframe=cat_df,
                        index="index",
                        make_index=True,
                        logical_types={"head": Categorical, "tail": Categorical},
                    )
                    es.add_relationship("target_entities", "head", f"C{k}", "head")
                    prev_df_names.append(f"C{k}")
        ### Depth >= 2: Repeat the previous steps in a tree-like structure
        current_depth = 1
        ignore_columns = {}
        new_df_names = []
        while self.depth > current_depth:
            # Loop over previous categorical dataframes
            for prev_df_name in tqdm(prev_df_names):
                # Current depth = 1, 3, 5... : Make next layer with depth 0 dataframes
                tail_name = f"tail_{prev_df_name}"
                if current_depth % 2 == 1:
                    mask = all_entities["head"].isin(es[prev_df_name]["tail"])
                    tail_df = all_entities[mask]
                    es.add_dataframe(
                        dataframe_name=tail_name,
                        dataframe=tail_df,
                        index="head",
                        logical_types=dtypes,
                    )
                    es.add_relationship(tail_name, "head", prev_df_name, "tail")
                    ignore_columns[prev_df_name] = ["head"]
                # Current depth = 2, 4, 6... : Make next layer with depth 1 dataframes
                else:
                    # Add numerical dataframes
                    for j, num_df in enumerate(num_rel_df_list):
                        if not num_df["head"].is_unique:
                            df_name = f"N{j}_{prev_df_name}"
                            es.add_dataframe(
                                dataframe_name=df_name,
                                dataframe=num_df.copy(),
                                index="index",
                            )
                            es.add_relationship(tail_name, "head", df_name, "head")
                    # Add categorical dataframes
                    for j, cat_df in enumerate(cat_rel_df_list):
                        if not cat_df["head"].is_unique:
                            df_name = f"C{j}_{prev_df_name}"
                            es.add_dataframe(
                                dataframe_name=df_name,
                                dataframe=cat_df.copy(),
                                index="index",
                            )
                            es.add_relationship(tail_name, "head", df_name, "head")
                            new_df_names.append(df_name)
                    prev_df_names = new_df_names
                    new_df_names = []
            current_depth += 1
        ### Run Deep Feature Synthesis
        print("Compute Deep Features")
        if self.features_only:
            features = ft.dfs(
                entityset=es,
                target_dataframe_name="target_entities",
                n_jobs=self.n_jobs,
                max_depth=self.depth,
                features_only=True,
                ignore_columns=ignore_columns,
                chunk_size=self.chunk_size,
                verbose=True,
            )
            n_features = len(features)
            # Add one-hot encoded features
            n_cat_features = 0
            for f in features:
                if f._name[:4] == "MODE":
                    n_cat_features += 1
            n_features_enc = n_features + 9 * n_cat_features
            print("Number of features = ", n_features_enc)
            return
        
        else:
            self.raised_memory_error = False
            mid_time = time()
            feature_matrix, features = ft.dfs(
                entityset=es,
                target_dataframe_name="target_entities",
                n_jobs=self.n_jobs,
                max_depth=self.depth,
                features_only=False,
                ignore_columns=ignore_columns,
                chunk_size=self.chunk_size,
                verbose=True,
            )
            # Encode categorical features
            feature_matrix_enc, features_enc = ft.encode_features(
                feature_matrix, features, top_n=10
            )
            feature_matrix_enc, features_enc = ft.selection.remove_single_value_features(
                feature_matrix_enc, features=features_enc, count_nan_as_value=True
            )
            # Remove single value features
            feature_matrix, features = ft.selection.remove_single_value_features(
                feature_matrix, features=features, count_nan_as_value=True
            )
            end_time = time()
            self.feature_matrix, self.features = feature_matrix, features
            self.n_features_enc = feature_matrix_enc.shape[1]
            self.build_duration = mid_time - start_time
            self.dfs_duration = end_time - mid_time
        return
