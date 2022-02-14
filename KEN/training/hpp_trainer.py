# -*- coding: utf-8 -*-

"""A class to easily train and save multiple embedding models for a range of
hyper-parameters. It also creates a csv file that maps parameters to the associated
checkpoints."""

import copy
import hashlib
import json
from memory_profiler import memory_usage
import pandas as pd
from pathlib import Path
from sklearn.model_selection import ParameterSampler
from time import time
import torch
from torch.optim import Optimizer
from typing import Union, Tuple, List

from KEN.dataloader import DataLoader
from KEN.models.numerical_embedding import NumericalEmbeddingModel
from KEN.sampling import PseudoTypedNegativeSampler
from KEN.training import SLCWATrainingLoop

from pykeen.losses import Loss
from pykeen.models import Model
from pykeen.sampling import NegativeSampler


class HPPTrainer:
    def __init__(
        self,
        triples_dir: Path,
        use_features: str,
        emb_model: Model,
        emb_params: dict,
        num_emb_model: Union[NumericalEmbeddingModel, None],
        num_emb_params: dict,
        negative_sampler: NegativeSampler,
        sampler_params: dict,
        loss: Loss,
        loss_params: dict,
        optimizer: Optimizer,
        optimizer_params: dict,
        param_distributions: dict,
        param_mapping: dict,
        n_param_samples: int,
        num_epoch_list: List[int],
        # batch_size: int,
        sub_batch_size: int,
        random_state: int,
        device: str,
        checkpoint_dir: Union[str, Path],
        saved_dataframe: Union[str, Path],
    ) -> None:
        # Dataloader
        self.triples_dir = triples_dir
        self.use_features = use_features
        # Embedding model
        self.emb_model = emb_model
        self.emb_params = emb_params
        # Numerical embedding model
        self.num_emb_model = num_emb_model
        self.num_emb_params = num_emb_params
        # Negative sampler
        self.negative_sampler = negative_sampler
        self.sampler_params = sampler_params
        # Loss function
        self.loss = loss
        self.loss_params = loss_params
        # Optimizer
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        # Hyper-parameters sampling
        self.param_distributions = param_distributions
        self.param_mapping = param_mapping
        self.n_param_samples = n_param_samples
        # Training params
        self.num_epoch_list = num_epoch_list
        # self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.random_state = random_state
        # Other params
        self.device = device
        self.checkpoint_dir = checkpoint_dir.as_posix()
        self.saved_dataframe = saved_dataframe.as_posix()
        self.config_list = []
        return

    def run(self) -> None:
        # Run the experiment while profiling the memory usage
        peak_memory_usage = memory_usage(
            (self._run,), max_usage=True, include_children=True, max_iterations=1
        )
        # Store the configs in a dataframe
        df = pd.json_normalize(self.config_list, sep="_")
        df["peak_memory_usage"] = peak_memory_usage
        if Path(self.saved_dataframe).is_file():
            old_df = pd.read_parquet(self.saved_dataframe)
            df = df.append(old_df).reset_index(drop=True)
        df.to_parquet(self.saved_dataframe, index=False)
        del self.training_loop
        torch.cuda.empty_cache()
        return

    def _run(self) -> None:
        # Init triples factory
        print("Load triples")
        dl = DataLoader(self.triples_dir, self.use_features)
        if self.emb_model.__name__ == "DistMultLiteralGated":
            tf = dl.get_numeric_triples_factory()
        else:
            tf = dl.get_triples_factory()
        # Loop over randomly sampled params
        param_sampler = ParameterSampler(
            self.param_distributions,
            self.n_param_samples,
            random_state=self.random_state,
        )
        for sampled_params in param_sampler:
            init_params, config = self.get_training_config(sampled_params)
            # Init loss
            loss = self.loss(**init_params["loss"])
            # Init negative sampler
            print("Init negative sampler")
            negative_sampler = self.negative_sampler(
                triples_factory=tf, **init_params["negative_sampler"]
            )
            if self.negative_sampler == PseudoTypedNegativeSampler:
                negative_sampler.data = negative_sampler.data.to(self.device)
            # Init numerical embedding model
            if self.num_emb_model != None:
                num_emb_model = self.num_emb_model(
                    triples_factory=tf,
                    **init_params["num_emb_model"],
                )
            else:
                num_emb_model = None
            # Init embedding model
            print("Init embedding model")
            if self.emb_model.__name__ == "DistMultLiteralGated":
                emb_model = self.emb_model(
                    triples_factory=tf,
                    loss=loss,
                    **init_params["emb_model"],
                )
            else:
                emb_model = self.emb_model(
                    num_emb_model,
                    triples_factory=tf,
                    loss=loss,
                    **init_params["emb_model"],
                )
            # Init optimizer
            optimizer = self.optimizer(
                params=emb_model.get_grad_params(), **init_params["optimizer"]
            )
            # Init training loop
            self.training_loop = SLCWATrainingLoop(
                triples_factory=tf,
                model=emb_model,
                negative_sampler=negative_sampler,
                optimizer=optimizer,
                automatic_memory_optimization=(self.sub_batch_size == None),
            )
            # Start training
            start_time = time()
            self.training_loop.train(
                tf,
                num_epochs=max(self.num_epoch_list),
                batch_size=sampled_params["batch_size"],
                sub_batch_size=self.sub_batch_size,
                num_workers=3,
                checkpoint_directory=self.checkpoint_dir,
                checkpoint_name=config["id"],
                checkpoint_frequency=2 ** 100,
                checkpoint_epochs=self.num_epoch_list,
            )
            end_time = time()
            config.update(
                start_time=str(start_time),
                end_time=str(end_time),
                duration=(end_time - start_time),
                epochs=self.num_epoch_list,
                checkpoint_dir=self.checkpoint_dir,
            )
            self.config_list.append(config)
        return

    def get_training_config(self, sampled_params: dict) -> Tuple[dict]:
        # Make a "init_params" dict with parameters used to instanciate embedding
        # models, the loss, optimizer and sampler
        init_params = {}
        fixed_params = {
            "emb_model": self.emb_params,
            "num_emb_model": self.num_emb_params,
            "negative_sampler": self.sampler_params,
            "loss": self.loss_params,
            "optimizer": self.optimizer_params,
        }
        for obj, D in fixed_params.items():
            D2 = {p: sampled_params[p] for p in self.param_mapping[obj]}
            init_params[obj] = {**D, **D2}
        init_params["emb_model"].update(
            {"random_seed": self.random_state, "preferred_device": self.device}
        )
        init_params["num_emb_model"].update(
            {"random_state": self.random_state, "device": self.device}
        )
        init_params["batch_size"] = sampled_params["batch_size"]
        # Make a "config" dict with all the parameters used in the experiment,
        # to be stored with the path to the checkpoint
        config = copy.deepcopy(init_params)
        for obj in fixed_params.keys():
            # Add model, loss, optimizer, sampler names
            try:
                config[obj]["name"] = getattr(self, obj).__name__
            except AttributeError:
                config[obj]["name"] = "None"
        for attr in ("use_features", "random_state"):
            # Add other global parameters
            config[attr] = getattr(self, attr)
        # Store triples_dir as a string to be hashed
        config["triples_dir"] = Path(self.triples_dir).as_posix()
        # Remove device parameter, since it doesn't change the training
        del config["emb_model"]["preferred_device"]
        del config["num_emb_model"]["device"]
        # Compute a unique ID for the config
        config["id"] = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("ascii")
        ).hexdigest()

        return init_params, config
