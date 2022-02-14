# -*- coding: utf-8 -*-

"""A modification of PyKEEN SLCWATrainingLoop class to save checkpoints at several
epochs and append their names to them."""

from class_resolver import HintOrType
import datetime
import gc
import logging
import numpy as np
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from tqdm.autonotebook import tqdm, trange
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional, Mapping, Union, IO, Any, List

from pykeen.constants import PYKEEN_CHECKPOINTS, PYKEEN_DEFAULT_CHECKPOINT
from pykeen.lr_schedulers import LRScheduler
from pykeen.models import Model, RGCN
from pykeen.sampling import NegativeSampler
from pykeen.stoppers import Stopper
from pykeen.training import SLCWATrainingLoop as BaseClass
from pykeen.training.callbacks import (
    MultiTrainingCallback,
    TrackerCallback,
    TrainingCallbackHint,
)
from pykeen.training.training_loop import (
    SubBatchingNotSupportedError,
    _get_optimizer_kwargs,
    _get_lr_scheduler_kwargs,
)
from pykeen.training.schlichtkrull_sampler import GraphSampler
from pykeen.trackers import ResultTracker
from pykeen.triples import CoreTriplesFactory, Instances
from pykeen.utils import format_relative_comparison, get_batchnorm_modules

logger = logging.getLogger(__name__)


class SLCWATrainingLoop(BaseClass):
    def __init__(
        self,
        model: Model,
        triples_factory: CoreTriplesFactory,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: Optional[Mapping[str, Any]] = None,
        automatic_memory_optimization: bool = True,
    ) -> None:

        super().__init__(
            model,
            triples_factory,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            negative_sampler=negative_sampler,
            negative_sampler_kwargs=negative_sampler_kwargs,
            automatic_memory_optimization=automatic_memory_optimization,
        )
        return

    def train(
        self,
        triples_factory: CoreTriplesFactory,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        use_tqdm: bool = True,
        use_tqdm_batch: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        result_tracker: Optional[ResultTracker] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        clear_optimizer: bool = False,
        checkpoint_directory: Union[None, str, Path] = None,
        checkpoint_name: Optional[str] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_epochs: List[int] = [],
        checkpoint_on_failure: bool = False,
        drop_last: Optional[bool] = None,
        callbacks: TrainingCallbackHint = None,
    ) -> Optional[List[float]]:

        # Create training instances. Use the _create_instances function to allow subclasses
        # to modify this behavior
        training_instances = self._create_instances(triples_factory)

        # In some cases, e.g. using Optuna for HPO, the cuda cache from a previous run is not cleared
        torch.cuda.empty_cache()

        # A checkpoint root is always created to ensure a fallback checkpoint can be saved
        if checkpoint_directory is None:
            checkpoint_directory = PYKEEN_CHECKPOINTS
        checkpoint_directory = Path(checkpoint_directory)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        logger.debug("using checkpoint_root at %s", checkpoint_directory)

        # If a checkpoint file is given, it must be loaded if it exists already
        save_checkpoints = False
        checkpoint_path = None
        best_epoch_model_file_path = None
        last_best_epoch = None
        if checkpoint_name:
            # checkpoint_path = checkpoint_directory.joinpath(checkpoint_name)
            # NEW: automatically search for checkpoints with the same name
            checkpoint_path = self.get_previous_checkpoint_path(
                checkpoint_directory, checkpoint_name, num_epochs
            )
            if checkpoint_path.is_file():
                best_epoch_model_file_path, last_best_epoch = self._load_state(
                    path=checkpoint_path,
                    triples_factory=triples_factory,
                )
                if stopper is not None:
                    stopper_dict = (
                        stopper.load_summary_dict_from_training_loop_checkpoint(
                            path=checkpoint_path
                        )
                    )
                    # If the stopper dict has any keys, those are written back to the stopper
                    if stopper_dict:
                        stopper._write_from_summary_dict(**stopper_dict)
                    else:
                        logger.warning(
                            "the training loop was configured with a stopper but no stopper configuration was "
                            "saved in the checkpoint",
                        )
                continue_training = True
            else:
                logger.info(
                    f"=> no checkpoint found at '{checkpoint_path}'. Creating a new file."
                )
            # The checkpoint frequency needs to be set to save checkpoints
            if checkpoint_frequency is None:
                checkpoint_frequency = 30
            save_checkpoints = True
        elif checkpoint_frequency is not None:
            logger.warning(
                "A checkpoint frequency was set, but no checkpoint file was given. No checkpoints will be created",
            )

        checkpoint_on_failure_file_path = None
        if checkpoint_on_failure:
            # In case a checkpoint frequency was set, we warn that no checkpoints will be saved
            date_string = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            # If no checkpoints were requested, a fallback checkpoint is set in case the training loop crashes
            checkpoint_on_failure_file_path = checkpoint_directory.joinpath(
                PYKEEN_DEFAULT_CHECKPOINT.replace(".", f"_{date_string}."),
            )

        # If the stopper loaded from the training loop checkpoint stopped the training, we return those results
        if getattr(stopper, "stopped", False):
            result: Optional[List[float]] = self.losses_per_epochs
        else:
            result = self._train(
                num_epochs=num_epochs,
                batch_size=batch_size,
                slice_size=slice_size,
                label_smoothing=label_smoothing,
                sampler=sampler,
                continue_training=continue_training,
                only_size_probing=only_size_probing,
                use_tqdm=use_tqdm,
                use_tqdm_batch=use_tqdm_batch,
                tqdm_kwargs=tqdm_kwargs,
                stopper=stopper,
                result_tracker=result_tracker,
                sub_batch_size=sub_batch_size,
                num_workers=num_workers,
                save_checkpoints=save_checkpoints,
                checkpoint_path=checkpoint_path,
                checkpoint_frequency=checkpoint_frequency,
                checkpoint_epochs=checkpoint_epochs,  # NEW
                checkpoint_on_failure_file_path=checkpoint_on_failure_file_path,
                best_epoch_model_file_path=best_epoch_model_file_path,
                last_best_epoch=last_best_epoch,
                drop_last=drop_last,
                callbacks=callbacks,
                triples_factory=triples_factory,
                training_instances=training_instances,
            )

        # Ensure the release of memory
        torch.cuda.empty_cache()

        # Clear optimizer
        if clear_optimizer:
            self.optimizer = None
            self.lr_scheduler = None

        return result

    #@profile
    def _train(  # noqa: C901
        self,
        triples_factory: CoreTriplesFactory,
        training_instances: Instances,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        use_tqdm: bool = True,
        use_tqdm_batch: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        result_tracker: Optional[ResultTracker] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        save_checkpoints: bool = False,
        checkpoint_path: Union[None, str, Path] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_epochs: List[int] = [],
        checkpoint_on_failure_file_path: Union[None, str, Path] = None,
        best_epoch_model_file_path: Optional[Path] = None,
        last_best_epoch: Optional[int] = None,
        drop_last: Optional[bool] = None,
        callbacks: TrainingCallbackHint = None,
    ) -> Optional[List[float]]:

        if self.optimizer is None:
            raise ValueError("optimizer must be set before running _train()")
        # When using early stopping models have to be saved separately at the best epoch, since the training loop will
        # due to the patience continue to train after the best epoch and thus alter the model
        if (
            stopper is not None
            and not only_size_probing
            and last_best_epoch is None
            and best_epoch_model_file_path is None
        ):
            # Create a path
            best_epoch_model_file_path = Path(NamedTemporaryFile().name)
        best_epoch_model_checkpoint_file_path: Optional[Path] = None

        if isinstance(self.model, RGCN) and sampler != "schlichtkrull":
            logger.warning(
                'Using RGCN without graph-based sampling! Please select sampler="schlichtkrull" instead of %s.',
                sampler,
            )

        # Prepare all of the callbacks
        callback = MultiTrainingCallback(callbacks)
        # Register a callback for the result tracker, if given
        if result_tracker is not None:
            callback.register_callback(TrackerCallback(result_tracker))

        callback.register_training_loop(self)

        # Take the biggest possible training batch_size, if batch_size not set
        batch_size_sufficient = False
        if batch_size is None:
            if self.automatic_memory_optimization:
                # Using automatic memory optimization on CPU may result in undocumented crashes due to OS' OOM killer.
                if self.model.device.type == "cpu":
                    batch_size = 256
                    batch_size_sufficient = True
                    logger.info(
                        "Currently automatic memory optimization only supports GPUs, but you're using a CPU. "
                        "Therefore, the batch_size will be set to the default value '{batch_size}'",
                    )
                else:
                    batch_size, batch_size_sufficient = self.batch_size_search(
                        triples_factory=triples_factory,
                        training_instances=training_instances,
                    )
            else:
                batch_size = 256
                logger.info(
                    f"No batch_size provided. Setting batch_size to '{batch_size}'."
                )

        # This will find necessary parameters to optimize the use of the hardware at hand
        if (
            not only_size_probing
            and self.automatic_memory_optimization
            and not batch_size_sufficient
            and not continue_training
        ):
            # return the relevant parameters slice_size and batch_size
            sub_batch_size, slice_size = self.sub_batch_and_slice(
                batch_size=batch_size,
                sampler=sampler,
                triples_factory=triples_factory,
                training_instances=training_instances,
            )

        if (
            sub_batch_size is None or sub_batch_size == batch_size
        ):  # by default do not split batches in sub-batches
            sub_batch_size = batch_size
        elif get_batchnorm_modules(self.model):  # if there are any, this is truthy
            raise SubBatchingNotSupportedError(self.model)

        model_contains_batch_norm = bool(get_batchnorm_modules(self.model))
        if batch_size == 1 and model_contains_batch_norm:
            raise ValueError(
                "Cannot train a model with batch_size=1 containing BatchNorm layers."
            )
        if drop_last is None:
            drop_last = model_contains_batch_norm
            if drop_last and not only_size_probing:
                logger.info(
                    "Dropping last (incomplete) batch each epoch (%s batches).",
                    format_relative_comparison(part=1, total=len(training_instances)),
                )

        # Force weight initialization if training continuation is not explicitly requested.
        if not continue_training:
            # Reset the weights
            self.model.reset_parameters_()

            # Create new optimizer
            optimizer_kwargs = _get_optimizer_kwargs(self.optimizer)
            self.optimizer = self.optimizer.__class__(
                params=self.model.get_grad_params(),
                **optimizer_kwargs,
            )

            if self.lr_scheduler is not None:
                # Create a new lr scheduler and add the optimizer
                lr_scheduler_kwargs = _get_lr_scheduler_kwargs(self.lr_scheduler)
                self.lr_scheduler = self.lr_scheduler.__class__(
                    self.optimizer, **lr_scheduler_kwargs
                )
        elif not self.optimizer.state:
            raise ValueError("Cannot continue_training without being trained once.")

        # Ensure the model is on the correct device
        self.model = self.model.to(self.device)

        # Create Sampler
        if sampler == "schlichtkrull":
            if triples_factory is None:
                raise ValueError(
                    "need to pass triples_factory when using graph sampling"
                )
            sampler = GraphSampler(triples_factory, num_samples=sub_batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        if num_workers is None:
            num_workers = 0

        # Bind
        num_training_instances = len(training_instances)

        _use_outer_tqdm = not only_size_probing and use_tqdm
        _use_inner_tqdm = _use_outer_tqdm and use_tqdm_batch

        # When size probing, we don't want progress bars
        if _use_outer_tqdm:
            # Create progress bar
            _tqdm_kwargs = dict(desc=f"Training epochs on {self.device}", unit="epoch")
            if tqdm_kwargs is not None:
                _tqdm_kwargs.update(tqdm_kwargs)
            epochs = trange(
                self._epoch + 1,
                1 + num_epochs,
                **_tqdm_kwargs,
                initial=self._epoch,
                total=num_epochs,
            )
        elif only_size_probing:
            epochs = range(1, 1 + num_epochs)
        else:
            epochs = range(self._epoch + 1, 1 + num_epochs)

        logger.debug(f"using stopper: {stopper}")

        train_data_loader = DataLoader(
            sampler=sampler,
            dataset=training_instances,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )

        # Save the time to track when the saved point was available
        last_checkpoint = time.time()

        # Training Loop
        for epoch in epochs:
            # When training with an early stopper the memory pressure changes, which may allow for errors each epoch
            try:
                # Enforce training mode
                self.model.train()

                # Accumulate loss over epoch
                current_epoch_loss = 0.0

                # Batching
                # Only create a progress bar when not in size probing mode
                if _use_inner_tqdm:
                    batches = tqdm(
                        train_data_loader,
                        desc=f"Training batches on {self.device}",
                        leave=False,
                        unit="batch",
                    )
                else:
                    batches = train_data_loader

                # Flag to check when to quit the size probing
                evaluated_once = False

                for batch in batches:
                    # Recall that torch *accumulates* gradients. Before passing in a
                    # new instance, you need to zero out the gradients from the old instance
                    self.optimizer.zero_grad()

                    # Get batch size of current batch (last batch may be incomplete)
                    current_batch_size = self._get_batch_size(batch)

                    # accumulate gradients for whole batch
                    for start in range(0, current_batch_size, sub_batch_size):
                        stop = min(start + sub_batch_size, current_batch_size)

                        # forward pass call
                        batch_loss = self._forward_pass(
                            batch,
                            start,
                            stop,
                            current_batch_size,
                            label_smoothing,
                            slice_size,
                        )
                        current_epoch_loss += batch_loss
                        callback.on_batch(
                            epoch=epoch, batch=batch, batch_loss=batch_loss
                        )

                    # when called by batch_size_search(), the parameter update should not be applied.
                    if not only_size_probing:
                        # update parameters according to optimizer
                        self.optimizer.step()

                    # After changing applying the gradients to the embeddings, the model is notified that the forward
                    # constraints are no longer applied
                    self.model.post_parameter_update()

                    # For testing purposes we're only interested in processing one batch
                    if only_size_probing and evaluated_once:
                        break

                    callback.post_batch(epoch=epoch, batch=batch)

                    evaluated_once = True

                del batch
                del batches
                gc.collect()
                self.optimizer.zero_grad()
                self._free_graph_and_cache()

                # When size probing we don't need the losses
                if only_size_probing:
                    return None

                # Update learning rate scheduler
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch=epoch)

                # Track epoch loss
                epoch_loss = current_epoch_loss / num_training_instances
                self.losses_per_epochs.append(epoch_loss)

                # Print loss information to console
                if _use_outer_tqdm:
                    epochs.set_postfix(
                        {
                            "loss": self.losses_per_epochs[-1],
                            "prev_loss": self.losses_per_epochs[-2]
                            if epoch > 2
                            else float("nan"),
                        }
                    )

                # Save the last successful finished epoch
                self._epoch = epoch

                should_stop = False
                if stopper is not None and stopper.should_evaluate(epoch):
                    if stopper.should_stop(epoch):
                        should_stop = True
                    # Since the model is also used within the stopper, its graph and cache have to be cleared
                    self._free_graph_and_cache()
                # When the stopper obtained a new best epoch, this model has to be saved for reconstruction
                if (
                    stopper is not None
                    and stopper.best_epoch != last_best_epoch
                    and best_epoch_model_file_path is not None
                ):
                    self._save_state(
                        path=best_epoch_model_file_path, triples_factory=triples_factory
                    )
                    last_best_epoch = epoch
            # When the training loop failed, a fallback checkpoint is created to resume training.
            except (MemoryError, RuntimeError) as e:
                # During automatic memory optimization only the error message is of interest
                if only_size_probing:
                    raise e

                logger.warning(
                    f"The training loop just failed during epoch {epoch} due to error {str(e)}."
                )
                if checkpoint_on_failure_file_path:
                    # When there wasn't a best epoch the checkpoint path should be None
                    if (
                        last_best_epoch is not None
                        and best_epoch_model_file_path is not None
                    ):
                        best_epoch_model_checkpoint_file_path = (
                            best_epoch_model_file_path
                        )
                    self._save_state(
                        path=checkpoint_on_failure_file_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )
                    logger.warning(
                        "However, don't worry we got you covered. PyKEEN just saved a checkpoint when this "
                        f"happened at '{checkpoint_on_failure_file_path}'. To resume training from the checkpoint "
                        f"file just restart your code and pass this file path to the training loop or pipeline you "
                        f"used as 'checkpoint_file' argument.",
                    )
                # Delete temporary best epoch model
                if (
                    best_epoch_model_file_path is not None
                    and best_epoch_model_file_path.is_file()
                ):
                    os.remove(best_epoch_model_file_path)
                raise e

            # Includes a call to result_tracker.log_metrics
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss)

            # If a checkpoint file is given, we check whether it is time to save a checkpoint
            if save_checkpoints and checkpoint_path is not None:
                minutes_since_last_checkpoint = (time.time() - last_checkpoint) // 60
                # MyPy overrides are because you should
                if (
                    minutes_since_last_checkpoint >= checkpoint_frequency  # type: ignore
                    or should_stop
                    or epoch == num_epochs
                    or epoch in checkpoint_epochs  # NEW: save ckpt at several epochs
                ):
                    # When there wasn't a best epoch the checkpoint path should be None
                    if (
                        last_best_epoch is not None
                        and best_epoch_model_file_path is not None
                    ):
                        best_epoch_model_checkpoint_file_path = (
                            best_epoch_model_file_path
                        )
                    # NEW: Append the epoch to the ckpt name
                    ckpt_dir = Path(checkpoint_path).parent
                    ckpt_name = Path(checkpoint_path).name.split("_")[0] + f"_{epoch}"
                    self._save_state(
                        path=Path(ckpt_dir) / ckpt_name,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )  # type: ignore
                    last_checkpoint = time.time()

            if (
                should_stop
                and last_best_epoch is not None
                and best_epoch_model_file_path is not None
            ):
                self._load_state(path=best_epoch_model_file_path)
                # Delete temporary best epoch model
                if Path.is_file(best_epoch_model_file_path):
                    os.remove(best_epoch_model_file_path)
                return self.losses_per_epochs

        callback.post_train(losses=self.losses_per_epochs)

        # If the stopper didn't stop the training loop but derived a best epoch, the model has to be reconstructed
        # at that state
        if (
            stopper is not None
            and last_best_epoch is not None
            and best_epoch_model_file_path is not None
        ):
            self._load_state(path=best_epoch_model_file_path)
            # Delete temporary best epoch model
            if Path.is_file(best_epoch_model_file_path):
                os.remove(best_epoch_model_file_path)

        return self.losses_per_epochs

    # # USED TO TRANSFER BATCHES ACROSS DEVICES IN FORWARD
    # def _process_batch(
    #     self,
    #     batch,
    #     start: int,
    #     stop: int,
    #     label_smoothing: float = 0.0,
    #     slice_size: Optional[int] = None,
    # ) -> torch.FloatTensor:  # noqa: D102
    #     # Slicing is not possible in sLCWA training loops
    #     if slice_size is not None:
    #         raise AttributeError("Slicing is not possible for sLCWA training loops.")

    #     # Send positive batch to device
    #     positive_batch = batch[start:stop].to(device=self.device)

    #     # Create negative samples, shape: (batch_size, num_neg_per_pos, 3)
    #     negative_batch, positive_filter = self.negative_sampler.sample(
    #         positive_batch=positive_batch
    #     )

    #     # apply filter mask
    #     if positive_filter is None:
    #         negative_batch = negative_batch.view(-1, 3)
    #     else:
    #         negative_batch = negative_batch[positive_filter]

    #     # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
    #     # BasicNegativeSampler, BernoulliNegativeSampler
    #     negative_batch = negative_batch.to(self.device)

    #     # Compute negative and positive scores
    #     positive_scores = self.model.score_hrt(positive_batch)
    #     negative_scores = self.model.score_hrt(negative_batch).view(
    #         *negative_batch.shape[:-1]
    #     )

    #     return (
    #         self.loss.process_slcwa_scores(
    #             positive_scores=positive_scores,
    #             negative_scores=negative_scores,
    #             label_smoothing=label_smoothing,
    #             batch_filter=positive_filter,
    #             num_entities=self.model.num_entities,
    #         )
    #         + self.model.collect_regularization_term()
    #     )

    def get_previous_checkpoint_path(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        num_epochs: int,
    ) -> Path:
        """Return the number of epochs of the previous checkpoint."""
        files = list(Path(checkpoint_dir).glob(f"{checkpoint_name}_*"))
        epochs = np.array([int(file.name.split("_")[1]) for file in files])
        if len(epochs) == 0:  # empty
            last_epoch = 0
        else:
            last_epoch = epochs[epochs <= num_epochs].max()

        return Path(checkpoint_dir) / (str(checkpoint_name) + f"_{last_epoch}")
