import logging
import os

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy

logger = logging.getLogger(__name__)


class Trainer:
    """
    A custom model trainer class for training an agent.
    """

    def __init__(self):
        pass

    def _create_output_dirs(self):
        """
        Creates the output directories needed.
        """
        # Create the model directory
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)

    def _init_logs(self):
        """
        Initializes the logs.
        """
        # Initialize the TensorBoard logging
        self._train_summary_writer = SummaryWriter(
            os.path.join(str(self.output_dir), "logs", "train")
        )
        self._val_summary_writer = SummaryWriter(
            os.path.join(str(self.output_dir), "logs", "val")
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        max_steps: int,
        grad_accum_steps: int,
        val_check_interval_steps: int,
        val_sanity_check: bool = True,
    ):
        """
        Fits the model to the training data.

        Parameters
        ----------
        train_loader : DataLoader
            The training data loader.
        val_loader : DataLoader
            The validation data loader.
        max_epochs : int
            The maximum number of epochs to train.
        max_steps : int
            The maximum number of steps to train.
        grad_accum_steps : int
            The number of gradient accumulation steps.
        val_check_interval_steps : int
            The number of steps between validation checks.
        val_sanity_check : bool, optional
            Whether to perform a validation sanity check. The default is True.

        Returns
        -------
        None.

        """
        # Store the input parameters
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.val_check_interval_steps = val_check_interval_steps

        # Initialize the tracking variables
        self._step_ind = 0
        self._epoch_ind = 0
        self._best_val_loss = float("inf")
        self._batch_ind = 0
        self._train_epoch_loss = 0
        self._train_epoch_ind_loss = [0] * len(self.loss_fn_list)
        self._accumulated_steps = 0
        self._train_batch_loss = 0
        self._train_batch_ind_loss = [0] * len(self.loss_fn_list)

        # Initialize extra metrics
        self.train_mlm_accuracy = MulticlassAccuracy(device=self.model.device)
        self.train_nsp_accuracy = MulticlassAccuracy(device=self.model.device)
        self.train_mlm_accuracy.reset()
        self.train_nsp_accuracy.reset()

        # Log the current status
        if self._fabric.global_rank == 0:
            logger.info("Starting training loop.")
            logger.info(
                f"Effective batch size: {self.grad_accum_steps * self.train_loader.batch_size * self._fabric.world_size}"
            )
        self._fabric.barrier()

        # Run a sanity check
        if val_sanity_check and self.val_loader is not None:
            self.validate()

        # Run the training loop
        self._train_loop()

        # Run the validation loop
        if self.val_loader is not None:
            self.validate()

        # Save the model
        self._save_model()
