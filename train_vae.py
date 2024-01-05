"""Variational Autoencoder (VAE) aimed at risk evaluation in the context of carbon credit trend analysis."""

import torch
from torch import optim
from accelerate import Accelerator
from vae import VAE
from omegaconf import DictConfig
import shutil
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import logging
import os
import math
from torch.utils.data import DataLoader
from series_dataset import TimeSeriesDataset
from tqdm.auto import tqdm

logger = get_logger(__name__)


class VaeTrainer:
    """
    Initializes the VaeTrainer class, responsible for training a Variational Autoencoder (VAE).

    The VAE is trained to assess the risk associated with carbon credit trends based on historical data.
    This assessment aids in understanding the potential risks involved in carbon credit allocation for
    various projects, contributing to more informed decision-making.

    Args:
        args (DictConfig): An argparse.Namespace object containing training configurations.
            It includes:
            - with_tracking (bool): Flag to enable tracking.
            - report_to (str): The reporting destination for logging.
            - output_dir (str): Directory to save output files.
            - gradient_accumulation_steps (int): Number of steps for gradient accumulation.
            - seed (Optional[int]): Random seed for reproducibility.
            - weight_decay (float): Weight decay parameter for optimizer.
            - learning_rate (float): Learning rate for training.
            - max_train_steps (Optional[int]): Maximum number of training steps.
            - num_train_epochs (int): Number of training epochs.
            - model_kwargs (dict): Keyword arguments for the VAE model.
            - per_device_train_batch_size (int): Batch size per device.
            - checkpointing_steps (Union[int, str]): Interval for checkpointing.
            - resume_from_checkpoint (Optional[str]): Path to resume training from a checkpoint.
            - max_grad_norm (float): Maximum gradient norm for clipping.
            - checkpoints_total_limit (Optional[int]): Total limit for saved checkpoints.
    """

    def __init__(self, args: DictConfig):
        """Constructor"""

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
        # in the environment
        accelerator_log_kwargs = {}

        if args.with_tracking:
            accelerator_log_kwargs["log_with"] = args.report_to
            accelerator_log_kwargs["project_dir"] = args.output_dir

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            **accelerator_log_kwargs,
        )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        # Seed setting for reproducibility.
        if args.seed is not None:
            set_seed(args.seed)

        # Create output directory for saving models and logs
        if self.accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        model = VAE(**args.model_kwargs)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate
        )

        train_dataset = TimeSeriesDataset(**args.dataset_kwargs)

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.per_device_train_batch_size,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Prepare everything with our `accelerator`.
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if args.with_tracking:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config[
                "lr_scheduler_type"
            ].value
            self.accelerator.init_trackers("clm_no_trainer", experiment_config)

        # Train!
        total_batch_size = (
            args.per_device_train_batch_size
            * self.accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        self.progress_bar = tqdm(
            range(args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        self.completed_steps = 0
        self.starting_epoch = 0

        self.args = args

    def train(self):
        """
        Executes the training process for the VAE model.

        This method handles training over epochs, implements gradient accumulation,
        performs checkpointing, and manages training resumption from saved states.

        The training process includes:
        - Loading the model and optimizer states if resuming from a checkpoint.
        - Iterating over the training data for the specified number of epochs.
        - Computing loss and backpropagation.
        - Gradient clipping and optimizer step.
        - Logging and checkpointing based on specified intervals.
        - Saving the final trained model.

        Note:
        This method relies on the arguments passed during the class initialization.
        """
        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if (
                self.args.resume_from_checkpoint is not None
                or self.args.resume_from_checkpoint != ""
            ):
                checkpoint_path = self.args.resume_from_checkpoint
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[
                    -1
                ]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * self.num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = (
                    int(training_difference.replace("step_", ""))
                    * self.args.gradient_accumulation_steps
                )
                starting_epoch = resume_step // len(self.train_dataloader)
                completed_steps = resume_step // self.args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(self.train_dataloader)

        # update the progress_bar if load from checkpoint
        self.progress_bar.update(completed_steps)

        for epoch in range(starting_epoch, self.args.num_train_epochs):
            self.model.train()
            if self.args.with_tracking:
                total_loss = 0
            if (
                self.args.resume_from_checkpoint
                and epoch == starting_epoch
                and resume_step is not None
            ):
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = self.accelerator.skip_first_batches(
                    self.train_dataloader, resume_step
                )
            else:
                active_dataloader = self.train_dataloader
            train_loss = 0.0
            for step, batch in enumerate(active_dataloader):
                with self.accelerator.accumulate(self.model):
                    credit = batch

                    pred, mu, log_var = self.model(credit)
                    loss_dict = self.model.loss_function(
                        pred, credit, mu, log_var, kld_weight=self.args["kld_weight"]
                    )

                    loss = loss_dict["loss"]

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(
                        loss.repeat(self.args.train_batch_size)
                    ).mean()
                    train_loss += (
                        avg_loss.item() / self.args.gradient_accumulation_steps
                    )

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    self.progress_bar.update(1)
                    completed_steps += 1
                    self.accelerator.log(
                        {"train_loss": train_loss}, step=completed_steps
                    )
                    total_loss += train_loss
                    train_loss = 0.0

                    if isinstance(self.checkpointing_steps, int):
                        if completed_steps % self.checkpointing_steps == 0:
                            if self.accelerator.is_main_process:
                                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                                if self.args.checkpoints_total_limit is not None:
                                    checkpoints = os.listdir(self.args.output_dir)
                                    checkpoints = [
                                        d
                                        for d in checkpoints
                                        if d.startswith("checkpoint")
                                    ]
                                    checkpoints = sorted(
                                        checkpoints, key=lambda x: int(x.split("-")[1])
                                    )

                                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                    if (
                                        len(checkpoints)
                                        >= self.args.checkpoints_total_limit
                                    ):
                                        num_to_remove = (
                                            len(checkpoints)
                                            - self.args.checkpoints_total_limit
                                            + 1
                                        )
                                        removing_checkpoints = checkpoints[
                                            0:num_to_remove
                                        ]

                                        logger.info(
                                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                        )
                                        logger.info(
                                            f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                        )

                                        for removing_checkpoint in removing_checkpoints:
                                            removing_checkpoint = os.path.join(
                                                self.args.output_dir,
                                                removing_checkpoint,
                                            )
                                            shutil.rmtree(removing_checkpoint)

                                save_path = os.path.join(
                                    self.args.output_dir,
                                    f"checkpoint-{completed_steps}",
                                )
                                self.accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                self.progress_bar.set_postfix(**logs)

                if completed_steps >= self.args.max_train_steps:
                    break

            if self.args.with_tracking:
                self.accelerator.log(
                    {
                        "train_loss": total_loss.item() / len(self.train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

            if self.args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if self.args.output_dir is not None:
                    output_dir = os.path.join(self.args.output_dir, output_dir)
                self.accelerator.save_state(output_dir)

        if self.args.with_tracking:
            self.accelerator.end_training()

        if self.args.output_dir is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                self.args.output_dir,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
            )
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(self.args.output_dir)
