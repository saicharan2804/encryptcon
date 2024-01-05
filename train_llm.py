"""
Fine-tuning the Language Model (LLM) specifically designed for condensing Project Design Document (PDD) .pdf files 
and predicting carbon credit issuance trends.

"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.


import logging
import math
import os
import shutil
import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import shift_tokens_right
import transformers
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdd_data import PDFDocumentDataset
from transformers import (
    AutoConfig,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.37.0.dev0")

logger = get_logger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

class Trainer:
    """
        Initializes the Trainer class for a Language Model (LLM) finetuning.

        This LLM aims to simplify the evaluation of PDDs by condensing extensive documents into 
        accessible tabular formats and providing predictions on annual carbon credit allotments. 
        The model facilitates easy interaction and dynamic updating with real-time project information, 
        enhancing the efficiency and accuracy of project evaluation.

        Args:
            args (DictConfig): An argparse.DictConfig object containing all training configurations. Expected attributes:
                - model_name_or_path (str): Path or identifier for the pretrained model.
                - with_tracking (bool): Flag to enable tracking of the training process.
                - report_to (str): Destination to report training logs.
                - output_dir (str): Directory for saving output files and checkpoints.
                - gradient_accumulation_steps (int): Number of gradient accumulation steps.
                - seed (Optional[int]): Seed for random number generators for reproducibility.
                - weight_decay (float): Weight decay parameter for the optimizer.
                - learning_rate (float): Learning rate for the optimizer.
                - max_train_steps (Optional[int]): Maximum number of training steps.
                - num_train_epochs (int): Number of training epochs.
                - per_device_train_batch_size (int): Batch size per training device.
                - checkpointing_steps (Union[int, str]): Interval for saving checkpoints.
                - resume_from_checkpoint (Optional[str]): Path to resume training from a checkpoint.
                - low_cpu_mem_usage (bool): Flag to optimize model for low CPU memory usage.
                - trust_remote_code (bool): Whether to trust and execute remote code in custom models.
                - use_slow_tokenizer (bool): Flag to use a slower but more customizable tokenizer.
                - lr_scheduler_type (str): Type of learning rate scheduler.
                - num_warmup_steps (int): Number of warm-up steps for the scheduler.
                - max_grad_norm (float): Maximum norm for gradient clipping.
                - checkpoints_total_limit (Optional[int]): Maximum number of checkpoints to retain.
    """
    def __init__(self, args):
        """Constructor"""
        # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
        # information sent is the one passed as arguments along with your Python/PyTorch versions.
        send_example_telemetry("run_clm_no_trainer", args)

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
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        # Load pretrained model and tokenizer/feature extractor
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )

        self.tokenizer = DonutProcessor.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )

        model = VisionEncoderDecoderModel.from_pretrained(
            args.model_name_or_path,
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )

        processor = DonutProcessor.from_pretrained(args.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
        model.to(self.accelerator.device)

        train_dataset = PDFDocumentDataset(**args.data_kwargs, processor = processor, model = model)

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=args.per_device_train_batch_size,
        )

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

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        # Prepare everything with our `accelerator`.
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        if self.accelerator.distributed_type == DistributedType.TPU:
            model.tie_weights()

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
        Executes the training process for the causal language modeling task.

        This method handles:
        - Loading model and optimizer states from a checkpoint if specified.
        - Iterating over the training data for the specified number of epochs.
        - Calculating loss and performing backpropagation.
        - Updating model parameters and learning rate scheduler.
        - Performing checkpointing based on specified intervals.
        - Logging training progress and metrics.
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
                    # Predict the logits and compute loss
                    embeddings, labels, mask = batch['embeddings'], batch['labels'], batch['attn_mask']

                    decoder_input_ids = shift_tokens_right(
                        labels,
                        self.config.pad_token_id,
                        self.config.decoder_start_token_id,
                    )

                    # Decode
                    outputs = self.model.decoder(
                        input_ids=decoder_input_ids,
                        attention_mask=mask,
                        encoder_hidden_states=embeddings,
                    )

                    logits = outputs.logits
                    loss = self.criterion(
                        logits.reshape(-1, self.decoder.config.vocab_size),
                        labels.reshape(-1),
                    )

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

            self.model.eval()

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
