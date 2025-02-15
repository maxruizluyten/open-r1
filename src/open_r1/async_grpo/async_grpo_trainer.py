
from transformers import Trainer
from trl import SFTTrainer
import trl
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from torch import Value, nn
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
    TrainerState,
)
import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from typing import Callable, Optional, Union
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)

from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


from accelerate import Accelerator
if is_wandb_available():
    import wandb

def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q

# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class AsyncGRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    
    remote_gen_model_url: str = field(
        default=None,
    )
    remote_ref_model_url: str = field(
        default=None,
    )
    
class RemoteModel():
    def __init__(self, remote_model_url, stop_token_id=None):
        self.remote_model_url = remote_model_url
        self.stop_token_id
        
    def generate(self, input_ids: list[list[int]], max_new_tokens=256, temperature=0.8):
            # Prepare the request body
            request_body = {
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "stop_token_ids": [self.stop_token_id],
                    },
                    "stream": False,
                    "return_logprob": True,
                    "logprob_start_len": 0,
            }

            # Send the POST request to the server
            # add a few retries?
            response = requests.post(f"http://{HOST_URL}:30010/generate", json=request_body)
    
    def load_weights_from_path(path:str):
        pass

class AsyncGRPOTrainer(Trainer):
    _tag_names = ["trl", "async_grpo"]
    def __init__(self, 
                model: Union[str, PreTrainedModel],
                reward_funcs: Union[RewardFunc, list[RewardFunc]],
                args: AsyncGRPOConfig, 
                train_dataset: Dataset,
                processing_class: Optional[PreTrainedTokenizerBase] = None,
                data_collator: Optional[DataCollatorWithPadding] = None, 
                callbacks: Optional[list[TrainerCallback]] = None,
                optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
                
                ) -> None:
        
        self.args = args
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
            
        self.model = model

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        self.processing_class = processing_class
        self.train_dataset = train_dataset
        
        if data_collator is not None:
            raise ValueError("")
        
        def data_collator(features):  # No data collation is needed in GRPO
            return features       
        self.data_collator = data_collator
        
        local_dataloader_batch_size = exact_div(
            args.per_device_train_batch_size * args.gradient_accumulation_steps, 
            args.num_generations, "per_device_train_batch_size * gradient_accumulation_steps must >= num_generations to remain on policy"
        )
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        
        
        self.train_dataset_len = len(self.train_dataset)
        num_total_samples = int(self.args.num_train_epochs * self.train_dataset_len)
        self.total_steps_per_device = num_total_samples // (local_dataloader_batch_size * self.accelerator.num_processes)
        self.create_optimizer_and_scheduler(num_training_steps=self.total_steps_per_device)     
        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)

    def train(self,         
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ):
        start_step = 1 # todo, set this when we resume + load model, opt state etc
        
        if self.args.logging_steps is not None:
            if self.args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * self.args.logging_steps)
            else:
                self.state.logging_steps = self.args.logging_steps
                
        if self.args.save_steps is not None:
            if self.args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * self.args.save_steps)
            else:
                self.state.save_steps = self.args.save_steps
                
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        
        def repeat_generator():
            while True:
                yield from self.dataloader
        iter_dataloader = iter(repeat_generator())
        
        self.model.train()
        for step in range(start_step, self.total_steps_per_device + 1):
            batch = next(iter_dataloader)
            print(batch)
            
            # generation
            
            
            
            # optimization
            for mini_batch in batch:
                pass
            
            
            # maybe weight sync
            
            
            
            
            # logging stats
            metrics = {}
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            
            self.state.epoch = step / self.total_steps_per_device
            self.log(metrics)
            
            
            self.lr_scheduler.step()
            self.state.global_step += 1
            
            
            
            
            self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(self.model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(self.model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                
if __name__ == "__main__":
    url = ""
    remote_model = RemoteModel(url, stop_token_id)
    