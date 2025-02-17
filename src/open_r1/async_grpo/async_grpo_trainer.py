
from multiprocessing import reduction
from transformers import Trainer
from open_r1.async_grpo.remote_model import RemoteModel
import trl
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
    TrainerState,
)
import math
import os
from typing import Callable, Optional, Union
import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from trl.trainer.utils import pad, selective_log_softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from trl.data_utils import maybe_apply_chat_template, is_conversational
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
        default="0.0.0.0",
    )
    remote_ref_model_url: str = field(
        default="0.0.0.0",
    )
    remote_gen_model_port: str = field(
        default="30010",
    )
    remote_ref_model_port: str = field(
        default="30010",
    )

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
        self.reward_funcs = reward_funcs
        # Reward weights (move this logic to post_init of config?)
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = args.reward_weights
        else:
            self.reward_weights = [1.0] * len(reward_funcs),
        
        
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
        
        
        self.remote_gen_model = RemoteModel(args.remote_gen_model_url, args.remote_gen_model_port, self.processing_class.eos_token_id)
        self.remote_ref_model = RemoteModel(args.remote_ref_model_url, args.remote_ref_model_port, self.processing_class.eos_token_id)
        
        
        
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
            drop_last=True,
        )
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)


    def prepare_batch(self, batch):
        """
        This will:
        - generate k samples for each problem
        - compute ref logprobs for each generation
        - using internal reward model(s) to get rewards
        """
        prompts = [x["prompt"] for x in batch]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in batch]
        prompt_inputs = self.processing_class(prompts_text)
        generations = self.remote_gen_model.generate(prompt_inputs["input_ids"],
                                                     max_new_tokens=self.args.max_completion_length,
                                                     num_generations=self.args.num_generations,
                                                     )
        prompt_completion_ids = [example["prompt_ids"] + example["completion_ids"] for example in generations]
        
        # generate for 1 step to get the log_probs
        ref_generations = self.remote_ref_model.generate(prompt_completion_ids, max_new_tokens=1, num_generations=1)
        
        gen_log_probs = [example["prompt_log_probs"] + example["completion_log_probs"] for example in generations]
        # drop last as we generated 1 extra token
        ref_log_probs = [example["prompt_log_probs"] + example["completion_log_probs"][:-1] for example in ref_generations]
        
        # calculate the rewards, assume each RF maps string problem, completion -> reward for now
        # repeat the problems
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt]*self.args.num_generations)
        
        completions_text = self.processing_class.batch_decode([example["completion_ids"] for example in generations])
        if is_conversational(batch[0]):
            completions = []
            for prompt, completion in zip(repeated_prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        rewards = torch.zeros(len(completions), len(self.reward_funcs))
        reward_kwargs = {} # TODO: add this 
        for i, reward_func in enumerate(self.reward_funcs):
            keys = [key for key in batch[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [] for key in keys} # or use defaultdict
            for example in batch:
                for key in keys:
                    reward_kwargs[key].extend([example[key]]*self.args.num_generations)
            output_rewards = reward_func(prompts=repeated_prompts, completions=completions, **reward_kwargs)
            rewards[:, i] = torch.Tensor(output_rewards) * self.reward_weights[i]
            # TODO: log the rewards
        
        # calculate the advantages
        grouped_rewards = rewards.sum(-1).view(len(prompts), self.args.num_generations)
        EPS = 1e-4
        grouped_advantages = (grouped_rewards - grouped_rewards.mean(-1, keepdim=True)) /  (grouped_rewards.std(-1, keepdim=True) + EPS)
        advantages = grouped_advantages.flatten().tolist()
        
        # build batch as list of dicts
        examples = []
        for i, prompt in enumerate(repeated_prompts):
            example = {
                "prompt": prompt,
                "prompt_ids": prompt_inputs["input_ids"][i // self.args.num_generations],
                "completion": completions[i],
                "completion_ids": generations[i]["completion_ids"],
                "prompt_completion_ids": prompt_completion_ids[i],
                "gen_log_probs": gen_log_probs[i], # may be used for clipping in future version
                "ref_log_probs": ref_log_probs[i],
                "advantages": advantages[i], 
                "rewards": rewards[i]
            }
            examples.append(example)
        
        return examples
        
        

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
            
            batch = self.prepare_batch(batch)
            # TODO: log completions, rewards, etc
            gen_dataset = Dataset.from_list(batch)
            
            @torch.no_grad()
            def mini_batch_collator(examples):
                all_prompt_completion_ids = [torch.LongTensor(example["prompt_completion_ids"]) for example in examples]
                padded_prompt_completion_ids = pad(all_prompt_completion_ids)
                gen_log_probs = [torch.Tensor(example["gen_log_probs"][1:]) for example in examples]
                ref_log_probs = [torch.Tensor(example["ref_log_probs"][1:]) for example in examples]
                
                completion_mask = torch.zeros_like(padded_prompt_completion_ids, dtype=torch.float32)
                for i, example in enumerate(examples):
                    prompt_ids = example["prompt_ids"]
                    prompt_completion_ids = example["prompt_completion_ids"]
                    # mask for the loss computation
                    completion_mask[i,:len(prompt_completion_ids)] += 1.0
                    completion_mask[i,:len(prompt_ids)] *= 0.0
                    
                advantages = torch.Tensor([example["advantages"] for example in examples])
                
                device = self.model.device
                
                return {
                    "padded_prompt_completion_ids": padded_prompt_completion_ids.to(device),
                    "padded_gen_log_probs": pad(gen_log_probs).to(device),
                    "padded_ref_log_probs": pad(ref_log_probs).to(device),
                    "completion_mask": completion_mask.to(device),
                    "advantages": advantages.to(device), 
                }

            
            gen_dataloader = DataLoader(
                gen_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=mini_batch_collator
            )
            # optimization
            for mini_batch in gen_dataloader:
                with self.accelerator.accumulate(self.model):
                    # optimization step, attn mask not needed as we are right padded
                    logits = self.model(input_ids=mini_batch["padded_prompt_completion_ids"]).logits
                    logits = logits[:, :-1, :]
                     # drop first label as there is no logit predicted for it
                    labels = mini_batch["padded_prompt_completion_ids"][:, 1:]
                    log_probs = selective_log_softmax(logits, labels)
                    ref_log_probs = mini_batch["padded_ref_log_probs"]
                    gen_log_probs = mini_batch["padded_gen_log_probs"]
                    
                    assert log_probs.shape == ref_log_probs.shape
                    # TODO: we can just use the gen_log_probs here and truncate the logprobs to just the outputs
                    per_token_kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
                    advantages = mini_batch["advantages"]
                    per_token_loss = torch.exp(log_probs - gen_log_probs.detach()) * advantages.unsqueeze(-1)
                    
                    per_token_loss = -(per_token_loss - self.args.beta * per_token_kl)
                    completion_mask = mini_batch["completion_mask"]
                    loss = (per_token_loss * completion_mask[:,1:]).sum() / completion_mask.sum()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

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
                
