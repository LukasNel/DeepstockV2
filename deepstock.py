import re
from dataclasses import dataclass, field
import json
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import os
import sys
from dataclasses import dataclass, field
import logging
import datasets
import torch
import transformers
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
import re
import json
import math
from datasets import load_dataset

def get_answer(content) -> re.Match:
    pattern = r"^.*?</think>.*<answer>\s*(buy|sell)\s*</answer>$"
    match = re.match(pattern, content, re.IGNORECASE | re.DOTALL)
    if not match:
       return None
    
    return match.group(1).lower().strip()
   

def accuracy_reward(completions, company_info, **kwargs):
    """
    Reward function that checks if the completion correctly predicted diagnosis.
    Returns 1.0 if prediction matches actual diagnosis, 0.0 otherwise.
    Ignores whitespace in the answer.
    """

    try:
      rewards = []
      contents = [completion[0]["content"] for completion in completions]
      for completion_contents, company_info_obj in zip(contents, company_info):
        close_price = float(company_info_obj["price"]["close"])
        open_price = float(company_info_obj["price"]["open"])
        actual_movement = "buy" if close_price > open_price else "sell"
        match = get_answer(completion_contents.strip())
        if not match:
            rewards.append(0.0)
            continue

        # Extract prediction and remove all whitespace
        prediction = match
        reward = 1.0 if prediction == actual_movement else 0.0
        rewards.append(reward)
      return rewards
    except Exception as e:
      print(company_info, completions)
      raise e


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format and answer content."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [get_answer(content) for content in completion_contents]
    lengths = [len(content) for content in completion_contents]
    rewards = []
    for match_r, length in zip(matches, lengths):
      if match_r:
        rewards.append(1.0)
      else:
        rewards.append(0.0)
    return rewards
def test_reward_functions():
    # Sample completions in various formats
    completions = [
        [{"content": "<think>price went upprice went upprice went upprice went upprice went upprice went upprice went upprice went upprice went upprice went upprice went upprice went up</think><answer>buy</answer>"}],  # Correct format, "up"
        [{"content": "<think>price dropped</think><answer>  sell  </answer>"}],  # Correct format with whitespace
        [{"content": "<think>analysis</think><answer>sideways</answer>"}],  # Wrong answer
        [{"content": "just saying up"}],  # Wrong format
        [{"content": "<think>going up</think><answer> buy </answer>"}],  # Correct format with mixed case
    ]

    # Sample company data with price going up
    company_info_up = [{
        "price": {
            "open": 100.0,
            "close": 110.0
        }
    }]*len(completions)

    # Sample company data with price going down
    company_info_down = [{
        "price": {
            "open": 110.0,
            "close": 100.0
        }
    }]*len(completions)

    # Test format reward
    format_results = format_reward(completions)
    expected_format = [1.0, 1.0, 0.0, 0.0, 1.0]
    assert format_results == expected_format, f"Format reward test failed. Got {format_results}, expected {expected_format}"

    # Test accuracy reward with upward movement
    accuracy_results_up = accuracy_reward(completions, company_info_up)
    expected_accuracy_up = [1.0, 0.0, 0.0, 0.0, 1.0]
    assert accuracy_results_up == expected_accuracy_up, f"Accuracy reward (yes) test failed. Got {accuracy_results_up}, expected {expected_accuracy_up}"

    # Test accuracy reward with downward movement
    accuracy_results_down = accuracy_reward(completions, company_info_down)
    expected_accuracy_down = [0.0, 1.0, 0.0, 0.0, 0.0]
    assert accuracy_results_down == expected_accuracy_down, f"Accuracy reward (no) test failed. Got {accuracy_results_down}, expected {expected_accuracy_down}"

    print("All tests passed!")

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format",],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        # "cosine": get_cosine_scaled_reward(
        #     min_value_wrong=script_args.cosine_min_value_wrong,
        #     max_value_wrong=script_args.cosine_max_value_wrong,
        #     min_value_correct=script_args.cosine_min_value_correct,
        #     max_value_correct=script_args.cosine_max_value_correct,
        #     max_len=script_args.cosine_max_len,
        # ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["user_prompt"]},
            ],
        }

    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        # callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)