from modal import Image, App
import modal

huggingface_secret = modal.Secret.from_name(
    "huggingface_secret", required_keys=["HF_TOKEN", "WANDB_API_KEY"]
)
# Create stub and image
app = App("deepstock-v2")
GPU_USED = "A100-80GB:3"
DATASET_ID="2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell"
MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
def download_models():
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    dataset = load_dataset(DATASET_ID)


image = (
    Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.7.2", gpu=GPU_USED)
    .apt_install("git")
    .apt_install("wget")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # turn on faster downloads from HF
    .run_commands("""ls && \
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
dpkg -i cuda-keyring_1.1-1_all.deb && \
apt-get update && \
apt-get install -y cuda-toolkit    
""")
    .run_commands(
        'git clone https://github.com/huggingface/open-r1.git && cd open-r1 && pip install -e ".[dev]"', gpu=GPU_USED
    )
    .run_commands("ls && pwd")
    .run_function(download_models, secrets=[huggingface_secret])
    .pip_install("wandb")
    .pip_install("peft")
    #install fromm https://github.com/huggingface/trl/compare/main...LukasNel:trl:patch-2
    .add_local_file("lotus_diabetes_seek.py", "/open-r1/src/open_r1/grpo_lukas.py")
)



@app.function(image=image, secrets=[huggingface_secret], gpu="A100-80GB:4", timeout=43200,
              volumes={
                  "/data": modal.Volume.from_name("deepstock-data")
              })
async def run_training():
    import os
    import subprocess
    os.chdir('/open-r1')
    with open('zero3.yaml', 'w') as f:
#         f.write("""
# compute_environment: LOCAL_MACHINE
# debug: false
# deepspeed_config:
#   zero_stage: 3
# distributed_type: DEEPSPEED
# downcast_bf16: 'no'
# machine_rank: 0
# main_training_function: main
# mixed_precision: bf16
# num_machines: 1
# num_processes: 4
# rdzv_backend: static
# same_network: true
# tpu_env: []
# tpu_use_cluster: false
# tpu_use_sudo: false
# use_cpu: false
# """)
        f.write("""
            compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
            )

        
    cmd = [
        'accelerate', 'launch',
        '--config_file', 'zero3.yaml',
        'src/open_r1/grpo_lukas.py',
        '--output_dir', '/data/deepstock-R1-Distill-Qwen-7B-GRPO-full',
        # '--model_name_or_path', 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        '--model_name_or_path',MODEL_ID,
        "--report_to", "wandb",
        '--dataset_name', DATASET_ID,
        '--max_prompt_length', '512',
        '--max_completion_length', '1024',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '16',
        "--save_steps", "10",
        '--num_generations', '4',
        "--log_completions", "True",
        '--logging_steps', '1',
        '--log_level', 'debug',
        '--run_name', 'deepstock-check',
        # '--project_name', 'diabetesseek',
        # "--repo_id", "2084Collective/deepstock-v1",
        '--num_train_epochs', '1',
        '--bf16', 'true',
        # "--use_peft", "true",  
    ]
    subprocess.run(cmd, check=True)
    

@app.local_entrypoint()
async def main():
    await run_training.remote.aio()