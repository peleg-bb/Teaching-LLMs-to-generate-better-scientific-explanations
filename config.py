import os
import torch
from huggingface_hub import login
from transformers import TrainingArguments
from peft import LoraConfig

# Hugging Face and Neptune credentials
HUGGING_FACE_KEY = ""
NEPTUNE_API_TOKEN = ""
NEPTUNE_PROJECT = ""

# Device and Model configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
base_model_name = "meta-llama/Llama-2-7b-chat-hf"

# SFT configuration          
sft_training_args = TrainingArguments(
per_device_train_batch_size=4,
gradient_accumulation_steps=4,
optim="paged_adamw_32bit",
learning_rate=2e-5,
fp16=False,
bf16=False,
max_grad_norm=0.3,
num_train_epochs=2,
eval_steps=20,
save_steps=20,
logging_steps=20,
warmup_ratio=0.05,
evaluation_strategy="steps",
save_strategy="steps",
group_by_length=True,
output_dir="try_py",
load_best_model_at_end=True,
report_to="neptune",
save_safetensors=True,
lr_scheduler_type="cosine",
seed=42,    
)

sft_lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


# Reward model configuration          
rm_training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=1,
    eval_steps=20,
    warmup_ratio=0.05,
    evaluation_strategy="steps",
    save_strategy="steps", 
    load_best_model_at_end=True,
    report_to="neptune",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    seed=42,
    output_dir="try_rm_py",
)

rm_lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],   
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["scores"],
)

# RS configuration based on SFT configuration with modifications
rs_training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-5,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    num_train_epochs=2,
    eval_steps=2,
    save_steps=2,
    logging_steps=2,
    warmup_ratio=0.05,
    evaluation_strategy="steps",
    save_strategy="steps",
    group_by_length=True,
    output_dir="try_rs_py",
    load_best_model_at_end=True,
    report_to="neptune",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
)

rs_lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

def connect_to_neptune():
    """
    Sets up the environment variables required to connect to the Neptune project.

    This function sets the `NEPTUNE_API_TOKEN` and `NEPTUNE_PROJECT` environment variables
    using the values imported from `secret_config`. These variables are essential for 
    authenticating and interacting with the Neptune platform.

    Usage:
        Call this function before initializing any Neptune-related operations to ensure 
        that the required environment variables are set up correctly.
    """
    os.environ["NEPTUNE_API_TOKEN"] = NEPTUNE_API_TOKEN
    os.environ["NEPTUNE_PROJECT"] = NEPTUNE_PROJECT

def connect_to_huggingface():
    """
    Logs in to the Hugging Face Hub using the provided API token.

    This function authenticates your session with the Hugging Face Hub by calling the 
    `login` function from `huggingface_hub`, passing in the API token imported from 
    `secret_config`. This allows you to push and pull models from your Hugging Face 
    account.

    Usage:
        Call this function before performing any operations that require authentication 
        with the Hugging Face Hub, such as pushing a trained model to the hub.
    """
    login(token=HUGGING_FACE_KEY)
