import os
import datasets
from common import fine_tuner as common_ft
from peft import LoraConfig

cache_dir = common_ft.get_unique_cache_dir()

# The first X% of `train` split.
# With 2 V100 GPUs and the training options selected below, fine-tuning on this portion of the dataset takes approximately 30 minutes
dataset_fraction = 30
data = datasets.load_dataset('teknium/GPTeacher-General-Instruct', split=f'train[:{dataset_fraction}%]', cache_dir=cache_dir)

# Dataset modification function to merge multiple columns and add in some special tokens
def merge_columns(example):
    if example["input"]:
      prediction_format = """<Instruction>: %s
<Input>: %s
<Response>: %s"""
      example["prediction"] = prediction_format %(example["instruction"], example["input"], example["response"])
    else:
      prediction_format = """<Instruction>: %s
<Response>: %s"""
      example["prediction"] = prediction_format %(example["instruction"], example["response"])
    return example


# Create prediction column with the format we want for the fine tuning
tuning_data = data.map(merge_columns)

# Load a model using PEFT library for finetuning
ft = common_ft.AMPFineTuner("bigscience/bloom-1b1")

# Set LoRA training configuration
ft.set_lora_config(
LoraConfig(
          r=16,
          lora_alpha=32,
          target_modules=["query_key_value", "xxx"],
          lora_dropout=0.05,
          bias="none",
          task_type="CAUSAL_LM"
      )
)

# Set training arguments 
# see fine_tuner.py for list of defaults and huggingface's transformers.TrainingArguments
# or the full list of arguments
ft.training_args.num_train_epochs=1
ft.training_args.warmup_ratio=0.03
ft.training_args.max_grad_norm=0.3
ft.training_args.learning_rate=2e-4

# Execute training and save adapter
ft.train(tuning_data, "prediction", os.getenv("CUSTOM_LORA_ADAPTERS_DIR")+"/bloom1b1-lora-instruct")