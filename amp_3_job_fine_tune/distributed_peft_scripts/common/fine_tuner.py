import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["BITSANDBYTES_NOWELCOME"]="1"
os.environ["DATASETS_VERBOSITY"]="error"

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model 
from trl import SFTTrainer
import datasets

def get_unique_cache_dir():
        # Use a cache dir specific to this session, since workers and sessions will share project files
        return "~/.cache/" + os.environ['CDSW_ENGINE_ID'] + "/huggingface/datasets"

class AMPFineTuner:

    # Load basemodel from huggingface
    # Default: bigscience/bloom-1b1
    def __init__(self, base_model, auth_token = ""):

        # Load the base model and tokenizer
        print("Load the base model and tokenizer...\n")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token = auth_token)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        compute_dtype = getattr(torch, "float16")

        # Configuration to load the model in 4bit quantized mode
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
            
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token = auth_token,
        )

        # transformers.TrainingArguments defaults
        self.training_args = TrainingArguments(
                output_dir="outputs",
                num_train_epochs=1,
                optim="paged_adamw_32bit",
                per_device_train_batch_size=1, 
                gradient_accumulation_steps=4,
                warmup_ratio=0.03, 
                max_grad_norm=0.3,
                learning_rate=2e-4, 
                fp16=True,
                logging_steps=1,
                lr_scheduler_type="constant",
                disable_tqdm=True,
                report_to='tensorboard',
        )


    # Use PEFT library to set LoRA training config and get trainable peft model
    def set_lora_config(self, lora_config):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

        self.lora_config = lora_config

        self.model = get_peft_model(self.model, self.lora_config)
    
    # Train/Fine-tune model with SFTTrainer and a provided dataset
    def train(self, tuning_data, dataset_text_field, output_dir, packing = True, max_seq_length = 1024):
        trainer = SFTTrainer(
            model=self.model, 
            train_dataset=tuning_data,
            peft_config=self.lora_config,
            tokenizer=self.tokenizer,
            dataset_text_field = dataset_text_field,
            packing=packing,
            max_seq_length=max_seq_length,
            args=self.training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        print("Begin Training....")
        trainer.train()
        print("Training Complete!")
        # Will only save from the main process.
        # This works for peft models thanks to https://github.com/huggingface/transformers/pull/24073
        trainer.save_model(output_dir)