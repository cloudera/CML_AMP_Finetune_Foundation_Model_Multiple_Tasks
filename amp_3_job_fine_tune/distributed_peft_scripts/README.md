# distributed_peft_scripts
In this directory are all the scripts used to demonstrate distributed QLoRA finetuning.

## task_*_fine_tuner.py
These files are the scripts which describe the individual fine-tuning tasks.
In each script we define:
- Dataset
    - Using the datasets library from huggingface
    - Using publicly available datasets from huggingface
    - A mapping function to format the samples to be used for fine-tuning
- Base Model
    - The base model from huggingface to use for fine-tuning
- LoraConfig
    - The LoRA adapter fine-tuning configuration to use with the peft library
    - See [Common LoRA parameters in PEFT](https://huggingface.co/docs/peft/conceptual_guides/lora#common-lora-parameters-in-peft)
- Training Arguments
    - The training arguments to use during the fine-tuning loop
    - See [transformers.trainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- Execution
    - Finally a simple wrapper function to launch fine-tuning and set an output directory

## common
This directory holds some common code for usage of huggingface training libraries and accelerate configuration setup
### accelerate_configs
Template and temporary directory for distributed/single execution with [accelerate](https://github.com/huggingface/accelerate)
### accelerate_launcher.py
This script is the entrypoint for the CML Jobs created with this amp. It performs three main tasks:
- Creation of accelerate config yamls
    - Each accelerate worker requires a different config file to perform distributed fine-tuning
- Launching CML Workers
    - The [CML Workers API](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html) makes it easy to launch accelerate workers from within a CML Session or Job workload. These CML Workers share the same project filesystem as the main Session/Job and work seamlessly with accelerate.
- Executing accelerate cli command
    - The easiest way to launch accelerate for distributed fine-tuning is to use the accelerate cli pointing to a copy of your fine-tuning script.
    - The CML Workers described above makes this possible without extra copies of project files.
### fine_tuner.py
This script is a simple python module containing shared code between our multiple fine-tuning jobs.
The class AMPFineTuner handles the following  components:
#### Setup
- Model Loading
    - The base model is loaded and reloaded as a PEFT library "peft_model" to be able to perform PEFT fine-tuning techniques
- BitsAndBytesConfig
    - For quantizing the model for QLoRA
- transformers TrainingArguments
    - Defaults are set here, but these are modified in task_*_fine_tuner.py as well
- LoraConfig
    - Used during loading base model as a "peft_model"
    - Defaults are set here, but these are modified in task_*_fine_tuner.py as well
#### SFTTrainer
Finally a train function which is a wrapper around SFTTrainer

[SFTTrainer documentation](https://huggingface.co/docs/trl/main/en/sft_trainer)
- This is a convenient huggingface API that simplifies fine-tuning code
- Uses the training TrainingArguments and LoraConfig from above or in launcher scripts
- Also has some extra optimizations like [packing](https://huggingface.co/docs/trl/main/en/sft_trainer#packing-dataset-constantlengthdataset)

SFTTrainer is integrated with accelerate, so saving the resulting fine-tuned adapters requires no modifications to the code. This class can be used with or without accelerate.
