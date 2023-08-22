## Improving on the Sample Adapters
We wanted to make sure to produce examples that could be easily and very cheaply replicated by CML users, so we've made a number of conscious choixes to minimize the time spent on expensive hardware. The following items could be changed to improve performance of the resulting adapters (with some increase in fine-tuning time and hardware cost).

### Different or Large Base LLM
There are many families of opensource base models that are available today (bloom, falcon, llama2 to name a few)
- Each of these model families have released base models of varying size, allowing for better downstream performance for fine-tuned tasks when using the larger variant.
- Larger base models means longer training times and larger GPU memory requirements

> Requirements: The implementations shown in this repository rely on the accelerate library, any model that supports loading via the accelerate library (we use the `device_map` accelerate argument) should be quantizable and therefore fine-tunable with QLoRA. (see huggingface [guide on this topic](https://huggingface.co/blog/4bit-transformers-bitsandbytes#what-are-the-supported-models))

### Larger and Better Curated Datasets
Fine-tuning is a wasted effort without a good dataset, larger and better curated datasets are the most critical part of producing continually improving fine-tuning results
- Larger Datasets means longer training times

### Fine-tuning Arguments
We make use of the huggingface fine-tuning libraries in TRL (https://huggingface.co/docs/trl/) and PEFT (https://huggingface.co/docs/peft/index). These libraries help to launch fine-tuning operations and also allow for the modification of training arguments used by the underlying pytorch.

#### - trl
This library implements further conveniences for fine-tuning 
  - [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
    - Adjusting learning_rate, num_epochs and etc can change the fine-tuning time and resulting adapater performance
  - [SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
    - Bonus optimizations like packing are implemented in this library and speed up fine-tuning at the cost of some performance

#### peft
  - [LoRA configuration](https://huggingface.co/docs/peft/conceptual_guides/lora#common-lora-parameters-in-peft)
    - Configurations here allow for customization of how LoRA is applied to the loaded base model
  - [BitsAndBytes configuration](https://huggingface.co/docs/transformers/main_classes/quantization)
    - The Q in QLoRA requires quantization which is controlled by a bitsandbytes configuration

### Merging Weights for improved latency
At larger scales, loading both the base model and LoRA may incur some extra latency time. It is possible to merge the the adapter weights to the base model to create a new standalone model for inference, see [Merge LoRA weights into the base model](https://huggingface.co/docs/peft/conceptual_guides/lora#merge-lora-weights-into-the-base-model) in the huggingface PEFT documentation.