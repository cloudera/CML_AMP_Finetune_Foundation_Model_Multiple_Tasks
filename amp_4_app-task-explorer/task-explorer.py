import gradio as gr
import time
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from glob import glob
from collections import namedtuple 


BASE_MODEL = "bigscience/bloom-1b1"
PREBUILT_LORA_ADAPTERS_DIR = "amp_adapters_prebuilt"
if os.path.exists(PREBUILT_LORA_ADAPTERS_DIR):
  print("Found prebuilt adapters dir")
  
prebuilt_lora_adapter_dirs = glob(PREBUILT_LORA_ADAPTERS_DIR+"/*/", recursive = False)

CUSTOM_LORA_ADAPTERS_DIR = os.getenv("CUSTOM_LORA_ADAPTERS_DIR")
if os.path.exists(CUSTOM_LORA_ADAPTERS_DIR):
  print("Found custom adapters dir")
  
custom_lora_adapter_dirs = glob(CUSTOM_LORA_ADAPTERS_DIR+"/*/", recursive = False)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, return_dict=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

all_lora_adapter_dirs = prebuilt_lora_adapter_dirs + custom_lora_adapter_dirs

# List of generative AI usecases to display in the UI, mapped to adapter folder path
# Custom adapters located in CUSTOM_LORA_ADAPTERS_DIR will be appended to this list as Custom Adapter: [adapter-dir-name]
# Custom adapters will not have sample engineered prompts autofilled

usecase_adapter_dict = {"Generate SQL":"amp_adapters_prebuilt/bloom1b1-lora-sql/", "Detoxify Statement":"amp_adapters_prebuilt/bloom1b1-lora-toxic/", "General Instruction-Following":"amp_adapters_prebuilt/bloom1b1-lora-instruct/"}
for custom_adapter in custom_lora_adapter_dirs:
   usecase_adapter_dict["Custom Adapter: %s" % custom_adapter] = custom_adapter

for adapter in all_lora_adapter_dirs:
  # See https://github.com/huggingface/peft/issues/211
  # This is a PEFT Model, we can load another adapter
  if hasattr(model, 'load_adapter'):
    model.load_adapter(adapter, adapter_name=adapter)
  # This is a regular AutoModelForCausalLM, we should use PeftModel.from_pretrained for this first adapter load
  else:
    model = PeftModel.from_pretrained(model=model, model_id=adapter, adapter_name=adapter)
  print("Loaded PEFT Adapter: %s" % adapter)

loaded_adapters = list(model.peft_config.keys())

def generate(prompt, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k):
  batch = tokenizer(prompt, return_tensors='pt')
  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch,
                                    max_new_tokens=max_new_tokens,
                                    repetition_penalty=repetition_penalty,
                                    temperature=temperature,
                                    num_beams=num_beams,
                                    top_p=top_p,
                                    top_k=top_k)
  prompt_length = len(prompt)
  return tokenizer.decode(output_tokens[0], skip_special_tokens=True)[prompt_length:]

def get_responses(adapter_select, prompt, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k):
  # Using this syntax to ensure inference without adapter
  # https://github.com/huggingface/peft/issues/430
  with model.disable_adapter():
    base_generation = generate(prompt,
                                max_new_tokens,
                                temperature,
                                repetition_penalty,
                                num_beams,
                                top_p,
                                top_k)
  
  model.set_adapter(adapter_select)

  if "None" in  adapter_select:
    lora_generation = ""
  else:
    lora_generation = generate(prompt,
                               max_new_tokens,
                               temperature,
                               repetition_penalty,
                               num_beams,
                               top_p,
                               top_k)
  print("Generating with  PEFT Adapter: %s" % adapter_select)
  return (gr.Textbox.update(value=base_generation, visible=True,  lines=5), gr.Textbox.update(value=lora_generation, visible=True,  lines=5))

theme = gr.themes.Default().set(
    block_title_padding='*spacing_md',
    block_title_text_size='*text_lg',

)

css = """
"""

with gr.Blocks(theme=theme, css=css) as demo:
    with gr.Row():
        gr.Markdown("# Fine-tuned Foundation Model for Multiple Tasks")
    with gr.Row():
         with gr.Box():
                with gr.Row():
                    with gr.Column():
                        usecase_select = gr.Radio(list(usecase_adapter_dict.keys()), value="Please select a task to complete...", label="Choose a generative AI task", info="Compare Base Model and Fine-tuned PEFT Adapter Responses", interactive=True)
                    
                        with gr.Row():
                            with gr.Row(variant="panel"):
                                with gr.Column():
                                    base_model = gr.TextArea(label="\tBase Model", value="bigscience/bloom1b1", container = False, lines=1, visible=True, interactive=False)
                                with gr.Column():
                                    adapter_select = gr.TextArea(label="\tPEFT[LoRA] Adapter", container = False, value="...", lines=1, visible=True, interactive=False)
                        input_txt = gr.Textbox(label="Engineered Prompt", value="Select a task example above to edit...", lines=3, interactive=False)
                        with gr.Accordion("Advanced Generation Options", open=False):
                            with gr.Column():
                                with gr.Row():
                                    max_new_tokens = gr.Slider(
                                        minimum=0, maximum=256, step=1, value=50,
                                        label="Max New Tokens",
                                    )
                                    num_beams = gr.Slider(
                                        minimum=1, maximum=10, step=1, value=1,
                                        label="Num Beams",
                                    )
                                    repetition_penalty = gr.Slider(
                                        minimum=0.01, maximum=4.5, step=0.01, value=1.1,
                                        label="Repeat Penalty",
                                    )

                                with gr.Row():
                                    temperature = gr.Slider(
                                        minimum=0.01, maximum=1.99, step=0.01, value=0.7,
                                        label="Temperature",
                                    )

                                    top_p = gr.Slider(
                                        minimum=0, maximum=1.0, step=0.01, value=1.0,
                                        label="Top P", interactive = True,
                                    )

                                    top_k = gr.Slider(
                                        minimum=0, maximum=200, step=1, value=0,
                                        label="Top K",
                                    )
                        with gr.Row():
                            gen_btn = gr.Button(value="Generate", variant="primary", interactive=False)
                            clear_btn = gr.ClearButton(value="Reset", components=[], queue=False)
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            output_plain_txt = gr.Textbox(value="", label="Base Model Response",lines=1, interactive=False, visible=True, placeholder="...", container = False)
                        with gr.Row():
                            output_adapter_txt = gr.Textbox(value="", label="Fine-tuned PEFT Adapter Response", lines=1, interactive=False, visible=True, placeholder="...", container = False)
                        
             

    examples_params_list= [input_txt, repetition_penalty, temperature, max_new_tokens, num_beams, top_p, top_k]
    example_tuple = namedtuple("example_named",["input_txt", "repetition_penalty", "temperature", "max_new_tokens", "num_beams", "top_p","top_k", "placeholder_txt"])
    ex_instruct = example_tuple("<Instruction>: Answer the question using the provided input, be concise. How does CML unify self-service data science and data engineering?\n\n<Input>: Cloudera Machine Learning is Cloudera’s cloud-native machine learning platform built for CDP. Cloudera Machine Learning unifies self-service data science and data engineering in a single, portable service as part of an enterprise data cloud for multi-function analytics on data anywhere.\n\n<Response>:", 0.99, 0.75, 33, 1, 1.0, 0, "")
    ex_sql = example_tuple("<TABLE>: CREATE TABLE jedi (id VARCHAR, lightsaber_color VARCHAR)\n<QUESTION>: Give me a list of jedi that have gold color lightsabers.\n<SQL>: ", 1.15, 0.8, 14, 1, 1.0, 0, "")
    ex_toxic = example_tuple("<Toxic>: I hate Obi Wan, he always craps on me about the dark side of the force.\n<Neutral>: ", 1.22, 0.75, 19, 1, 1.0, 0, "")
    ex_empty = example_tuple("",1.0, 0.7, 50, 1, 1.0, 0, "Select a task example above to edit...")
    ex_custom = example_tuple("",1.0, 0.7, 50, 1, 1.0, 0, "")

    def set_example(usecase):
        interactive_prompt = True
        # Set example inputs (prompt, inference options) for known usecases and custom adapters
        if usecase in list(usecase_adapter_dict.keys()):
            if "General Instruction-Following" in usecase:
                update_tuple = ex_instruct
            elif "Generate SQL" in usecase:
                update_tuple = ex_sql
            elif "Detoxify Statement" in usecase:
                update_tuple = ex_toxic
            elif "Custom" in usecase:
                update_tuple = ex_custom
        # Clear out inputs when UI is reset
        else:
            interactive_prompt = False
            update_tuple = ex_empty

        return (gr.Textbox.update(value=update_tuple.input_txt, interactive=interactive_prompt, placeholder=update_tuple.placeholder_txt),
                gr.Slider.update(value=update_tuple.repetition_penalty),
                gr.Slider.update(value=update_tuple.temperature),
                gr.Slider.update(value=update_tuple.max_new_tokens),
                gr.Slider.update(value=update_tuple.num_beams),
                gr.Slider.update(value=update_tuple.top_p),
                gr.Slider.update(value=update_tuple.top_k),
                gr.Textbox.update(value="", visible=True, lines=1),
                gr.Textbox.update(value="", visible=True, lines=1))
    
    def set_usecase(usecase):
        # Slow user down to highlight changes
        time.sleep(0.5)
        print(usecase)
        if usecase in usecase_adapter_dict:
           return (gr.Textbox.update(value=usecase_adapter_dict[usecase], visible=True), gr.Button.update(interactive=True))
        else: 
            return (gr.TextArea.update(value="...", visible=True), gr.Button.update(interactive=False))
    
    def clear_out():
        empty_example = set_example("")
        cleared_tuple = empty_example + (gr.TextArea.update(value="..."), gr.TextArea.update(value="", lines=1), gr.Textbox.update(value="", lines=1), gr.Textbox.update(value="Please select a fine-tuned adapter...")) 
        return cleared_tuple
    
    def show_outputs():
        return (gr.Textbox.update(visible=True), gr.Textbox.update(visible=True))
    
    def disable_gen():
        return gr.Button.update(interactive=False)
    
    usecase_select.change(set_usecase, inputs = [usecase_select], outputs=[adapter_select, gen_btn])

    adapter_select.change(set_example, inputs = [usecase_select], outputs=[input_txt, repetition_penalty, temperature, max_new_tokens, num_beams, top_p, top_k, output_plain_txt, output_adapter_txt])

    clear_btn.click(disable_gen, queue = False, inputs = [], outputs=[gen_btn]).then(clear_out, queue = False, inputs = [], outputs=[input_txt, repetition_penalty, temperature, max_new_tokens, num_beams, top_p, top_k, output_plain_txt, output_adapter_txt, adapter_select, output_adapter_txt, output_plain_txt, usecase_select])

    gen_btn.click(show_outputs, inputs = [], outputs=[output_plain_txt,output_adapter_txt]).then(get_responses, inputs=[adapter_select, input_txt, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k],
                        outputs=[output_plain_txt,output_adapter_txt])

demo.launch(server_port=int(os.getenv('CDSW_APP_PORT')),
           enable_queue=True,
           show_error=True,
           server_name='127.0.0.1',
)