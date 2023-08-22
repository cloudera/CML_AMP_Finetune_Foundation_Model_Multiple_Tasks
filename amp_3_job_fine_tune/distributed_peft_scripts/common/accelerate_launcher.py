import sys
import os
import subprocess
from string import Template

def launch_distributed_script(tune_script):
  try:
    NUM_GPU_WORKERS = int(os.getenv("NUM_GPU_WORKERS")) # Total number of workers to use (including the main session)
  except ValueError as verr:
    sys.exit(
        "Invalid value for GPUs for fine-tuning selected: NUM_GPU_WORKERS = %s" % os.getenv("NUM_GPU_WORKERS")
    )
  worker_cpu = 2
  worker_memory = 8

  try:
      # Launch workers when using CML
      from cml.workers_v1 import launch_workers
  except ImportError:
      # Launch workers when using CDSW
      from cdsw import launch_workers
      
  MASTER_IP = os.environ["CDSW_IP_ADDRESS"]


  # Create the config files used for launching Accelerate
  print("Set up Accelerate config files for distributed fine-tuning...")
  if NUM_GPU_WORKERS == 1:
    text_file = open("amp_3_job_fine_tune/distributed_peft_scripts/common/accelerate_configs/accelerate_single_config.yaml.tmpl")
  elif NUM_GPU_WORKERS > 1:
    text_file = open("amp_3_job_fine_tune/distributed_peft_scripts/common/accelerate_configs/accelerate_multi_config.yaml.tmpl")
  else:
    sys.exit(
        "Invalid number of GPUs for fine-tuning selected: NUM_GPU_WORKERS = %d" % NUM_GPU_WORKERS
    )
  data = text_file.read()
  text_file.close()


  conf_dir = "./.tmp_accelerate_configs/"
  config_path_tmpl = conf_dir + "${WORKER}_config.yaml"
  command_tmpl = "accelerate launch --config_file $CONF_PATH $TUNE_SCRIPT"

  print("Number of workers: ", NUM_GPU_WORKERS)


  os.makedirs(conf_dir, exist_ok=True)
  for i in range(NUM_GPU_WORKERS):
    config_file = Template(data)
    config_file = config_file.substitute(MACHINE_RANK=i, MAIN_SESSION_IP=MASTER_IP, NUM_MACHINES=NUM_GPU_WORKERS, NUM_PROCESSES=NUM_GPU_WORKERS)
    
    config_path = Template(config_path_tmpl).substitute(WORKER=i)
    
    text_file = open(config_path, "w")
    text_file.write(config_file)
    text_file.close()
    
    command = Template(command_tmpl).substitute(CONF_PATH=config_path, TUNE_SCRIPT=tune_script)
    # If worker num 0 this is the main process and should run locally in this session
    if i == 0:
      print("Launch accelerate locally (this session acts as worker of rank 1)...")
      print("\t Command: [%s]" % command)
      main_cmd = subprocess.Popen([f'bash -c "{command}" '], shell=True)

    # All other accelerate launches will use rank 1+
    else:
      print(("Launch CML worker %i and launch accelerate within them ...", i))
      print("\t Command: [%s]" % command)
      launch_workers(name=f'LoRA Train Worker {i}', n=1, cpu=worker_cpu, memory=worker_memory, nvidia_gpu = 1,  code="!"+command + " &> /dev/null")
      
  # Waiting for all subworkers to ready up...
  main_cmd.communicate()