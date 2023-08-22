import os
import requests

try:
    # Launch workers when using CML
    from cml.workers_v1 import launch_workers, await_workers
except ImportError:
    # Launch workers when using CDSW
    from cdsw import launch_workers, await_workers

try:
    NUM_GPU_WORKERS = int(os.getenv("NUM_GPU_WORKERS"))
except ValueError as verr:
    sys.exit(
        "Invalid value for GPUs for fine-tuning selected: NUM_GPU_WORKERS = %s" % os.getenv("NUM_GPU_WORKERS")
    )

# Check that the current workspace allows workloads to use GPUs
def check_gpu_enabled():
    APIv1 = os.getenv("CDSW_API_URL")
    PATH = "site/config/"
    API_KEY = os.getenv("CDSW_API_KEY")

    if NUM_GPU_WORKERS < 1:
        sys.exit(
            "Invalid number of GPUs for fine-tuning selected: NUM_GPU_WORKERS = %d" % NUM_GPU_WORKERS
        )

    url = "/".join([APIv1, PATH])
    res = requests.get(
        url,
        headers={"Content-Type": "application/json"},
        auth=(API_KEY, ""),
    )
    max_gpu_per_engine = res.json().get("max_gpu_per_engine")

    if max_gpu_per_engine < 1:
        # Failure at this point is because GPUs are not eabled on this workspace.
        # Ask your admin about quota and autoscaling rules for GPU
        sys.exit(
            "GPUs are not enabled in this CML Workspace"
        )
    print("GPUs are enabled in this workspace.")

# Check that there are available GPUs or autoscalable GPUs available
def check_gpu_launch():
    # Launch a worker that uses GPU to see if any gpu is available or autoscaling is possible
    worker = launch_workers(
        n=NUM_GPU_WORKERS, cpu=2, memory=4, nvidia_gpu=1, code="print('GPU Available')"
    )

    # Wait for 10 minutes to see if worker pod reaches success state
    worker_schedule_status = await_workers(
        worker, wait_for_completion=True, timeout_seconds=600
    )
    if len(worker_schedule_status["failures"]) == 1:
        cdsw.stop_workers(worker_schedule_status["failures"][0]["id"])
        # Failure at this point is due not enough GPU resources at the time of launch.
        # Ask your admin about quota and autoscaling rules for GPU
        sys.exit("Unable to allocate GPU resource within 10 minutes")

print("Checking the enablement and availibility of GPU in the workspace")
check_gpu_enabled()
check_gpu_launch()