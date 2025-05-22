import os
import json
import torch
from datetime import datetime
import time
from contextlib import contextmanager


def save_checkpoint(
    model, epoch, val_acc, hyperparams, checkpoint_dir="../checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f"model_epoch{epoch}_valacc{val_acc:.4f}.pth"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), path)
    # Save hyperparameters/config
    config_path = os.path.join(
        checkpoint_dir, f"config_epoch{epoch}_valacc{val_acc:.4f}.json"
    )
    with open(config_path, "w") as f:
        json.dump(hyperparams, f, indent=2)
    print(f"Checkpoint saved: {path}")


def log_metrics(metrics, log_dir="../logs", run_name=None):
    os.makedirs(log_dir, exist_ok=True)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}.json")
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics logged: {log_path}")


@contextmanager
def timer(name, timing_log=None):
    start = time.time()
    yield
    end = time.time()
    duration = end - start
    print(f"{name} took {duration:.2f} seconds.")
    if timing_log is not None:
        timing_log[name] = duration


def log_timing(timing_log, log_dir="../logs", run_name=None):
    os.makedirs(log_dir, exist_ok=True)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"timing_{run_name}.json")
    with open(log_path, "w") as f:
        json.dump(timing_log, f, indent=2)
    print(f"Timing metrics logged: {log_path}")
