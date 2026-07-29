import subprocess
import yaml
import tempfile
import os

model = "vgg16"

def run_experiment(config):
    with tempfile.NamedTemporaryFile(mode="w", suffix="yaml", encoding="utf-8", delete=False) as f:
        yaml.safe_dump(config, f, sort_keys=False)
        yaml_path = f.name
    try:
        subprocess.run(["python", "main.py", "--config", yaml_path], check=True)
    finally:
        os.remove(yaml_path)

with open(f"./configs/gpadmm_cifar10_{model}_pretrained_global_sgd_default.yaml", "r") as f:
    config = yaml.safe_load(f)

# GPU Setting
config["common"]["gpu"] = "cuda:1"

# Step 1 : cycle
cycle_list = range(1, 10, 1)

for cycle in cycle_list:
    experiment_name = f"./output/gpadmm_cifar10_{model}_pretrained_global_sgd_cycle_{cycle}"
    config["gpadmm"]["cycle"] = cycle
    config["common"]["save_dir"] = experiment_name
    config["common"]["output_dir"] = f"{experiment_name}.jsonl"
    run_experiment(config)


# Step 2 : grow internal
interval_list = range(3, 10, 1)
for interval in interval_list:
    experiment_name = f"./output/gpadmm_cifar10_{model}_pretrained_global_sgd_interval_{interval}"
    config["gpadmm"]["grow_internal"] = interval
    config["common"]["save_dir"] = experiment_name
    config["common"]["output_dir"] = f"{experiment_name}.jsonl"
    run_experiment(config)

# Step 3 : re epoch
re_epoch_list = range(3, 10, 1)
for re_epoch in re_epoch_list:
    experiment_name = f"./output/gpadmm_cifar10_{model}_pretrained_global_re_epoch_{re_epoch}"
    config["gpadmm"]["num_re_epochs"] = re_epoch
    config["common"]["save_dir"] = experiment_name
    config["common"]["output_dir"] = f"{experiment_name}.jsonl"
    run_experiment(config)


# Step 4 : learning rate
lr_list = [0.1, 0.05, 0.01, 0.005, 0.001]
for lr in lr_list:
    experiment_name = f"gpadmm_cifar10_{model}_pretrained_global_lr_{lr}"
    config["common"]["save_dir"] = experiment_name
    config["common"]["output_dir"] = f"{experiment_name}.jsonl"
    config["common"]["lr"] = lr
    run_experiment(config)