import subprocess
import yaml
import tempfile
import os
import argparse

argument_parser = argparse.ArgumentParser(description="Run experiments with different configurations.")
argument_parser.add_argument("--model", type=str, default="vgg16", help="Model to use for the experiment (default: vgg16)")
argument_parser.add_argument("--gpu", type=int, default=1, help="GPU index to use for the experiment (default: 1)")
args = argument_parser.parse_args()

model = args.model
gpu = args.gpu

def run_experiment(config):
    print(f"Running experiment with config: {config}")
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
config["common"]["device"] = f"cuda:{gpu}"

sparsity_list = [0.8, 0.9, 0.95, 0.98]
for sparsity in sparsity_list:
    config["gpadmm"]["sparsity"] = sparsity
    
    # Step 1 : cycle
    cycle_list = range(1, 10, 1)

    for cycle in cycle_list:
        experiment_name = f"gpadmm_cifar10_{sparsity}_{model}_pretrained_global_sgd_cycle_{cycle}"
        config["gpadmm"]["num_cycles"] = cycle
        config["common"]["save_dir"] = f"./runs/{experiment_name}"
        config["common"]["output_dir"] = f"./output/{experiment_name}.jsonl"
        run_experiment(config)


    # Step 2 : grow interval
    interval_list = range(3, 10, 1)
    for interval in interval_list:
        experiment_name = f"gpadmm_cifar10_{sparsity}_{model}_pretrained_global_sgd_interval_{interval}"
        config["gpadmm"]["grow_interval"] = interval
        config["common"]["save_dir"] = f"./runs/{experiment_name}"
        config["common"]["output_dir"] = f"./output/{experiment_name}.jsonl"
        run_experiment(config)

    # Step 3 : re epoch
    re_epoch_list = range(3, 10, 1)
    for re_epoch in re_epoch_list:
        experiment_name = f"gpadmm_cifar10_{sparsity}_{model}_pretrained_global_re_epoch_{re_epoch}"
        config["gpadmm"]["num_re_epochs"] = re_epoch
        config["common"]["save_dir"] = f"./runs/{experiment_name}"
        config["common"]["output_dir"] = f"./output/{experiment_name}.jsonl"
        run_experiment(config)


    # Step 4 : learning rate
    lr_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    for lr in lr_list:
        experiment_name = f"gpadmm_cifar10_{sparsity}_{model}_pretrained_global_lr_{lr}"
        config["common"]["save_dir"] = f"./runs/{experiment_name}"
        config["common"]["output_dir"] = f"./output/{experiment_name}.jsonl"
        config["common"]["lr"] = lr
        run_experiment(config)
        
    min_layer_ratio = [0.01, 0.05, 0.1, 0.2, 0.3]
    for ratio in min_layer_ratio:
        experiment_name = f"gpadmm_cifar10_{sparsity}_{model}_pretrained_global_min_layer_ratio_{ratio}"
        config["common"]["save_dir"] = f"./runs/{experiment_name}"
        config["common"]["output_dir"] = f"./output/{experiment_name}.jsonl"
        config["gpadmm"]["min_layer_ratio"] = ratio
        run_experiment(config)