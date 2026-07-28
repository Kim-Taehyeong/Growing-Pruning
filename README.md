# Growing-Pruning

## Config-based runs

Experiments can be launched from YAML:

```bash
python main.py --config configs/admm_mnist_lenet_smoke.yaml
python main.py --config configs/admm_mnist_lenet_sgd_smoke.yaml
python main.py --config configs/gpadmm_mnist_lenet_smoke.yaml
python main.py --config configs/gpadmm_layerwise_mnist_lenet_smoke.yaml
python main.py --config configs/admm_cifar10_vgg16_pretrained_smoke.yaml
python main.py --config configs/admm_cifar10_resnet56_pretrained_smoke.yaml
```

All experiment parameters live in YAML. Use `common` for shared training/data
settings and the method-specific section for pruning settings:

```yaml
method: gpadmm_full

common:
  dataset: mnist
  model: lenet
  seed: 1
  device: auto
  optimizer: adam
  lr: 0.001
  rho: 0.01
  num_pre_epochs: 3

gpadmm:
  gpadmm_prune_scope: global
  sparsity: 0.95
  sparsity_method: uniform
  num_cycles: 2
  grow_interval: 2
  num_re_epochs: 1
  c: 0.5
```

## Method parameters

Shared parameters include dataset/model selection, batch sizes, paths,
checkpoint/pretrained options, seed, learning rate, ADMM penalty parameters
(`rho`, `alpha`, `l2`), and dense pretraining epochs.

Device selection is config-driven:

- `device: auto` uses CUDA when available, otherwise CPU.
- `device: cpu` forces CPU.
- `device: cuda` uses the default CUDA device.
- `device: cuda:1` selects a specific GPU.
- `gpu_id: 1` can be used with `device: auto` or `device: cuda` as a shorthand.

Optimizer selection is also config-driven:

- `optimizer: adam` uses the existing `PruneAdam`.
- `optimizer: sgd` uses `PruneSGD`, which supports the same pruning masks during
  ADMM retraining.
- SGD-specific knobs are `momentum`, `dampening`, `weight_decay`, and
  `nesterov`.

CIFAR pretrained models can be loaded from `chenyaofo/pytorch-cifar-models` with
`cifar_pretrained: true`. Supported local config names include `resnet20`,
`resnet56`, `vgg16`, `vgg19`, and `mobilenet_v2`.

Reference dense accuracy from that model zoo:

| Dataset | Model | Top-1 | Top-5 |
| --- | --- | ---: | ---: |
| CIFAR-10 | resnet56 | 94.37 | 99.83 |
| CIFAR-10 | vgg16_bn | 94.16 | 99.71 |
| CIFAR-100 | resnet56 | 72.63 | 91.94 |
| CIFAR-100 | vgg16_bn | 74.00 | 90.56 |

ADMM uses layer-wise pruning settings:

- `percent`: per-prunable-layer sparsity list.
- `sparsity`: optional shortcut; when set, the first prunable layer is kept
  dense and all following layers use the same sparsity.
- `num_epochs`: ADMM main training epochs.
- `num_re_epochs`: fixed-mask retraining epochs.
- `l1`: use L1 projection/pruning instead of cardinality pruning.

GP-ADMM uses global cycle/growth settings:

- `gpadmm_prune_scope`: `global` keeps the original global-threshold
  GP-ADMM behavior. `layerwise` uses the `sparsity_method` allocation below for
  actual pruning and ADMM projection.
- `sparsity`: final global target sparsity.
- `sparsity_method`: one of `uniform`, `er`, or `erk`. In `layerwise` mode this
  determines each layer's final sparsity target; in `global` mode it is kept for
  config consistency while pruning uses the original global threshold.
- `num_cycles`: number of prune/ADMM/regrow cycles.
- `grow_interval`: ADMM epochs per cycle.
- `c`: regrowth ratio multiplier used to derive the per-cycle grow fraction.
- `min_layer_keep_ratio`: lower bound that prevents a layer from becoming fully
  pruned during global pruning.
- `num_re_epochs`: final fixed-mask retraining epochs.

Models and checkpoints are saved under `save_dir/run_name` or the generated run
directory:

```text
runs/<run-name>/
  checkpoints/
  models/
  metrics.jsonl
  run.log
```
