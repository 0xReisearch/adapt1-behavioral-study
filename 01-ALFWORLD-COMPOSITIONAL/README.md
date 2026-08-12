# ALFWorld compositional replication instructions

## 1. Fixed protocol

- Python 3.12
- ALFWorld 0.4.2
- TextWorld 1.7.0, installed by ALFWorld
- 3,520 official training games collected once with seed 42
- Candidate-policy initialization seeds 0, 1, 2, 3, and 4
- 45 training epochs per seed
- ID evaluation on all 140 official `valid_seen` games per seed
- OOD evaluation on all 134 official `valid_unseen` games per seed
- Maximum 50 actions per game
- No expert actions, labels, or online feedback during evaluation
- Primary score: unweighted macro success across the six official task families

The included planner file corresponds to the compositional implementation used for the recorded result. The clean procedure below rebuilds state from the official training split. No prior run state is included.

## 2. Create the environment

From this experiment directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools<81' wheel
python -m pip install --no-build-isolation 'alfworld[full]==0.4.2' pyyaml numpy
```

Install the PyTorch build appropriate for the host CPU, CUDA, or ROCm runtime. Follow the official PyTorch selector for the exact command.

Clone Adapt-1 and install the included planner implementation:

```bash
git clone https://github.com/00unitrei/neuroadapt-api.git neuroadapt-api
cp core/plan_memory.py neuroadapt-api/src/neuroadapt/plan_memory.py
python -m pip install -e neuroadapt-api
```

Download ALFWorld data and set the runtime paths:

```bash
export ALFWORLD_DATA="$PWD/alfworld-data"
alfworld-download
export PYTHONPATH="$PWD/neuroadapt-api/src:$PWD/harness"
export NEUROADAPT_TORCH_DEVICE=auto
```

Confirm these paths exist before starting:

```bash
test -d "$ALFWORLD_DATA/json_2.1.1/train"
test -d "$ALFWORLD_DATA/json_2.1.1/valid_seen"
test -d "$ALFWORLD_DATA/json_2.1.1/valid_unseen"
test -f "$ALFWORLD_DATA/logic/alfred.pddl"
```

## 3. Run a one-seed verification

This performs a complete clean collection, fit, and evaluation for seed 0:

```bash
python harness/run_multiseed.py \
  --config harness/config_tw.yaml \
  --output reproduced-seed-0 \
  --seeds 0,1 \
  --collection-seed 42 \
  --train-episodes 3520 \
  --train-batch-size 16 \
  --id-batch-size 20 \
  --ood-batch-size 67 \
  --epochs 45
```

The runner requires at least two unique seeds for aggregate statistics, so the smallest supported verification uses seeds 0 and 1.

## 4. Run the complete five-seed replication

```bash
python harness/run_multiseed.py \
  --config harness/config_tw.yaml \
  --output reproduced-five-seed \
  --seeds 0,1,2,3,4 \
  --collection-seed 42 \
  --train-episodes 3520 \
  --train-batch-size 16 \
  --id-batch-size 20 \
  --ood-batch-size 67 \
  --epochs 45
```

Do not change the batch sizes. ALFWorld's asynchronous evaluator requires a batch size that divides the selected split size. Use 20 for 140 ID games and 67 for 134 OOD games.

## 5. Check the result

Compare the generated aggregate and per-seed values with `EXPECTED_SCORES.json`. A clean recollection can vary slightly because the environment collection and policy initialization are repeated. Report the five-seed mean and sample standard deviation.

The recorded reference result is:

| Split | Macro success, mean | Sample SD | Micro successes by seed |
|---|---:|---:|---|
| ID | 93.46% | 1.13% | 132/140, 134/140, 133/140, 134/140, 130/140 |
| OOD | 90.87% | 0.97% | 122/134, 121/134, 122/134, 121/134, 124/134 |

The evaluation is valid only when the expert interface and all evaluation feedback remain unavailable to the policy until scoring.
