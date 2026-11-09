# Rotated Robustness (RoR)

Core implementation and single-bit fault demo for [*Rotated Robustness: A Training-Free Defense against Bit-Flip Attacks on Large Language Models*](https://arxiv.org/abs/2603.16382) (IEEE TIFS, accepted).

The code implements activation-channel sensitivity profiling, hierarchical orthogonal rotation (`Q = PBD`), offline weight fusion, per-output-channel INT8 quantization, and bit-flip injection. The demo reports WikiText-2 perplexity (PPL) and generated text. It does not include the paper's full attack evaluation or optimized inference kernels.

## Results

Measured on November 9, 2026 using an NVIDIA RTX 5090, WikiText-2 raw, 128 calibration windows and 8 test windows (2048 tokens per window, seed 0).

| Model | Baseline clean | Baseline with fault | RoR clean | RoR with fault |
|---|---:|---:|---:|---:|
| OPT-125M | 25.1726 | 2481.5306 | 25.1812 | 25.1849 |
| Qwen2.5-7B | 6.1321 | 426339.0109 | 6.1242 | 6.1286 |

Both models were tested with a bit-7 flip at the same **INT8 index** in each deployed representation:

| Model | Weight index | RoR β |
|---|---|---:|
| OPT-125M | `model.decoder.layers.2.fc1.weight[643,706]` | 0.15 |
| Qwen2.5-7B | `model.layers.1.mlp.gate_proj.weight[17094,1923]` | 0.30 |

RoR uses α = 6 for both models. The two deployments have different quantized weights, so flipping the same bit index does not reproduce the same real-valued perturbation. Defense-aware SPoF reconstruction is a separate experiment in the paper.

**Qwen2.5-7B**, prompt: `Question: What is the capital of France?\nAnswer:`

```text
Baseline, clean:       paris
Baseline, bit flipped: 11111111111900000000000000000000
RoR, clean:            paris
RoR, bit flipped:      paris
```

**OPT-125M**, prompt: `The capital of France is`

```text
Baseline, clean:       the capital of the French Republic.
Baseline, bit flipped: the,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
RoR, clean:            the capital of the French Republic.
RoR, bit flipped:      the capital of the French Republic.
```

These continuations illustrate the effect of the selected fault. OPT-125M is a base language model, so the examples measure preservation of its normal completion rather than question-answering accuracy. Complete outputs are in [`examples/`](examples/).

## Installation

Linux, an NVIDIA GPU, and a compatible driver are required. The following versions match the author's reference environment: Python 3.10.19, PyTorch 2.10.0+cu130, Transformers 4.57.6, and Datasets 2.19.2.

```bash
conda env create -f environment.yml
conda activate RoR
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Use `run_demo.sh` to launch the demo. It prioritizes cuBLASLt from the active PyTorch environment to avoid mixing it with a system CUDA installation.

## Models and dataset

Model weights and WikiText-2 are not included. Download them from [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), [OPT-125M](https://huggingface.co/facebook/opt-125m), and [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext). Pass their local paths to the demo.

`--model` must point to a Hugging Face checkpoint directory. `--dataset-root` accepts a local WikiText-2 folder containing either the original `wikitext.py` dataset loader (with `datasets==2.19.2`) or the `wikitext-2-raw-v1` Parquet files.

To download the Parquet dataset:

```bash
hf download Salesforce/wikitext --repo-type dataset \
  --include 'wikitext-2-raw-v1/*.parquet' \
  --local-dir /path/to/wikitext
```

## Run

Qwen2.5-7B:

```bash
GPU=0 bash run_demo.sh --model-key qwen7b \
  --model /path/to/Qwen2.5-7B \
  --dataset-root /path/to/wikitext \
  --question 'What is the capital of France?'
```

OPT-125M:

```bash
GPU=0 bash run_demo.sh --model-key opt125m \
  --model /path/to/opt-125m \
  --dataset-root /path/to/wikitext \
  --prompt 'The capital of France is'
```

`GPU` selects a physical GPU. You can add more `--question` or `--prompt` arguments, use `--interactive` for terminal input, or use `--no-ppl` for text-only output.

The first run profiles activations and stores the result under `cache/`; later runs reuse it. Use `--rebuild-profile` if you change the calibration data or model checkpoint. Each run writes `results.json` and `comparison.txt` under `results/<timestamp>/`.

The demo simulates INT8-deployed weights by dequantizing them to BF16/FP16 for matrix multiplication. It is intended to demonstrate correctness and fault sensitivity, not to benchmark INT8 GEMM latency.

## Citation

See [`CITATION.cff`](CITATION.cff) or cite the [paper](https://arxiv.org/abs/2603.16382).
