#!/usr/bin/env python3
"""Single-bit fault demonstration for RoR."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import gc
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ror.data import load_wikitext2, resolve_wikitext_source
from ror.logging_utils import logger
from ror.plan import load_or_build_activation_profile
from ror.quant import dequantize_row, quantize_all_linears_
from ror.rotation import inject_ror
from ror.spof import MODEL_PRESETS, flip_one_int8_bit

ROOT = Path(__file__).resolve().parent
PROMPTS = {
    'qwen7b': ['Question: What is the capital of France?\nAnswer:',
               'The largest planet in our solar system is'],
    'opt125m': ['The capital of France is',
                'The largest planet in our solar system is'],
}


def parse_args():
    p = argparse.ArgumentParser(description='RoR single-bit fault demo')
    p.add_argument('--model-key', choices=MODEL_PRESETS, default='qwen7b')
    p.add_argument('--model', required=True, help='Local Hugging Face model directory')
    p.add_argument('--dataset-root', required=True, help='Local WikiText-2 directory')
    p.add_argument('--question', action='append', help='Question to answer; may be repeated')
    p.add_argument('--prompt', action='append', help='Text completion; may be repeated')
    p.add_argument('--interactive', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--nsamples', type=int, default=128)
    p.add_argument('--eval-nsamples', type=int, default=8)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-seed', type=int, default=0)
    p.add_argument('--profile-path', default=None)
    p.add_argument('--rebuild-profile', action='store_true')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--no-ppl', action='store_true')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_model(path, model_type, device):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    if config.model_type != model_type:
        raise ValueError(f'Expected model_type={model_type}, got {config.model_type}')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info('Loading %s (%s)', path, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, dtype=dtype, device_map=None, low_cpu_mem_usage=True,
        local_files_only=True, trust_remote_code=True,
    )
    return model.to(device).eval()


def get_target(model, spec):
    layer = model.get_submodule(spec.module_path)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f'Expected Linear at {spec.module_path}')
    if not (0 <= spec.row < layer.out_features and 0 <= spec.col < layer.in_features):
        raise IndexError(f'Fault coordinate is out of range for {spec.module_path}')
    return layer


def perplexity(model, token_ids, seqlen, n, device):
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for i in range(n):
            ids = token_ids[:, i * seqlen:(i + 1) * seqlen].to(device)
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1, :]
            if not torch.isfinite(logits).all():
                return None
            for start in range(0, seqlen - 1, 256):
                end = min(start + 256, seqlen - 1)
                block = logits[:, start:end].float()
                loss = F.cross_entropy(block.reshape(-1, block.shape[-1]),
                                       ids[:, 1 + start:1 + end].reshape(-1),
                                       reduction='sum')
                if not torch.isfinite(loss):
                    return None
                loss_sum += loss.item()
                token_count += end - start
            logger.info('PPL %d/%d', i + 1, n)
    result = math.exp(loss_sum / token_count)
    return result if math.isfinite(result) else None


def generate(model, tokenizer, prompts, max_new_tokens, device):
    answers = []
    for prompt in prompts:
        batch = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        try:
            with torch.inference_mode():
                output = model.generate(
                    **batch, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id, use_cache=True,
                )
            text = tokenizer.decode(output[0, batch['input_ids'].size(1):],
                                    skip_special_tokens=True)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            text = f'[generation failed: {exc}]'
        answers.append(text)
    return answers


def evaluate(model, tokenizer, eval_ids, prompts, args):
    return {
        'ppl': None if args.no_ppl else perplexity(
            model, eval_ids, args.seqlen, args.eval_nsamples, 'cuda:0'),
        'answers': generate(model, tokenizer, prompts, args.max_new_tokens, 'cuda:0'),
    }


def flip_and_evaluate(model, row, spec, tokenizer, eval_ids, prompts, args):
    module = get_target(model, spec)
    corrupted, before, after = flip_one_int8_bit(row.q, col=spec.col, bit=spec.bit)
    with torch.no_grad():
        module.weight[spec.row].copy_(dequantize_row(
            corrupted, row.scale, module.weight.dtype, module.weight.device))
    logger.info('INT8 bit %d: %d -> %d (scale %.9g)', spec.bit, before, after, row.scale)
    try:
        result = evaluate(model, tokenizer, eval_ids, prompts, args)
    finally:
        with torch.no_grad():
            module.weight[spec.row].copy_(dequantize_row(
                row.q, row.scale, module.weight.dtype, module.weight.device))
    result['fault'] = {'before': before, 'after': after, 'scale': row.scale}
    return result


def questions(args):
    chosen = [f'Question: {q.strip()}\nAnswer:' for q in args.question or [] if q.strip()]
    chosen.extend(args.prompt or [])
    if args.interactive:
        print('Enter questions (empty line to finish):')
        while True:
            try:
                line = input('> ').strip()
            except EOFError:
                break
            if not line:
                break
            chosen.append(f'Question: {line}\nAnswer:')
    return chosen or PROMPTS[args.model_key]


def report(result, prompts):
    lines = [f"Model: {result['model']}", f"Fault: {result['fault']}",
             'Same INT8 coordinate in each deployed weight representation.',
             '', 'WikiText-2 perplexity:']
    states = [('Baseline clean', result['baseline']['clean']),
              ('Baseline bit flip', result['baseline']['fault']),
              ('RoR clean', result['ror']['clean']),
              ('RoR bit flip', result['ror']['fault'])]
    for name, value in states:
        lines.append(f"  {name:<18} {value['ppl'] if value['ppl'] is not None else 'N/A'}")
    for j, prompt in enumerate(prompts):
        lines.extend(['', f'Prompt: {prompt}'])
        for name, value in states:
            lines.append(f"  {name}: {value['answers'][j]!r}")
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    model_path = str(Path(args.model).expanduser().resolve())
    dataset_root = str(Path(args.dataset_root).expanduser().resolve())
    if not (Path(model_path) / 'config.json').is_file():
        raise FileNotFoundError(f'Model not found: {model_path}')
    resolve_wikitext_source(dataset_root)
    if min(args.nsamples, args.eval_nsamples, args.seqlen - 1) < 1:
        raise ValueError('Invalid calibration/evaluation sample count or sequence length')
    preset = MODEL_PRESETS[args.model_key]
    spec = preset['spof']
    prompts = questions(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir or ROOT / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.profile_path) if args.profile_path else ROOT / 'cache' / (
        f'{args.model_key}_seed{args.seed}_n{args.nsamples}_s{args.seqlen}.pt')
    result = {
        'model': preset['name'],
        'fault': f'{spec.module_path}.weight[{spec.row},{spec.col}], bit {spec.bit}',
        'parameters': {'alpha': 6, 'beta': preset['beta'], 'seed': args.seed,
                       'calibration_windows': args.nsamples,
                       'evaluation_windows': args.eval_nsamples, 'seqlen': args.seqlen},
        'prompts': prompts,
    }

    logger.info('Loading WikiText-2')
    tokenizer, calib_loader, eval_ids = load_wikitext2(
        dataset_root, model_path, nsamples=args.nsamples,
        eval_nsamples=args.eval_nsamples, seqlen=args.seqlen,
        seed=args.seed, eval_seed=args.eval_seed)
    logger.info('Baseline')
    baseline = load_model(model_path, preset['model_type'], 'cuda:0')
    profile, reused = load_or_build_activation_profile(
        baseline, calib_loader, torch.device('cuda:0'), path=profile_path,
        nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed,
        model_path=model_path, dataset=dataset_root,
        force_rebuild=args.rebuild_profile)
    result['profile_reused'] = reused
    row, count = quantize_all_linears_(baseline, target_module=spec.module_path, target_row=spec.row)
    result['baseline'] = {
        'clean': evaluate(baseline, tokenizer, eval_ids, prompts, args),
        'fault': flip_and_evaluate(baseline, row, spec, tokenizer, eval_ids, prompts, args),
    }
    del baseline, row
    gc.collect()
    torch.cuda.empty_cache()

    logger.info('RoR')
    rotated = load_model(model_path, preset['model_type'], 'cuda:0')
    rotated, _ = inject_ror(rotated, activation_profile=profile,
                            alpha=6.0, beta=preset['beta'], seed=args.seed)
    layer = get_target(rotated, spec)
    plan = layer.alpha_beta_plan
    position = plan.ranking.index(spec.col)
    result['sensitivity_rank'] = position + 1
    result['mixing_depth'] = plan.leaf_depths[position]
    row_ror, count_ror = quantize_all_linears_(
        rotated, target_module=spec.module_path, target_row=spec.row)
    if count != count_ror:
        raise RuntimeError('Protected layer counts differ')
    result['ror'] = {
        'clean': evaluate(rotated, tokenizer, eval_ids, prompts, args),
        'fault': flip_and_evaluate(rotated, row_ror, spec, tokenizer, eval_ids, prompts, args),
    }
    result_path = output_dir / 'results.json'
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = report(result, prompts)
    (output_dir / 'comparison.txt').write_text(summary, encoding='utf-8')
    print('\n' + summary)
    print(f'Results: {output_dir}')


if __name__ == '__main__':
    main()
