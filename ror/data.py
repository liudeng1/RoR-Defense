"""WikiText-2 loading and token window sampling."""
from pathlib import Path
import random

import torch
from ror.logging_utils import logger


def resolve_wikitext_source(dataset_root):
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    legacy = root / 'wikitext' if (root / 'wikitext').is_dir() else root
    if (legacy / 'wikitext.py').is_file():
        return 'legacy', legacy
    for path in (root, root / 'wikitext-2-raw-v1', root / 'wikitext' / 'wikitext-2-raw-v1'):
        if path.is_dir() and all(list(path.glob(f'{s}-*.parquet')) for s in ('train', 'test')):
            return 'parquet', path
    raise FileNotFoundError(f'WikiText-2 not found in {root}')


def _windows(text, tokenizer, count, seqlen, seed):
    span = max(32768, seqlen * 16)
    if len(text) <= span:
        raise ValueError('WikiText-2 split is too short')
    maximum = len(text) - span
    rng = random.Random(seed)
    windows = []
    starts = []
    for i in range(count):
        left = i * (maximum + 1) // count
        right = max(left, min((i + 1) * (maximum + 1) // count - 1, maximum))
        start = rng.randint(left, right)
        if start and not text[start - 1].isspace():
            for pos in range(start, min(len(text), start + 128)):
                if text[pos].isspace():
                    start = min(pos + 1, maximum)
                    break
        starts.append(start)
        length = span
        while True:
            chunk = text[start:min(len(text), start + length)]
            ids = tokenizer(chunk, return_tensors='pt', add_special_tokens=False).input_ids
            if ids.shape[1] >= seqlen:
                windows.append(ids[:, :seqlen].contiguous())
                break
            if start + length >= len(text):
                raise ValueError(f'Insufficient tokens near character {start}')
            length *= 2
    return windows, starts


def load_wikitext2(dataset_root, model_path, *, nsamples=128, eval_nsamples=8,
                   seqlen=2048, seed=0, eval_seed=0):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    kind, path = resolve_wikitext_source(dataset_root)
    logger.info('WikiText-2: %s (%s)', kind, path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True, local_files_only=True)

    def load(split):
        if kind == 'legacy':
            return load_dataset(str(path), 'wikitext-2-raw-v1', split=split, trust_remote_code=True)
        files = sorted(str(f) for f in path.glob(f'{split}-*.parquet'))
        return load_dataset('parquet', data_files={split: files}, split=split)

    train_text = '\n\n'.join(load('train')['text'])
    test_text = '\n\n'.join(load('test')['text'])
    train, train_starts = _windows(train_text, tokenizer, nsamples, seqlen, seed)
    test, test_starts = _windows(test_text, tokenizer, eval_nsamples, seqlen, eval_seed)
    logger.info('Train starts: %s; test starts: %s', train_starts[:5], test_starts[:5])
    calib = []
    for x in train:
        target = x.clone()
        target[:, :-1] = -100
        calib.append((x, target))
    return tokenizer, calib, torch.cat(test, dim=1)
