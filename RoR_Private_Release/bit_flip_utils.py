# bit_flip_utils.py - 存放所有辅助函数

import torch
import numpy as np
import time
import random
import pickle
import os
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
from custom_logger import logger
# from gptq.biterror1.ttt_7_18_all_value_flip import Error_injection_opt_cpu_gpu
from modelutils import *
import gc
import shutil
from biterror1.ttt_7_18_all_value_flip import Error_injection, Error_injection_fewbit, compare_tensor_bits, Error_injection_random_and_fix, Error_injection_opt, Error_injection_opt_cpu_gpu
import sys
import time
import pandas as pd
from tqdm import tqdm
dev = DEV
import json
from typing import List, Tuple, Dict, Any
import torch.nn as nn

# ================= 配置区域 =================
# 设置一个全局开关，或者你可以通过 argparse 从外部传入
# True: 尝试寻找并Hook自定义层； False: 只处理标准的 nn.Linear
ENABLE_CUSTOM_MODULES = False
# ===========================================

# --- 1. 尝试导入或本地定义自定义类 ---
try:
    # 尝试从 transformers 导入 (兼容你以前修改过源码的情况)
    from transformers.models.llama.modeling_llama import MatmulModule, IdentityModule

    logger.info("✅ 成功从 transformers 源码中检测到自定义类 (MatmulModule/IdentityModule)")
except ImportError:
    # 如果是官方库，没有这些类，我们就在本地定义它们作为占位符
    # 这样 isinstance 检查不会报错，代码也能跑通
    if ENABLE_CUSTOM_MODULES:
        logger.warning("⚠️ 未在 transformers 中找到自定义类，使用本地定义的包装类。")


    class MatmulModule(nn.Module):
        def forward(self, x): return x  # 默认行为，视具体需求修改


    class IdentityModule(nn.Module):
        def forward(self, x): return x

# from transformers.models.llama.modeling_llama import MatmulModule, IdentityModule  # 自己定义的一个类，必须要引用源代码
import glob

import torch
import torch.nn as nn
import pandas as pd
import os
from tqdm import tqdm
from custom_logger import logger
import math



def create_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

# 内存监控函数
def log_memory(dev):
    from custom_logger import logger
    import gc
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"CPU Memory: {mem_info.rss / 1024 ** 3:.2f} GB")
    if torch.cuda.is_available():
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f} GB")
        logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved(dev) / 1024 ** 3:.2f} GB")

def set_random_seed(seed=None, saved_file=None):
    """
    Set the global random seed for reproducibility. Optionally append the seed to a saved_file.

    Args:
        seed: The seed value to use (default: None).
              If None, a new seed is generated.
        saved_file: Path to a file to append the seed value (default: None).
                    If None, the seed is not saved to a history file.

    Returns:
        int: The seed value used.
    """
    import torch
    import numpy as np
    import random
    import os

    # 如果未提供种子，生成一个新种子
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"No seed provided, generated new random seed {seed}")
    else:
        logger.info(f"Using provided random seed {seed}")

    # 设置全局随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 如果提供了 saved_file，追加种子到 saved_file
    if saved_file:
        os.makedirs(os.path.dirname(saved_file), exist_ok=True)
        with open(saved_file, 'a') as f:
            f.write(f"{seed}\n")  # 追加一行种子值
        logger.info(f"Appended random seed {seed} to history file {saved_file}")

    return seed

# 统一定义到quant.py里面
from quant import real_quantize, fake_quantize
def real_quantize_2_3(x, scale, wbits, dev):
    """量化权重为整数值（对称量化，无零点），优化内存使用"""
    """量化权重为整数值，支持 BF16/FP16"""
    maxq = 2 ** (wbits - 1) - 1
    if maxq < 0:
        logger.error("maxq < 0")
        raise ValueError("maxq must be non-negative")

    dtype = x.dtype
    x = x.to(dev)
    scale = scale.to(dev, dtype=dtype)
    # print(f"x.shape = {x.shape}")
    # print(f"scale.shape = {scale.shape}")
    # 使用原地操作减少中间张量
    x.div_(scale)  # 原地除法
    x.round_()  # 原地取整
    x.clamp_(-maxq, maxq)  # 原地裁剪
    W_q = x.to(torch.int16)

    # 释放输入张量
    del x

    return W_q

def print_all_layer_info(layers):
    # Get submodules using find_layers
    logger.info(f"打印所有 层.子层")
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)
        for name, module in subset.items():
            logger.info(f"layer{layer_idx}.{name} = {module.weight.data.shape}")

# 推荐定义一个辅助函数
def is_llama_like(model):
    return model.config.model_type in ["llama", "qwen", "qwen2", "mistral"]

def get_opt(model, model_config):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

def get_llama(model, model_config):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

def get_qwen(model, model_config):
    import torch
    # 1. 跳过随机初始化的优化 (和 get_llama 一致)
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # 2. 直接导入 Qwen2 的模型类
    # 注意：Qwen2.5 同样使用 Qwen2ForCausalLM 这个类名
    from transformers import Qwen2ForCausalLM

    # 3. 显式调用类方法加载模型
    model = Qwen2ForCausalLM.from_pretrained(
        model,
        torch_dtype=model_config.get("torch_dtype", "auto"),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        trust_remote_code=True,
        device_map=None
    )

    # 4. 设置评估用的序列长度
    model.seqlen = 2048
    return model

def get_model(model, model_config):
    model_type = model_config["model_type"]
    if model_type == 'opt':
        model = get_opt(model, model_config)
    elif model_type == 'llama':
        model = get_llama(model, model_config)
    elif model_type == 'qwen': # 新增 Qwen 支持
        model = get_qwen(model, model_config)
    return model

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    # 【修复】使用 isinstance 允许子类 (RotatedLinear 是 nn.Linear 的子类)
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def loss_inference(model, last_layer_outs, nsamples, testenc, dev, model_type="opt"):
    """
    计算模型在给定输入上的损失和困惑度。
    增强版：
    1. 自动处理 NaN/Inf (返回大数值而不崩溃)。
    2. 支持 OPT 和 LLaMA 架构。
    """
    dtype = next(model.parameters()).dtype
    # 1. 准备模型组件
    model.lm_head = model.lm_head.to(dev)

    # 移动归一化层到设备
    if model_type == "opt":
        if hasattr(model.model, 'decoder'):
            if getattr(model.model.decoder, 'final_layer_norm', None):
                model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
            if getattr(model.model.decoder, 'project_out', None):
                model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    elif is_llama_like(model):
        if getattr(model.model, 'norm', None):
            model.model.norm = model.model.norm.to(dev)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    testenc = testenc.to(dev)
    seqlen = model.seqlen

    losses = []
    nlls = []
    last_layer_outs = last_layer_outs.to(dtype=dtype, device=dev)
    # 2. 遍历样本计算
    for i in range(nsamples):
        # 获取当前样本的 hidden_states: [1, seqlen, hidden_size]
        hidden_states = last_layer_outs[i].unsqueeze(0).to(dev)

        # --- 鲁棒性检查: NaN/Inf 熔断 ---
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            logger.warning(f"[LossInfer] Sample {i}: Hidden states containing NaN/Inf! Model is broken.")
            # 返回惩罚性大数值，视为模型崩溃
            # 使用一个足够大的数，例如 100.0 Loss (PPL ~ e^100)
            losses.append(torch.tensor(100.0, device=dev))
            nlls.append(torch.tensor(100.0 * (seqlen - 1), device=dev))
            continue

        # 3. 前向传播 (Apply Final Norm & Head)
        try:
            # Apply Norm
            if model_type == "opt":
                if getattr(model.model.decoder, 'final_layer_norm', None):
                    hidden_states = model.model.decoder.final_layer_norm(hidden_states)
                if getattr(model.model.decoder, 'project_out', None):
                    hidden_states = model.model.decoder.project_out(hidden_states)
            elif is_llama_like(model):
                if getattr(model.model, 'norm', None):
                    hidden_states = model.model.norm(hidden_states)

            # Apply Head
            lm_logits = model.lm_head(hidden_states)  # [1, seqlen, vocab_size]

            # 再次检查 Logits 是否炸了
            if torch.isnan(lm_logits).any() or torch.isinf(lm_logits).any():
                raise RuntimeError("Logits containing NaN/Inf")

            # 4. 计算 Loss
            shift_logits = lm_logits[:, :-1, :].contiguous()

            # 正确切片 Label: 第 i 个样本对应的 token
            # testenc 形状假设是 [1, total_len] 或 [nsamples, seqlen]
            # 这里兼容这两种情况
            if testenc.dim() == 2 and testenc.size(0) == 1:
                # 长序列切片模式
                sample_tokens = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            else:
                # Batch 模式
                sample_tokens = testenc[i:i + 1, :]

            shift_labels = sample_tokens[:, 1:].to(dev)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses.append(loss)
            nlls.append(loss * (seqlen - 1))

        except RuntimeError as e:
            # 捕获计算过程中的溢出错误
            logger.warning(f"[LossInfer] Sample {i} calculation failed: {e}")
            losses.append(torch.tensor(100.0, device=dev))
            nlls.append(torch.tensor(100.0 * (seqlen - 1), device=dev))

    # 5. 汇总结果
    # 即使所有样本都炸了，我们也返回平均惩罚值，而不是报错
    if not losses:
        # 极罕见情况（nsamples=0）
        return float('nan'), float('nan')

    avg_loss = torch.stack(losses).mean()

    # PPL 计算: exp( sum(NLL) / total_tokens )
    total_tokens = nsamples * (seqlen - 1)
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)

    return avg_loss, ppl

# 为了兼容旧代码，保留 compute_loss 接口，但直接转调 loss_inference
def compute_loss(model, last_layer_outs, testenc, dev, seqlen, nsamples, model_type="opt"):
    """
    兼容层：直接调用 loss_inference。
    """
    # 确保 last_layer_outs 在 GPU 上 (如果显存允许)
    # 或者 loss_inference 内部会处理 device
    avg_loss, ppl = loss_inference(model, last_layer_outs, nsamples, testenc, dev, model_type)

    # 转换为 python float 返回
    return avg_loss.item(), ppl.item()

def forward_layer(layer, input, attention_mask, position_ids=None):
    """前向传播单个层"""
    if position_ids is not None:
        return layer(input, attention_mask=attention_mask, position_ids=position_ids)[0]
    return layer(input, attention_mask=attention_mask)[0]

def load_outlier_config(csv_path):
    """
    读取 scan_outliers_layerwise 生成的 csv 文件。
    返回: { "layer_name": [index_list] }
    """
    if not csv_path or not os.path.exists(csv_path):
        logger.warning(f"⚠️ [Adaptive Attack] 未找到异常列配置文件: {csv_path}，将执行标准攻击。")
        return {}

    df = pd.read_csv(csv_path)
    protected_map = {}

    for _, row in df.iterrows():
        # scan 函数保存的 Layer 格式通常是 "0.self_attn.q_proj"
        layer_name = str(row['Layer']).strip()
        indices_str = str(row['Outlier_Indices'])

        indices = []
        # 处理可能的不同分隔符
        if ";" in indices_str:
            indices = [int(x) for x in indices_str.split(";") if x.strip()]
        elif "," in indices_str:
            indices = [int(x) for x in indices_str.split(",") if x.strip()]
        else:
            try:
                indices = [int(indices_str)]
            except:
                pass

        if indices:
            protected_map[layer_name] = indices

    logger.info(f"🛡️ [Adaptive Attack] 已加载保护配置，覆盖 {len(protected_map)} 个层。这些层的 Outlier 将被屏蔽。")
    return protected_map


def scan_outliers_layerwise0(model, testenc, dev, seqlen=2048, nsamples=128,
                            filename="outlier_config.csv", model_kwargs=None):
    logger.info(f">>> [Profiler] 启动极速分层扫描 (Layer-wise, nsamples={nsamples})...")

    # 1. 初始化 model_kwargs
    if model_kwargs is None: model_kwargs = {}
    local_kwargs = model_kwargs.copy()

    # 2. H200 专属优化：全量 GPU
    # 不需要 modules_to_gpu 这种复杂的移来移去逻辑
    # 只要保证模型已经在 dev 上即可 (opt.py 里已经 model.to(dev) 了)

    # 获取数据类型
    dtype = next(model.parameters()).dtype

    # 3. 准备 Input
    # 直接在 GPU 上初始化，省去搬运
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError

    # 4. 运行 Catcher
    layers = get_model_layers(model)
    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)  # 这一步需要 Embedding 在 GPU 上才能运行！
        except ValueError:
            pass

    layers[0] = layers[0].module

    # 5. 处理 Mask 和 Position IDs (这就是你关心的部分)
    if cache['attention_mask'] is not None:
        local_kwargs['attention_mask'] = cache['attention_mask'].to(dev)

    # 优先使用 Catcher 抓到的，抓不到就自己生成 (完全不影响 PPL)
    if cache['position_ids'] is not None:
        local_kwargs['position_ids'] = cache['position_ids'].to(dev)
    elif 'position_ids' not in local_kwargs and is_llama_like(model):
        local_kwargs['position_ids'] = torch.arange(seqlen, dtype=torch.long, device=dev).unsqueeze(0)

    # 5. 扫描循环
    layer_stats = []
    current_inputs = inps  # 保持在 GPU

    for i, layer in enumerate(tqdm(layers, desc="Scanning Layers")):
        layer = layer.to(dev)

        # 注册 Hook
        subset = find_layers(layer, layers=[nn.Linear])
        layer_local_maxes = {}

        def stat_hook(name):
            def hook(module, inp, out):
                x = inp[0].detach()
                if x.dim() == 3:
                    col_max = x.abs().amax(dim=(0, 1))
                else:
                    col_max = x.abs().flatten().max()

                if name not in layer_local_maxes:
                    layer_local_maxes[name] = col_max
                else:
                    torch.maximum(layer_local_maxes[name], col_max, out=layer_local_maxes[name])

            return hook

        handles = [m.register_forward_hook(stat_hook(n)) for n, m in subset.items()]

        # 推理
        layer_output_list = []
        for j in range(nsamples):
            inp_j = current_inputs[j].unsqueeze(0)  # [1, seqlen, hidden]

            with torch.no_grad():
                # 【核心】直接传入合并后的字典 (含 RoPE, Mask, PID)
                out_j = layer(inp_j, **local_kwargs)[0]

            layer_output_list.append(out_j.squeeze(0))

        for h in handles: h.remove()

        # 统计逻辑 (保持不变)
        for name, max_val in layer_local_maxes.items():
            val_float = max_val.float()
            mean, std = val_float.mean().item(), val_float.std().item()
            thresh = max(mean + 6 * std, max(1.0, mean * 2.0))
            outliers = (val_float > thresh).nonzero(as_tuple=True)[0]

            if len(outliers) > 0:
                layer_stats.append({
                    "Layer": f"{i}.{name}",
                    "Outlier_Indices": ";".join(map(str, outliers.cpu().numpy())),
                    "Count": len(outliers)
                })

        current_inputs = torch.stack(layer_output_list)
        # H200 显存足够，不需要移回 CPU，不需要 empty_cache

    df = pd.DataFrame(layer_stats)
    df.to_csv(filename, index=False)
    return filename

def scan_outliers_layerwise(model, testenc, dev, seqlen=2048, nsamples=128,
                            filename="outlier_config.csv", model_kwargs=None, sigma_thresh=6.0, rel_thresh=2.0, abs_thresh=1.0):
    logger.info(f">>> [Profiler] 启动极速分层扫描 (Layer-wise, nsamples={nsamples})...")

    # ================= [调用新函数] =================
    # 直接在 GPU 上获取 Input 和 Kwargs
    inps, captured_kwargs = prepare_model_inputs(model, testenc, dev, seqlen, nsamples)

    # 合并传入的 kwargs
    if model_kwargs is None: model_kwargs = {}
    local_kwargs = model_kwargs.copy()
    local_kwargs.update(captured_kwargs)

    # 5. 扫描循环
    layers = get_model_layers(model)

    # 5. 扫描循环
    layer_stats = []
    current_inputs = inps  # 保持在 GPU

    for i, layer in enumerate(tqdm(layers, desc="Scanning Layers")):
        layer = layer.to(dev)

        # 注册 Hook
        subset = find_layers(layer, layers=[nn.Linear])
        layer_local_maxes = {}

        def stat_hook(name):
            def hook(module, inp, out):
                x = inp[0].detach()

                if x.dim() >= 2:
                    # view(-1, hidden_dim) -> 强行拉平为 2D [Total_Tokens, Hidden]
                    # amax(dim=0) -> 求每一列的最大值
                    col_max = x.view(-1, x.shape[-1]).abs().amax(dim=0)
                else:
                    # 极罕见的 1D 情况
                    col_max = x.abs()

                # if x.dim() == 3:
                #     col_max = x.abs().amax(dim=(0, 1))
                # else:
                #     logger.error(f"处理不了不是3d的输入x")
                #     # col_max = x.abs().flatten().max()

                if name not in layer_local_maxes:
                    layer_local_maxes[name] = col_max
                else:
                    torch.maximum(layer_local_maxes[name], col_max, out=layer_local_maxes[name])

            return hook

        handles = [m.register_forward_hook(stat_hook(n)) for n, m in subset.items()]

        # 推理
        layer_output_list = []
        for j in range(nsamples):
            inp_j = current_inputs[j].unsqueeze(0)  # [1, seqlen, hidden]

            with torch.no_grad():
                # 【核心】直接传入合并后的字典 (含 RoPE, Mask, PID)
                out_j = layer(inp_j, **local_kwargs)[0]

            layer_output_list.append(out_j.squeeze(0))

        for h in handles: h.remove()

        # 统计逻辑 (保持不变)
        for name, max_val in layer_local_maxes.items():
            val_float = max_val.float()
            mean, std = val_float.mean().item(), val_float.std().item()
            # thresh = max(mean + 6 * std, max(1.0, mean * 2.0))
            thresh = max(mean + sigma_thresh * std, max(abs_thresh, mean * rel_thresh))
            outliers = (val_float > thresh).nonzero(as_tuple=True)[0]

            if len(outliers) > 0:
                layer_stats.append({
                    "Layer": f"{i}.{name}",
                    "Outlier_Indices": ";".join(map(str, outliers.cpu().numpy())),
                    "Count": len(outliers)
                })

        current_inputs = torch.stack(layer_output_list)
        # H200 显存足够，不需要移回 CPU，不需要 empty_cache

    df = pd.DataFrame(layer_stats)
    df.to_csv(filename, index=False)
    return filename



def scan_outliers_layerwise_add_grad(model, testenc, dev, seqlen=2048, nsamples=128,
                            filename="outlier_config.csv", model_kwargs=None,
                            sigma_thresh=6.0, rel_thresh=2.0, abs_thresh=1.0,
                            grad_topk=128):  # [新增参数] 梯度敏感列的数量
    """
    [Profiler V2] 分层扫描：结合激活值离群点 + 梯度敏感度
    """
    logger.info(f">>> [Profiler] 启动极速分层扫描 (Layer-wise, nsamples={nsamples})...")
    logger.info(f"    策略: Activation (Sigma={sigma_thresh}) + Gradient (Top-{grad_topk})")

    # ================= [准备输入] =================
    # 直接在 GPU 上获取 Input 和 Kwargs
    # 假设 prepare_model_inputs 返回的是 (Tensor[nsamples, seqlen, hidden], kwargs)
    inps, captured_kwargs = prepare_model_inputs(model, testenc, dev, seqlen, nsamples)

    # 合并传入的 kwargs
    if model_kwargs is None: model_kwargs = {}
    local_kwargs = model_kwargs.copy()
    local_kwargs.update(captured_kwargs)

    # 获取所有层
    if hasattr(model, 'model'):
        layers = model.model.layers if hasattr(model.model, 'layers') else model.model.decoder.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Could not find layers list in model")

    # 结果容器
    layer_stats = []

    # 初始输入 (保持在 GPU)
    current_inputs = inps

    # ================= [扫描循环] =================
    for i, layer in enumerate(tqdm(layers, desc="Scanning Layers")):
        layer = layer.to(dev)

        # 1. 找到该层所有的 Linear 模块
        subset = find_layers(layer, layers=[nn.Linear])

        # 容器：存储每一层的统计信息
        # act_max_records: {name: tensor([hidden_dim])} -> 记录激活值最大值
        # grad_sum_records: {name: tensor([hidden_dim])} -> 记录梯度累积值
        act_max_records = {}
        grad_sum_records = {}

        # ================= [Hook 定义] =================

        # A. Forward Hook: 统计激活值 Outlier
        def get_act_hook(name):
            def hook(module, inp, out):
                x = inp[0].detach()  # 激活值不需要梯度

                # 处理维度: [Batch, Seq, Hidden] -> [Hidden]
                if x.dim() >= 2:
                    # 求每一列(Feature Dim)的最大绝对值
                    col_max = x.view(-1, x.shape[-1]).abs().amax(dim=0)
                else:
                    col_max = x.abs()

                if name not in act_max_records:
                    act_max_records[name] = col_max
                else:
                    torch.maximum(act_max_records[name], col_max, out=act_max_records[name])

            return hook

        # B. Backward Hook: 统计梯度敏感度
        # 我们需要统计: Loss 对 输入(Input) 的梯度 还是 对 权重(Weight) 的梯度？
        # RoR 保护的是 Input Channel，对应权重的列。
        # 所以我们需要 Weight.grad 的列范数。
        # 由于 PyTorch backward hook 对 weight 处理比较麻烦，我们采用
        # "开启参数梯度 -> forward -> backward -> 读 .grad" 的显式方式更稳健。

        # 注册 Forward Hooks
        handles = []
        for name, module in subset.items():
            handles.append(module.register_forward_hook(get_act_hook(name)))
            # [关键] 开启当前层 Linear 权重的梯度计算
            module.weight.requires_grad = True

            # ================= [推理与反向传播] =================
        layer_output_list = []

        # 为了计算梯度，我们需要一个 Loss 目标。
        # 无监督情况下，通常用 Output 的 L2 Norm 或 Sum 作为伪 Loss，
        # 或者假设 Output 自身就是下一层的 Input，最大化其激活值（Hessian 近似）。
        # 最简单有效的方法：Loss = Output.norm() 或 Output[0].sum()
        # 这里我们用 Output.float().norm() 作为“敏感度探测器”。

        # 临时开启梯度模式 (即使外层是 no_grad)
        with torch.enable_grad():
            for j in range(nsamples):
                # 准备输入
                # 注意：current_inputs[j] 可能是 detach 的，需要 requires_grad=True 吗？
                # 不需要，我们只关心 weight 的梯度。
                inp_j = current_inputs[j].unsqueeze(0).detach()  # [1, seqlen, hidden]
                inp_j.requires_grad = True  # 让 Input 也可以传导梯度（虽然主要看 Weight）

                # Forward
                # 注意：这里不能用 **local_kwargs 直接传，因为可能会有 buffer 导致 graph 问题
                # 但对于 Layer-wise 扫描，通常没问题。
                out_j = layer(inp_j, **local_kwargs)[0]

                # 收集 Output 用于下一层 (detach 掉，下一层不需要反传回这一层)
                layer_output_list.append(out_j.detach().squeeze(0))

                # Backward (计算敏感度)
                # 伪 Loss: 我们希望找到对 Output 变化影响最大的权重
                # Loss = out_j.sum() 或 out_j.norm()
                pseudo_loss = out_j.float().norm()
                pseudo_loss.backward()

                # 累积梯度信息
                for name, module in subset.items():
                    if module.weight.grad is not None:
                        # module.weight.shape = [Out_Features, In_Features]
                        # 我们关心每一列 (In_Features) 的敏感度
                        # Metric: L1 Norm of gradients for each column
                        # dim=0 表示沿着 Out_Features 求和，得到 [In_Features]
                        grad_col_norm = module.weight.grad.detach().abs().sum(dim=0)

                        if name not in grad_sum_records:
                            grad_sum_records[name] = grad_col_norm
                        else:
                            grad_sum_records[name] += grad_col_norm

                        # 清空梯度，防止累积到下一轮
                        module.weight.grad = None

        # 清理 Hook 和 梯度状态
        for h in handles: h.remove()
        for name, module in subset.items():
            module.weight.requires_grad = False  # 关掉梯度，节省显存

        # ================= [统计与合并] =================
        for name, module in subset.items():
            # 1. 处理 Activation Outliers (原有逻辑)
            if name in act_max_records:
                val_float = act_max_records[name].float()
                mean, std = val_float.mean().item(), val_float.std().item()
                thresh = max(mean + sigma_thresh * std, max(abs_thresh, mean * rel_thresh))
                act_outliers = (val_float > thresh).nonzero(as_tuple=True)[0]
            else:
                act_outliers = torch.tensor([], device=dev)

            # 2. 处理 Gradient Outliers (新增逻辑)
            if name in grad_sum_records:
                grad_scores = grad_sum_records[name].float()
                # 选 Top-K 敏感列
                # 注意：如果层很小，topk 不能超过维度
                k = min(grad_topk, grad_scores.numel())
                _, grad_outliers = torch.topk(grad_scores, k)
            else:
                grad_outliers = torch.tensor([], device=dev)

            # 3. 取并集 (Union)
            # unique() 会自动排序并去重
            final_outliers = torch.cat([act_outliers, grad_outliers]).unique()
            final_outliers = final_outliers.sort()[0]  # 确保有序

            if len(final_outliers) > 0:
                # 记录来源用于 Debug (可选)
                # logger.info(f"Layer {i}.{name}: Act={len(act_outliers)}, Grad={len(grad_outliers)}, Union={len(final_outliers)}")

                layer_stats.append({
                    "Layer": f"{i}.{name}",
                    "Outlier_Indices": ";".join(map(str, final_outliers.cpu().numpy())),
                    "Count": len(final_outliers)
                })

        # 准备下一层的输入
        current_inputs = torch.stack(layer_output_list)

        # 显存清理 (可选，H200可能不需要，但为了稳健)
        # torch.cuda.empty_cache()

    # ================= [保存结果] =================
    df = pd.DataFrame(layer_stats)
    df.to_csv(filename, index=False)
    logger.info(f">>> [Profiler] 扫描完成。结果已保存至 {filename}")
    return filename



def scan_outliers_layerwise_L1_abssum(model, testenc, dev, seqlen=2048, nsamples=128,
                            filename="outlier_config.csv", model_kwargs=None,
                            sigma_thresh=6.0, rel_thresh=2.0, abs_thresh=1.0):
    """
    [L1 Norm / Sum 版本] 分层扫描 Outlier
    修改说明：
    1. 统计指标从 Max (峰值) 改为 Sum (总能量/L1范数)。
    2. 聚合方式从 Maximum (取最大) 改为 Accumulate (累加)。
    """
    logger.info(f">>> [Profiler] 启动极速分层扫描 (L1-Norm/Sum Mode, nsamples={nsamples})...")

    # ================= [准备输入] =================
    # 直接在 GPU 上获取 Input 和 Kwargs
    inps, captured_kwargs = prepare_model_inputs(model, testenc, dev, seqlen, nsamples)

    # 合并传入的 kwargs
    if model_kwargs is None: model_kwargs = {}
    local_kwargs = model_kwargs.copy()
    local_kwargs.update(captured_kwargs)

    # 获取层列表
    layers = get_model_layers(model)

    # ================= [扫描循环] =================
    layer_stats = []
    current_inputs = inps  # 保持在 GPU

    for i, layer in enumerate(tqdm(layers, desc="Scanning Layers")):
        layer = layer.to(dev)

        # 注册 Hook
        subset = find_layers(layer, layers=[nn.Linear])
        layer_local_metrics = {}  # [修改] 重命名变量，存储累加值

        def stat_hook(name):
            def hook(module, inp, out):
                x = inp[0].detach()

                # [核心修改 1]: 计算 Sum (L1 Norm) 而不是 Max
                if x.dim() == 3:
                    # Sum over Batch(0) and Sequence(1)
                    col_metric = x.abs().sum(dim=(0, 1))
                else:
                    logger.error(f"处理不了不是3d的输入x")
                    return
                    # col_metric = x.abs().flatten().sum() # 如果想兼容2D，用这个

                # [核心修改 2]: 累加 (Accumulate) 而不是取最大
                if name not in layer_local_metrics:
                    layer_local_metrics[name] = col_metric
                else:
                    layer_local_metrics[name] += col_metric  # In-place addition

            return hook

        handles = [m.register_forward_hook(stat_hook(n)) for n, m in subset.items()]

        # 推理 (Forward Pass)
        layer_output_list = []
        for j in range(nsamples):
            inp_j = current_inputs[j].unsqueeze(0)  # [1, seqlen, hidden]

            with torch.no_grad():
                out_j = layer(inp_j, **local_kwargs)[0]

            layer_output_list.append(out_j.squeeze(0))

        for h in handles: h.remove()

        # 统计分析
        for name, metric_val in layer_local_metrics.items():
            val_float = metric_val.float()

            # 计算均值和方差
            mean, std = val_float.mean().item(), val_float.std().item()

            # 计算阈值 (逻辑保持不变，但数值量级会变大)
            # 注意：由于现在是 Sum，数值很大，abs_thresh (1.0) 基本失效，主要靠 mean * rel_thresh 起作用
            thresh = max(mean + sigma_thresh * std, max(abs_thresh, mean * rel_thresh))

            outliers = (val_float > thresh).nonzero(as_tuple=True)[0]

            if len(outliers) > 0:
                layer_stats.append({
                    "Layer": f"{i}.{name}",
                    "Outlier_Indices": ";".join(map(str, outliers.cpu().numpy())),
                    "Count": len(outliers)
                })

        current_inputs = torch.stack(layer_output_list)
        # H200 显存足够，不需要移回 CPU

    # 保存结果
    df = pd.DataFrame(layer_stats)
    df.to_csv(filename, index=False)
    return filename


def scan_outliers_layerwise_L1_sumabs(model, testenc, dev, seqlen=2048, nsamples=128,
                            filename="outlier_config_abs_sum.csv", model_kwargs=None,
                            sigma_thresh=6.0, rel_thresh=2.0, abs_thresh=1.0):
    """
    [Abs(Sum) 版本] 分层扫描 Outlier
    逻辑：先累加所有激活值 (允许正负抵消)，最后取绝对值。
    公式：| Sum(x) |
    """
    logger.info(f">>> [Profiler] 启动极速分层扫描 (Abs-of-Sum Mode, nsamples={nsamples})...")

    # ================= [准备输入] =================
    inps, captured_kwargs = prepare_model_inputs(model, testenc, dev, seqlen, nsamples)

    if model_kwargs is None: model_kwargs = {}
    local_kwargs = model_kwargs.copy()
    local_kwargs.update(captured_kwargs)

    layers = get_model_layers(model)

    # ================= [扫描循环] =================
    layer_stats = []
    current_inputs = inps

    for i, layer in enumerate(tqdm(layers, desc="Scanning Layers")):
        layer = layer.to(dev)

        subset = find_layers(layer, layers=[nn.Linear])
        layer_local_metrics = {}

        def stat_hook(name):
            def hook(module, inp, out):
                x = inp[0].detach()

                # [核心修改 1]: 计算带符号的和 (Signed Sum)
                # 不做 abs()，允许正负值在求和时互相抵消
                if x.dim() == 3:
                    col_signed_sum = x.sum(dim=(0, 1))
                else:
                    # 兼容 2D 输入
                    col_signed_sum = x.sum(dim=0)

                # [核心修改 2]: 累加带符号的值
                if name not in layer_local_metrics:
                    layer_local_metrics[name] = col_signed_sum
                else:
                    layer_local_metrics[name] += col_signed_sum

            return hook

        handles = [m.register_forward_hook(stat_hook(n)) for n, m in subset.items()]

        # 推理
        layer_output_list = []
        for j in range(nsamples):
            inp_j = current_inputs[j].unsqueeze(0)

            with torch.no_grad():
                out_j = layer(inp_j, **local_kwargs)[0]

            layer_output_list.append(out_j.squeeze(0))

        for h in handles: h.remove()

        # [核心修改 3]: 结算时取绝对值
        # 此时 metric_val 存储的是整个数据集上该列的净值 (Net Value)
        for name, signed_sum_val in layer_local_metrics.items():
            # 在这里取绝对值！
            val_float = signed_sum_val.float().abs()

            # 后续统计逻辑不变
            mean, std = val_float.mean().item(), val_float.std().item()

            # 注意：因为是 Sum，数值很大，abs_thresh 建议忽略或调大
            thresh = max(mean + sigma_thresh * std, max(abs_thresh, mean * rel_thresh))

            outliers = (val_float > thresh).nonzero(as_tuple=True)[0]

            if len(outliers) > 0:
                layer_stats.append({
                    "Layer": f"{i}.{name}",
                    "Outlier_Indices": ";".join(map(str, outliers.cpu().numpy())),
                    "Count": len(outliers)
                })

        current_inputs = torch.stack(layer_output_list)

    df = pd.DataFrame(layer_stats)
    df.to_csv(filename, index=False)
    return filename


def scan_outliers_layerwise_average(model, testenc, dev, seqlen=2048, nsamples=128,
                            filename="outlier_config_simple.csv", model_kwargs=None,
                            threshold_ratio=6.0, rel_thresh=2.0, abs_thresh=1.0):  # <--- 【核心修改】只有一个参数了
    """
    [极简模式] 分层扫描 Outlier
    指标：列的和的绝对值 (Abs of Sum)
    阈值：全层平均值的 N 倍 (Ratio * Layer_Mean)
    """
    logger.info(f">>> [Profiler] 启动极简扫描 (Ratio={threshold_ratio}, Mode=Abs(Sum))...")

    # ================= [准备输入] =================
    inps, captured_kwargs = prepare_model_inputs(model, testenc, dev, seqlen, nsamples)

    if model_kwargs is None: model_kwargs = {}
    local_kwargs = model_kwargs.copy()
    local_kwargs.update(captured_kwargs)

    layers = get_model_layers(model)

    # ================= [扫描循环] =================
    layer_stats = []
    current_inputs = inps

    for i, layer in enumerate(tqdm(layers, desc="Scanning Layers")):
        layer = layer.to(dev)
        subset = find_layers(layer, layers=[nn.Linear])

        # 存储累加值 (带符号，允许正负抵消)
        layer_raw_sums = {}

        def stat_hook(name):
            def hook(module, inp, out):
                x = inp[0].detach()

                # 1. 计算带符号的和 (Signed Sum)
                if x.dim() == 3:
                    col_sum = x.sum(dim=(0, 1))
                else:
                    col_sum = x.sum(dim=0)

                # 2. 累加到历史数据中
                if name not in layer_raw_sums:
                    layer_raw_sums[name] = col_sum
                else:
                    layer_raw_sums[name] += col_sum

            return hook

        handles = [m.register_forward_hook(stat_hook(n)) for n, m in subset.items()]

        # 推理
        layer_output_list = []
        for j in range(nsamples):
            inp_j = current_inputs[j].unsqueeze(0)
            with torch.no_grad():
                out_j = layer(inp_j, **local_kwargs)[0]
            layer_output_list.append(out_j.squeeze(0))

        for h in handles: h.remove()

        # ================= [核心筛选逻辑] =================
        for name, raw_sum in layer_raw_sums.items():
            # 1. 这一步才取绝对值：|Sum|
            # 这代表了这一列在所有数据上的“净偏差” (Net Bias)
            col_magnitudes = raw_sum.float().abs()

            # 2. 计算整个层的平均水平 (Global Mean of this layer)
            # 这就是“矩阵总均值绝对值”
            layer_mean = col_magnitudes.mean().item()

            # 3. 唯一的阈值：均值的 Ratio 倍
            thresh = layer_mean * threshold_ratio

            # 4. 筛选
            outliers = (col_magnitudes > thresh).nonzero(as_tuple=True)[0]

            if len(outliers) > 0:
                layer_stats.append({
                    "Layer": f"{i}.{name}",
                    "Outlier_Indices": ";".join(map(str, outliers.cpu().numpy())),
                    "Count": len(outliers),
                    "Layer_Mean": f"{layer_mean:.2f}"  # 记录一下均值方便分析
                })

        current_inputs = torch.stack(layer_output_list)

    df = pd.DataFrame(layer_stats)
    df.to_csv(filename, index=False)
    return filename

def scan_layer_importance_by_global_grad(model, testenc, dev, seqlen=2048, nsamples=32, chunk_size=4, model_type='opt',
                                         filename="global_grad_importance.csv"):
    """
    [基于全局梯度的层重要性扫描 - H200 优化版]

    原理:
    运行 End-to-End 的反向传播，计算每一层权重的梯度。
    为了防止 Batch 内样本梯度符号相反导致的抵消 (Gradient Masking)，
    我们采用 "累加绝对值梯度" 的策略。

    统计指标:
    Importance = Sum( |W| * E[|G|] )
    其中 E[|G|] 是 nsamples 个样本梯度的平均绝对值。
    """
    logger.info(f">>> [Profiler] 启动全局梯度层重要性扫描 (Abs Gradient, nsamples={nsamples})...")

    # 1. 准备工作
    model.eval()  # 保持 eval 模式 (Dropout 关闭)
    model.zero_grad()

    # [H200优化] 确保整个模型在 GPU 上
    # 相比于分层扫描，全局扫描必须要求整个模型都在显存里才能 backward
    try:
        model = model.to(dev)
    except RuntimeError as e:
        logger.warning(f"模型太大无法全量移至 GPU，可能导致 OOM: {e}")

    # # 显式确保 Embedding 在 GPU (防止某些奇怪的模型加载策略留在 CPU)
    # if model_type == "opt":
    #     if hasattr(model.model.decoder, "embed_tokens"):
    #         model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    #     if hasattr(model.model.decoder, "embed_positions"):
    #         model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    #     if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
    #         model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    #     if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
    #         model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    # elif is_llama_like(model):
    #     if hasattr(model.model, "embed_tokens"):
    #         model.model.embed_tokens = model.model.embed_tokens.to(dev)
    #     if hasattr(model.model, "rotary_emb"):
    #         model.model.rotary_emb = model.model.rotary_emb.to(dev)

    # 开启梯度检查点以节省显存 (可选，H200 显存够大其实可以不开，开了会慢一点但更稳)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 获取层列表
    if model_type == "opt":
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    num_layers = len(layers)

    layer_stats = []

    # =======================================================
    # 外层循环: 按块 (Chunk) 遍历层
    # 目的: 显存优化。每次只打开一小部分层的 requires_grad，
    # 虽然是全模型 Backward，但只计算并存储这几层的梯度。
    # =======================================================
    for start_idx in range(0, num_layers, chunk_size):
        end_idx = min(start_idx + chunk_size, num_layers)
        target_layers_range = range(start_idx, end_idx)
        logger.info(f"Analysis Chunk: Layers {start_idx}-{end_idx} / {num_layers}")

        # 1. 冻结所有参数
        model.requires_grad_(False)

        # 2. 只解冻当前块内的 Linear 层
        active_modules = {}
        for i in target_layers_range:
            layer = layers[i]
            subset = find_layers(layer, layers=[nn.Linear])
            for name, module in subset.items():
                full_name = f"{i}.{name}"
                active_modules[full_name] = module
                module.weight.requires_grad_(True)  # 开启梯度

        if not active_modules:
            continue

        # 3. 准备累加器 (在 GPU 上)
        # 存储 sum(|g|)，初始化为 0
        grad_abs_sum_accum = {k: torch.zeros_like(m.weight, dtype=torch.float32, device=dev) for k, m in
                              active_modules.items()}

        # =======================================================
        # 内层循环: 逐样本计算 (Accumulate Gradients)
        # =======================================================
        # 限制单次最大 seqlen 防止 OOM
        current_seqlen = min(seqlen, 2048)

        for i in range(nsamples):
            model.zero_grad()  # 清空上一轮的梯度

            # 取一个样本
            # 注意：这里我们不需要手动传 mask/rope，因为直接调用的 model()，它自己会算
            batch = testenc[:, (i * seqlen):((i * seqlen) + current_seqlen)].to(dev)

            try:
                # Forward
                outputs = model(batch, labels=batch)
                loss = outputs.loss

                # Backward
                loss.backward()
            except RuntimeError as e:
                logger.error(f"Forward/Backward 失败 (Idx={i}): {e}")
                # 如果单个样本失败，跳过，不中断整个流程
                model.zero_grad()
                del batch
                continue

            # 累加梯度绝对值
            for full_name, module in active_modules.items():
                if module.weight.grad is not None:
                    # 【核心修正】: 取绝对值后再累加!
                    # 避免 Sample A 的正梯度和 Sample B 的负梯度抵消
                    g_abs = module.weight.grad.detach().abs().to(torch.float32)
                    grad_abs_sum_accum[full_name] += g_abs

            # 清理单次循环显存
            del outputs, loss, batch
            # 这里的 zero_grad 很重要，防止梯度在 PyTorch 内部默认累加
            model.zero_grad()

            # =======================================================
        # 计算该块的层重要性指标
        # =======================================================
        for full_name, module in active_modules.items():
            # 计算 E[|g|]
            avg_abs_grad = grad_abs_sum_accum[full_name] / nsamples

            # 获取权重 |W|
            w = module.weight.data.to(torch.float32).abs()

            # 核心指标: Saliency = Sum( |W| * E[|G|] )
            # 物理含义: 权重幅值大 且 梯度幅值大 的层，最重要
            importance_score = torch.sum(w * avg_abs_grad).item()

            # 辅助指标: 纯梯度的 L1 范数 (不考虑权重大小)
            grad_l1_norm = torch.sum(avg_abs_grad).item()

            layer_stats.append({
                "Layer": full_name,
                "Importance_Score": importance_score,
                "Grad_L1_Norm": grad_l1_norm
            })

            # 释放累加器显存
            del grad_abs_sum_accum[full_name]
            module.weight.grad = None  # 彻底释放


    # 4. 恢复状态 & 保存结果
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.zero_grad()
    model.requires_grad_(False)  # 恢复全冻结

    # 按 Importance 排序
    df = pd.DataFrame(layer_stats)
    if not df.empty:
        df = df.sort_values(by="Importance_Score", ascending=False)

        # 自动创建目录
        create_dir(filename)
        df.to_csv(filename, index=False)

        logger.info(f">>> [Profiler] 全局梯度扫描完成! 结果保存在: {filename}")
        logger.info(f"    Top 5 最关键层:\n{df.head(5)}")
    else:
        logger.warning(">>> [Profiler] 未收集到任何层信息，请检查模型结构或数据。")

    return filename

def progressive_bit_search_opt_1_29(model, layers, testenc, dev, wbits, quantizers, nsamples=16, attack_iters=10,
                               chunk_size=4, output_csv="attack_logs_PBS.csv"):
    """
    [PBS 修复版] Progressive Bit Search
    修复点：
    1. 修正了权重恢复逻辑：确保备份的是“初次修改前”的原始值，避免多次攻击同一权重时的状态覆盖。
    2. 只有当权重从未被备份过时，才进行备份。
    """
    import copy
    import pandas as pd

    logger.info(f">>> [PBS] 启动 PBS 搜索 (Iters={attack_iters}, Bits={wbits}, Mode=OffsetBinary)...")

    model.eval()
    model.zero_grad()

    attack_logs = []

    # [修复] 备份字典：Key = "Layer|Name|FlatIdx" (不包含 Bit!)
    modified_weights_backup = {}

    visited_bits = set()
    offset = 2 ** (wbits - 1)

    for iter_step in range(attack_iters):
        logger.info(f"--- PBS Iteration {iter_step + 1}/{attack_iters} ---")

        global_best_score = -1.0
        global_target = None

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        num_layers = len(layers)

        # === 搜索阶段 (保持不变) ===
        for start_idx in range(0, num_layers, chunk_size):
            end_idx = min(start_idx + chunk_size, num_layers)

            model.requires_grad_(False)
            active_params = []
            current_chunk_meta = []

            for i in range(start_idx, end_idx):
                layer = layers[i]
                subset = find_layers(layer, layers=[nn.Linear])

                for name, mod in subset.items():
                    q_key_under = f"{i}_{name}"
                    q_key_dot = f"{i}.{name}"
                    found_q_key = q_key_under if q_key_under in quantizers else (
                        q_key_dot if q_key_dot in quantizers else None)

                    if found_q_key:
                        mod.weight.requires_grad_(True)
                        active_params.append(mod.weight)
                        current_chunk_meta.append({
                            'layer_idx': i,
                            'sub_name': name,
                            'module': mod,
                            'quant_key': found_q_key
                        })

            if not active_params: continue

            model.zero_grad()
            batch = testenc[:, :min(testenc.shape[1], 512)].to(dev)
            try:
                loss = model(batch, labels=batch).loss
                loss.backward()
            except RuntimeError as e:
                logger.error(f"PBS OOM at chunk {start_idx}: {e}")
                model.zero_grad()
                torch.cuda.empty_cache()
                continue

            for meta in current_chunk_meta:
                mod = meta['module']
                if mod.weight.grad is None: continue

                quantizer = quantizers[meta['quant_key']]
                scale = quantizer.scale.to(dev)
                w = mod.weight.data
                g = mod.weight.grad.data
                w_int_signed = (w / scale).round().to(torch.int64)
                w_int_unsigned = w_int_signed + offset
                g_int = g * scale

                for b in range(wbits):
                    position_val = 2 ** b
                    current_bit = (w_int_unsigned >> b) & 1
                    valid_mask = ((current_bit == 0) & (g_int > 0)) | ((current_bit == 1) & (g_int < 0))
                    score_mat = g_int.abs() * position_val
                    score_mat = torch.where(valid_mask, score_mat, torch.tensor(-1.0, device=dev))

                    while True:
                        max_s, idx = score_mat.view(-1).max(0)
                        max_s = max_s.item()
                        if max_s <= global_best_score: break

                        flat_idx = idx.item()
                        # visit_key 用于判断是否攻击过该位置的该比特
                        visit_key = f"{meta['layer_idx']}|{meta['sub_name']}|{flat_idx}|{b}"

                        if visit_key in visited_bits:
                            score_mat.view(-1)[idx] = -1.0
                            continue
                        else:
                            global_best_score = max_s
                            grad_val = g_int.view(-1)[flat_idx].item()
                            row_idx = flat_idx // mod.weight.shape[1]
                            col_idx = flat_idx % mod.weight.shape[1]

                            global_target = {
                                'layer_idx': meta['layer_idx'],
                                'sub_name': meta['sub_name'],
                                'flat_idx': flat_idx,
                                'coords': [row_idx, col_idx],
                                'bit': b,
                                'score': max_s,
                                'direction': 1 if grad_val > 0 else -1,
                                'scale': scale.view(-1)[flat_idx % scale.nelement()].item(),
                                'visit_key': visit_key
                            }
                            break

            model.zero_grad()
            for p in active_params: p.requires_grad_(False)
            del loss, batch
            torch.cuda.empty_cache()

        # === 攻击执行阶段 ===
        if global_target is None:
            logger.warning(f"PBS Iter {iter_step + 1}: 搜索结束 (无有效目标)。")
            break

        visited_bits.add(global_target['visit_key'])
        log_layer_name = f"{global_target['layer_idx']}_{global_target['sub_name']}"

        attack_logs.append([
            iter_step + 1, log_layer_name, global_target['coords'], global_target['bit'], global_target['score']
        ])

        # 物理翻转
        layer = layers[global_target['layer_idx']]
        subset = find_layers(layer, layers=[nn.Linear])
        mod = subset[global_target['sub_name']]

        # --- [关键修复] 备份逻辑 ---
        # 仅使用 Layer|Name|FlatIdx 作为备份键 (去掉 Bit)
        backup_key_unique = f"{global_target['layer_idx']}|{global_target['sub_name']}|{global_target['flat_idx']}"

        # 只有当该位置从未被备份过时，才保存。这保证了保存的是 Iter 0 的原始值。
        if backup_key_unique not in modified_weights_backup:
            curr_w = mod.weight.data.view(-1)[global_target['flat_idx']].item()
            modified_weights_backup[backup_key_unique] = {
                'layer_idx': global_target['layer_idx'],
                'sub_name': global_target['sub_name'],
                'flat_idx': global_target['flat_idx'],
                'val': curr_w  # 这是原始值
            }
        # -------------------------

        # 施加扰动
        perturbation = global_target['direction'] * (2 ** global_target['bit']) * global_target['scale']
        with torch.no_grad():
            mod.weight.view(-1)[global_target['flat_idx']] += perturbation

        logger.info(
            f"PBS Iter {iter_step + 1}: Attacked {log_layer_name} | Bit {global_target['bit']} | Score {global_target['score']:.2f}")

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # 保存 CSV
    csv_data = [{"Iter": l[0], "Layer": l[1], "Coords": str(l[2]), "Bit": l[3], "Score": l[4]} for l in attack_logs]
    pd.DataFrame(csv_data).to_csv(output_csv, index=False)
    logger.info(f"PBS 攻击日志已保存至: {output_csv}")

    # --- [关键修复] 恢复逻辑 ---
    logger.info("PBS 搜索结束，正在恢复模型原始参数...")
    with torch.no_grad():
        for key, info in modified_weights_backup.items():
            layer = layers[info['layer_idx']]
            subset = find_layers(layer, layers=[nn.Linear])
            mod = subset[info['sub_name']]
            # 无论这个位置被叠加修改了多少次，直接重置回原始值
            mod.weight.view(-1)[info['flat_idx']] = info['val']

    logger.info("PBS 模型恢复完成。")

    return attack_logs


def progressive_bit_search_opt(model, layers, testenc, dev, wbits, quantizers, nsamples, seqlen, attack_iters=10,
                               chunk_size=4, output_csv="attack_logs_PBS.csv"):
    """
    [PBS 极速版] Progressive Bit Search (Vectorized)
    特点：
    1. 移除了 'for bit in range(wbits)' 循环，利用广播一次性计算所有 bit 的分数。
    2. 大幅减少 .item() 同步调用次数，适配高性能 GPU (如 H200)。
    """
    import copy
    import pandas as pd

    # 确保不越界
    total_tokens_needed = nsamples * seqlen
    if testenc.shape[1] < total_tokens_needed:
        logger.warning(f"测试集长度 ({testenc.shape[1]}) 不足以提供 {nsamples}*{seqlen} tokens，将自动缩减 nsamples。")
        actual_tokens = testenc.shape[1]
        # 强制截断到 seqlen 的整数倍，防止 reshape 报错
        nsamples = actual_tokens // seqlen
        pbs_seq_len = nsamples * seqlen
    else:
        pbs_seq_len = total_tokens_needed

    logger.info(f">>> [PBS] 启动 PBS 搜索 (Iters={attack_iters}, Bits={wbits}, pbs_seq_len={pbs_seq_len}, Mode=OffsetBinary-Vectorized) ...")

    model.eval()
    model.zero_grad()

    attack_logs = []
    # 备份字典：Key = "Layer|Name|FlatIdx"
    modified_weights_backup = {}

    # 记录已攻击位置: set("Layer|Name|FlatIdx|Bit")
    visited_bits = set()

    offset = 2 ** (wbits - 1)

    # 预计算比特位的掩码和权重 [Bits] -> [1, Bits]
    # 例如 8bit: [1, 2, 4, ..., 128]
    bit_range = torch.arange(wbits, device=dev)
    powers_of_2 = 2 ** bit_range

    for iter_step in range(attack_iters):
        logger.info(f"--- PBS Iteration {iter_step + 1}/{attack_iters} ---")

        global_best_score = -1.0
        global_target = None

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        num_layers = len(layers)

        # === 搜索阶段 ===
        for start_idx in range(0, num_layers, chunk_size):
            end_idx = min(start_idx + chunk_size, num_layers)

            # 1. 准备梯度
            model.requires_grad_(False)
            active_params = []
            current_chunk_meta = []

            for i in range(start_idx, end_idx):
                layer = layers[i]
                subset = find_layers(layer, layers=[nn.Linear])

                for name, mod in subset.items():
                    q_key_under = f"{i}_{name}"
                    q_key_dot = f"{i}.{name}"
                    found_q_key = q_key_under if q_key_under in quantizers else (
                        q_key_dot if q_key_dot in quantizers else None)

                    if found_q_key:
                        mod.weight.requires_grad_(True)
                        active_params.append(mod.weight)
                        current_chunk_meta.append({
                            'layer_idx': i,
                            'sub_name': name,
                            'module': mod,
                            'quant_key': found_q_key
                        })

            if not active_params: continue

            model.zero_grad()
            # 限制 batch 大小防止 OOM，通常 PBS 不需要全量数据
            # [核心修正] 提取数据并 Reshape 为 [Batch, Seq]
            # 1. 提取
            flat_input = testenc[:, :pbs_seq_len].to(dev)
            # 2. 重塑: [1, N*L] -> [N, L]
            # 这样做的目的是让模型并行处理 N 个样本，且每个样本长度不超过 seqlen
            # 避免超过模型最大上下文限制
            batch = flat_input.view(nsamples, seqlen)
            try:
                loss = model(batch, labels=batch).loss
                loss.backward()
            except RuntimeError as e:
                logger.error(f"PBS OOM at chunk {start_idx}: {e}")
                model.zero_grad()
                torch.cuda.empty_cache()
                continue

            # 2. 向量化计算分数
            for meta in current_chunk_meta:
                mod = meta['module']
                if mod.weight.grad is None: continue

                quantizer = quantizers[meta['quant_key']]
                scale = quantizer.scale.to(dev)
                if scale.dim() == 1:
                    raise ValueError("Scale cannot be 1D. Please reshape it to (N, 1) before calling real_quantize.")
                w = mod.weight.data
                g = mod.weight.grad.data

                # 量化处理
                # w_int_signed = (w / scale).round().to(torch.int64)
                w_int_signed = real_quantize(w, scale, wbits, dev).to(torch.int64)
                w_int_unsigned = w_int_signed + offset
                g_int = g * scale

                # --- [核心优化] 向量化计算所有 Bit ---
                # w_int_unsigned: [N] -> [N, 1]
                # bit_range: [B] -> [1, B]
                # 结果 w_bits_mat: [N, B] (每个权重的二进制展开)
                w_bits_mat = (w_int_unsigned.view(-1, 1) >> bit_range.view(1, -1)) & 1

                # g_int: [N] -> [N, 1]
                g_expanded = g_int.view(-1, 1)

                # 验证翻转有效性 (0->1 需要 g>0, 1->0 需要 g<0)
                # valid_mask: [N, B]
                valid_mask = ((w_bits_mat == 0) & (g_expanded > 0)) | \
                             ((w_bits_mat == 1) & (g_expanded < 0))

                # 计算分数矩阵: [N, B]
                # score = abs(grad) * 2^bit
                score_mat = g_expanded.abs() * powers_of_2.view(1, -1)

                # 应用 Mask (无效置为 -1)
                score_mat = torch.where(valid_mask, score_mat, torch.tensor(-1.0, device=dev))

                # --- 处理 Visited (去重) ---
                # 因为 visited 是 Python set，无法直接在 GPU tensor 操作中排除
                # 但因为 visited 数量极少 (iter 数)，我们可以用 while loop 剔除极个别的重复
                # 由于这是针对整个 Module [N, B] 做一次 max，冲突概率极低，循环几乎只跑一次

                while True:
                    # 针对整个矩阵求最大值
                    # flat_idx_in_mat 是 [0, N*B - 1] 的索引
                    max_s, flat_idx_in_mat = score_mat.view(-1).max(0)
                    max_s = max_s.item()

                    # 剪枝：如果当前层最大值都比不过全局最优，直接跳过
                    if max_s <= global_best_score:
                        break

                    # 解算坐标
                    flat_idx_in_mat = flat_idx_in_mat.item()
                    num_bits = wbits

                    weight_flat_idx = flat_idx_in_mat // num_bits
                    bit_idx = flat_idx_in_mat % num_bits

                    visit_key = f"{meta['layer_idx']}|{meta['sub_name']}|{weight_flat_idx}|{bit_idx}"

                    if visit_key in visited_bits:
                        # 如果已访问，Mask 掉并重试 (仅重试这一个点，极快)
                        score_mat.view(-1)[flat_idx_in_mat] = -1.0
                        continue
                    else:
                        # 找到新的全局最优
                        global_best_score = max_s

                        row_idx = weight_flat_idx // mod.weight.shape[1]
                        col_idx = weight_flat_idx % mod.weight.shape[1]
                        grad_val = g_expanded[weight_flat_idx, 0].item()
                        target_scale = scale[row_idx, 0].item()
                        global_target = {
                            'layer_idx': meta['layer_idx'],
                            'sub_name': meta['sub_name'],
                            'flat_idx': weight_flat_idx,
                            'coords': [row_idx, col_idx],
                            'bit': bit_idx,
                            'score': max_s,
                            'direction': 1 if grad_val > 0 else -1,
                            'scale': target_scale,
                            'visit_key': visit_key
                        }
                        break

            model.zero_grad()
            for p in active_params: p.requires_grad_(False)
            del loss, batch
            torch.cuda.empty_cache()

        # === 攻击执行阶段 ===
        if global_target is None:
            logger.warning(f"PBS Iter {iter_step + 1}: 搜索结束 (无有效目标)。")
            break

        visited_bits.add(global_target['visit_key'])
        log_layer_name = f"{global_target['layer_idx']}_{global_target['sub_name']}"

        attack_logs.append([
            iter_step + 1, log_layer_name, global_target['coords'], global_target['bit'], global_target['score']
        ])

        # 物理翻转
        layer = layers[global_target['layer_idx']]
        subset = find_layers(layer, layers=[nn.Linear])
        mod = subset[global_target['sub_name']]

        # 备份原始值
        backup_key_unique = f"{global_target['layer_idx']}|{global_target['sub_name']}|{global_target['flat_idx']}"
        if backup_key_unique not in modified_weights_backup:
            curr_w = mod.weight.data.view(-1)[global_target['flat_idx']].item()
            modified_weights_backup[backup_key_unique] = {
                'layer_idx': global_target['layer_idx'],
                'sub_name': global_target['sub_name'],
                'flat_idx': global_target['flat_idx'],
                'val': curr_w
            }

        # 施加扰动
        perturbation = global_target['direction'] * (2 ** global_target['bit']) * global_target['scale']
        with torch.no_grad():
            mod.weight.view(-1)[global_target['flat_idx']] += perturbation

        logger.info(
            f"PBS Iter {iter_step + 1}: Attacked {log_layer_name} | Bit {global_target['bit']} | Score {global_target['score']:.2e}")

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # 保存结果
    csv_data = [{"Iter": l[0], "Layer": l[1], "Coords": str(l[2]), "Bit": l[3], "Score": l[4]} for l in attack_logs]
    pd.DataFrame(csv_data).to_csv(output_csv, index=False)
    logger.info(f"PBS 攻击日志已保存至: {output_csv}")

    # 恢复模型
    logger.info("PBS 搜索结束，正在恢复模型原始参数...")
    with torch.no_grad():
        for key, info in modified_weights_backup.items():
            layer = layers[info['layer_idx']]
            subset = find_layers(layer, layers=[nn.Linear])
            mod = subset[info['sub_name']]
            mod.weight.view(-1)[info['flat_idx']] = info['val']

    logger.info("PBS 模型恢复完成。")
    return attack_logs


def progressive_bit_search_radar_adaptive(model, layers, testenc, dev, wbits, quantizers, nsamples, seqlen,
                                          attack_iters=10, chunk_size=4, output_csv="attack_logs_PBS_Radar.csv",
                                          skip_top_bits=0):
    """
    [PBS Radar 适配版] Progressive Bit Search (Vectorized)
    特点：
    1. 移除了 'for bit in range(wbits)' 循环，利用广播一次性计算所有 bit 的分数。
    2. 大幅减少 .item() 同步调用次数，适配高性能 GPU (如 H200)。
    3. [新增] skip_top_bits: 跳过最高有效的 K 个 bit，用于测试 RADAR 对低位攻击的检测盲区。
    """
    # 确保不越界
    total_tokens_needed = nsamples * seqlen
    if testenc.shape[1] < total_tokens_needed:
        logger.warning(f"测试集长度 ({testenc.shape[1]}) 不足以提供 {nsamples}*{seqlen} tokens，将自动缩减 nsamples。")
        actual_tokens = testenc.shape[1]
        # 强制截断到 seqlen 的整数倍，防止 reshape 报错
        nsamples = actual_tokens // seqlen
        pbs_seq_len = nsamples * seqlen
    else:
        pbs_seq_len = total_tokens_needed

    logger.info(
        f">>> [PBS Radar Adaptive] 启动 PBS 搜索 (Iters={attack_iters}, Bits={wbits}, SkipTop={skip_top_bits}, Mode=Vectorized) ...")

    model.eval()
    model.zero_grad()

    attack_logs = []
    # 备份字典：Key = "Layer|Name|FlatIdx"
    modified_weights_backup = {}

    # 记录已攻击位置: set("Layer|Name|FlatIdx|Bit")
    visited_bits = set()

    offset = 2 ** (wbits - 1)

    # 预计算比特位的掩码和权重 [Bits] -> [1, Bits]
    # 例如 8bit: [1, 2, 4, ..., 128] (实际是 2^0, 2^1... 2^7)
    bit_range = torch.arange(wbits, device=dev)
    powers_of_2 = 2 ** bit_range

    # [新增] 构建 Bit Mask
    # 如果 skip_top_bits = 1 (wbits=8), 则 valid_indices = [0, 1, ..., 6]
    # 对应的 bit_range 是 0..7. 我们需要 Mask 掉 7.
    bit_mask = torch.ones(wbits, device=dev, dtype=torch.bool)
    if skip_top_bits > 0:
        # 屏蔽最高的 skip_top_bits 位
        # 例如 8 bit, skip 1 => bit 7 set False
        # bit_range: [0, 1, 2, 3, 4, 5, 6, 7]
        # mask indices: [7]
        bit_mask[wbits - skip_top_bits:] = False
        logger.info(f"    -> Masked Bits (Ignored): {list(range(wbits - skip_top_bits, wbits))}")

    for iter_step in range(attack_iters):
        logger.info(f"--- PBS Iteration {iter_step + 1}/{attack_iters} ---")

        global_best_score = -1.0
        global_target = None

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        num_layers = len(layers)

        # === 搜索阶段 ===
        for start_idx in range(0, num_layers, chunk_size):
            end_idx = min(start_idx + chunk_size, num_layers)

            # 1. 准备梯度
            model.requires_grad_(False)
            active_params = []
            current_chunk_meta = []

            for i in range(start_idx, end_idx):
                layer = layers[i]
                subset = find_layers(layer, layers=[nn.Linear])

                for name, mod in subset.items():
                    q_key_under = f"{i}_{name}"
                    q_key_dot = f"{i}.{name}"
                    found_q_key = q_key_under if q_key_under in quantizers else (
                        q_key_dot if q_key_dot in quantizers else None)

                    if found_q_key:
                        mod.weight.requires_grad_(True)
                        active_params.append(mod.weight)
                        current_chunk_meta.append({
                            'layer_idx': i,
                            'sub_name': name,
                            'module': mod,
                            'quant_key': found_q_key
                        })

            if not active_params: continue

            model.zero_grad()
            # 限制 batch 大小防止 OOM
            flat_input = testenc[:, :pbs_seq_len].to(dev)
            batch = flat_input.view(nsamples, seqlen)
            try:
                loss = model(batch, labels=batch).loss
                loss.backward()
            except RuntimeError as e:
                logger.error(f"PBS OOM at chunk {start_idx}: {e}")
                model.zero_grad()
                torch.cuda.empty_cache()
                continue

            # 2. 向量化计算分数
            for meta in current_chunk_meta:
                mod = meta['module']
                if mod.weight.grad is None: continue

                quantizer = quantizers[meta['quant_key']]
                scale = quantizer.scale.to(dev)
                if scale.dim() == 1:
                    raise ValueError("Scale cannot be 1D. Please reshape it to (N, 1) before calling real_quantize.")

                w = mod.weight.data
                g = mod.weight.grad.data

                # 量化处理
                w_int_signed = real_quantize(w, scale, wbits, dev).to(torch.int64)
                w_int_unsigned = w_int_signed + offset
                g_int = g * scale

                # --- [核心优化] 向量化计算所有 Bit ---
                # w_int_unsigned: [N] -> [N, 1]
                # bit_range: [B] -> [1, B]
                # 结果 w_bits_mat: [N, B] (每个权重的二进制展开)
                w_bits_mat = (w_int_unsigned.view(-1, 1) >> bit_range.view(1, -1)) & 1

                # g_int: [N] -> [N, 1]
                g_expanded = g_int.view(-1, 1)

                # 验证翻转有效性 (0->1 需要 g>0, 1->0 需要 g<0)
                # valid_mask: [N, B]
                valid_mask = ((w_bits_mat == 0) & (g_expanded > 0)) | \
                             ((w_bits_mat == 1) & (g_expanded < 0))

                # 计算分数矩阵: [N, B]
                # score = abs(grad) * 2^bit
                score_mat = g_expanded.abs() * powers_of_2.view(1, -1)

                # 应用 Mask (无效置为 -1)
                score_mat = torch.where(valid_mask, score_mat, torch.tensor(-1.0, device=dev))

                # [新增] 应用 Bit Mask (跳过高位)
                # bit_mask: [B] -> [1, B] broadcast
                if skip_top_bits > 0:
                    score_mat[:, ~bit_mask] = -1.0

                # --- 处理 Visited (去重) ---
                while True:
                    # 针对整个矩阵求最大值
                    # flat_idx_in_mat 是 [0, N*B - 1] 的索引
                    max_s, flat_idx_in_mat = score_mat.view(-1).max(0)
                    max_s = max_s.item()

                    # 剪枝：如果当前层最大值都比不过全局最优，直接跳过
                    if max_s <= global_best_score:
                        break

                    # 解算坐标
                    flat_idx_in_mat = flat_idx_in_mat.item()
                    num_bits = wbits

                    weight_flat_idx = flat_idx_in_mat // num_bits
                    bit_idx = flat_idx_in_mat % num_bits

                    visit_key = f"{meta['layer_idx']}|{meta['sub_name']}|{weight_flat_idx}|{bit_idx}"

                    if visit_key in visited_bits:
                        # 如果已访问，Mask 掉并重试
                        score_mat.view(-1)[flat_idx_in_mat] = -1.0
                        continue
                    else:
                        # 找到新的全局最优
                        global_best_score = max_s

                        row_idx = weight_flat_idx // mod.weight.shape[1]
                        col_idx = weight_flat_idx % mod.weight.shape[1]
                        grad_val = g_expanded[weight_flat_idx, 0].item()
                        target_scale = scale[row_idx, 0].item()
                        global_target = {
                            'layer_idx': meta['layer_idx'],
                            'sub_name': meta['sub_name'],
                            'flat_idx': weight_flat_idx,
                            'coords': [row_idx, col_idx],
                            'bit': bit_idx,
                            'score': max_s,
                            'direction': 1 if grad_val > 0 else -1,
                            'scale': target_scale,
                            'visit_key': visit_key
                        }
                        break

            model.zero_grad()
            for p in active_params: p.requires_grad_(False)
            del loss, batch
            torch.cuda.empty_cache()

        # === 攻击执行阶段 ===
        if global_target is None:
            logger.warning(f"PBS Iter {iter_step + 1}: 搜索结束 (无有效目标)。")
            break

        visited_bits.add(global_target['visit_key'])
        log_layer_name = f"{global_target['layer_idx']}_{global_target['sub_name']}"

        attack_logs.append([
            iter_step + 1, log_layer_name, global_target['coords'], global_target['bit'], global_target['score']
        ])

        # 物理翻转
        layer = layers[global_target['layer_idx']]
        subset = find_layers(layer, layers=[nn.Linear])
        mod = subset[global_target['sub_name']]

        # 备份原始值
        backup_key_unique = f"{global_target['layer_idx']}|{global_target['sub_name']}|{global_target['flat_idx']}"
        if backup_key_unique not in modified_weights_backup:
            curr_w = mod.weight.data.view(-1)[global_target['flat_idx']].item()
            modified_weights_backup[backup_key_unique] = {
                'layer_idx': global_target['layer_idx'],
                'sub_name': global_target['sub_name'],
                'flat_idx': global_target['flat_idx'],
                'val': curr_w
            }

        # 施加扰动
        perturbation = global_target['direction'] * (2 ** global_target['bit']) * global_target['scale']
        with torch.no_grad():
            mod.weight.view(-1)[global_target['flat_idx']] += perturbation

        logger.info(
            f"PBS Iter {iter_step + 1}: Attacked {log_layer_name} | Bit {global_target['bit']} | Score {global_target['score']:.2e}")

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # 保存结果
    csv_data = [{"Iter": l[0], "Layer": l[1], "Coords": str(l[2]), "Bit": l[3], "Score": l[4]} for l in attack_logs]
    pd.DataFrame(csv_data).to_csv(output_csv, index=False)
    logger.info(f"PBS 攻击日志已保存至: {output_csv}")

    # 恢复模型
    logger.info("PBS 搜索结束，正在恢复模型原始参数...")
    with torch.no_grad():
        for key, info in modified_weights_backup.items():
            layer = layers[info['layer_idx']]
            subset = find_layers(layer, layers=[nn.Linear])
            mod = subset[info['sub_name']]
            mod.weight.view(-1)[info['flat_idx']] = info['val']

    logger.info("PBS 模型恢复完成。")
    return attack_logs


def progressive_bit_search_opt_obstacle(model, layers, testenc, dev, wbits, quantizers, nsamples, seqlen,
                               attack_iters=10, chunk_size=4, output_csv="attack_logs_PBS.csv",
                               outlier_config_path=None):  # <--- [修改1] 新增参数
    """
    [PBS 极速版] 支持 Adaptive Attack (屏蔽 Outlier 列)
    """
    import copy
    import pandas as pd

    # --- [修改2] 加载保护配置 ---
    protected_map = {}
    if outlier_config_path:
        protected_map = load_outlier_config(outlier_config_path)
    # -------------------------

    # 确保不越界 (保持原逻辑)
    total_tokens_needed = nsamples * seqlen
    if testenc.shape[1] < total_tokens_needed:
        logger.warning(f"测试集长度 ({testenc.shape[1]}) 不足以提供 {nsamples}*{seqlen} tokens，将自动缩减 nsamples。")
        actual_tokens = testenc.shape[1]
        nsamples = actual_tokens // seqlen
        pbs_seq_len = nsamples * seqlen
    else:
        pbs_seq_len = total_tokens_needed

    logger.info(f">>> [PBS] 启动 PBS 搜索 (Mode={'Adaptive' if protected_map else 'Standard'})...")

    model.eval()
    model.zero_grad()

    attack_logs = []
    modified_weights_backup = {}
    visited_bits = set()
    offset = 2 ** (wbits - 1)
    bit_range = torch.arange(wbits, device=dev)
    powers_of_2 = 2 ** bit_range

    for iter_step in range(attack_iters):
        logger.info(f"--- PBS Iteration {iter_step + 1}/{attack_iters} ---")

        global_best_score = -1.0
        global_target = None

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        num_layers = len(layers)

        # === 搜索阶段 ===
        for start_idx in range(0, num_layers, chunk_size):
            end_idx = min(start_idx + chunk_size, num_layers)

            # 1. 准备梯度
            model.requires_grad_(False)
            active_params = []
            current_chunk_meta = []

            for i in range(start_idx, end_idx):
                layer = layers[i]
                subset = find_layers(layer, layers=[nn.Linear])

                for name, mod in subset.items():
                    q_key_under = f"{i}_{name}"
                    q_key_dot = f"{i}.{name}"
                    found_q_key = q_key_under if q_key_under in quantizers else (
                        q_key_dot if q_key_dot in quantizers else None)

                    if found_q_key:
                        mod.weight.requires_grad_(True)
                        active_params.append(mod.weight)
                        # 构建与 scan 函数一致的 Key 用于查找配置
                        config_key = f"{i}.{name}"

                        current_chunk_meta.append({
                            'layer_idx': i,
                            'sub_name': name,
                            'module': mod,
                            'quant_key': found_q_key,
                            'config_key': config_key  # 用于查找 protected_map
                        })

            if not active_params: continue

            model.zero_grad()
            flat_input = testenc[:, :pbs_seq_len].to(dev)
            batch = flat_input.view(nsamples, seqlen)

            try:
                loss = model(batch, labels=batch).loss
                loss.backward()
            except RuntimeError as e:
                logger.error(f"PBS OOM at chunk {start_idx}: {e}")
                model.zero_grad()
                torch.cuda.empty_cache()
                continue

            # 2. 向量化计算分数
            for meta in current_chunk_meta:
                mod = meta['module']
                if mod.weight.grad is None: continue

                quantizer = quantizers[meta['quant_key']]
                scale = quantizer.scale.to(dev)
                w = mod.weight.data
                g = mod.weight.grad.data  # [Out, In]

                # =======================================================
                # [修改3] Adaptive Attack: 屏蔽受保护的列 (Mask Gradients)
                # =======================================================
                if meta['config_key'] in protected_map:
                    outlier_indices = protected_map[meta['config_key']]
                    # 转换为 LongTensor
                    idx_tensor = torch.tensor(outlier_indices, device=dev, dtype=torch.long)

                    # 权重形状通常是 [out_features, in_features]
                    # Outlier 是输入通道 (columns)，所以屏蔽对应的列
                    if g.dim() == 2 and g.shape[1] >= idx_tensor.max():
                        # 将这些列的梯度置 0，攻击者就不会选它们了
                        g[:, idx_tensor] = 0.0
                        # logger.debug(f"Layer {meta['config_key']} masked {len(outlier_indices)} columns.")
                # =======================================================

                # 量化处理
                w_int_signed = (w / scale).round().to(torch.int64)
                w_int_unsigned = w_int_signed + offset
                g_int = g * scale

                # --- 向量化计算 ---
                w_bits_mat = (w_int_unsigned.view(-1, 1) >> bit_range.view(1, -1)) & 1
                g_expanded = g_int.view(-1, 1)

                valid_mask = ((w_bits_mat == 0) & (g_expanded > 0)) | \
                             ((w_bits_mat == 1) & (g_expanded < 0))

                score_mat = g_expanded.abs() * powers_of_2.view(1, -1)
                score_mat = torch.where(valid_mask, score_mat, torch.tensor(-1.0, device=dev))

                while True:
                    max_s, flat_idx_in_mat = score_mat.view(-1).max(0)
                    max_s = max_s.item()

                    if max_s <= global_best_score:
                        break

                    flat_idx_in_mat = flat_idx_in_mat.item()
                    num_bits = wbits
                    weight_flat_idx = flat_idx_in_mat // num_bits
                    bit_idx = flat_idx_in_mat % num_bits

                    visit_key = f"{meta['layer_idx']}|{meta['sub_name']}|{weight_flat_idx}|{bit_idx}"

                    if visit_key in visited_bits:
                        score_mat.view(-1)[flat_idx_in_mat] = -1.0
                        continue
                    else:
                        global_best_score = max_s
                        row_idx = weight_flat_idx // mod.weight.shape[1]
                        col_idx = weight_flat_idx % mod.weight.shape[1]
                        grad_val = g_expanded[weight_flat_idx, 0].item()

                        global_target = {
                            'layer_idx': meta['layer_idx'],
                            'sub_name': meta['sub_name'],
                            'flat_idx': weight_flat_idx,
                            'coords': [row_idx, col_idx],
                            'bit': bit_idx,
                            'score': max_s,
                            'direction': 1 if grad_val > 0 else -1,
                            'scale': scale.view(-1)[weight_flat_idx % scale.nelement()].item(),
                            'visit_key': visit_key
                        }
                        break

            model.zero_grad()
            for p in active_params: p.requires_grad_(False)
            del loss, batch
            torch.cuda.empty_cache()

        # === 攻击执行阶段 (保持不变) ===
        if global_target is None:
            logger.warning(f"PBS Iter {iter_step + 1}: 搜索结束 (无有效目标)。")
            break

        visited_bits.add(global_target['visit_key'])
        log_layer_name = f"{global_target['layer_idx']}_{global_target['sub_name']}"

        attack_logs.append([
            iter_step + 1, log_layer_name, global_target['coords'], global_target['bit'], global_target['score']
        ])

        layer = layers[global_target['layer_idx']]
        subset = find_layers(layer, layers=[nn.Linear])
        mod = subset[global_target['sub_name']]

        backup_key_unique = f"{global_target['layer_idx']}|{global_target['sub_name']}|{global_target['flat_idx']}"
        if backup_key_unique not in modified_weights_backup:
            curr_w = mod.weight.data.view(-1)[global_target['flat_idx']].item()
            modified_weights_backup[backup_key_unique] = {
                'layer_idx': global_target['layer_idx'],
                'sub_name': global_target['sub_name'],
                'flat_idx': global_target['flat_idx'],
                'val': curr_w
            }

        perturbation = global_target['direction'] * (2 ** global_target['bit']) * global_target['scale']
        with torch.no_grad():
            mod.weight.view(-1)[global_target['flat_idx']] += perturbation

        logger.info(
            f"PBS Iter {iter_step + 1}: Attacked {log_layer_name} | Bit {global_target['bit']} | Score {global_target['score']:.2e}")

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    csv_data = [{"Iter": l[0], "Layer": l[1], "Coords": str(l[2]), "Bit": l[3], "Score": l[4]} for l in attack_logs]
    pd.DataFrame(csv_data).to_csv(output_csv, index=False)
    logger.info(f"PBS 攻击日志已保存至: {output_csv}")

    logger.info("PBS 搜索结束，正在恢复模型原始参数...")
    with torch.no_grad():
        for key, info in modified_weights_backup.items():
            layer = layers[info['layer_idx']]
            subset = find_layers(layer, layers=[nn.Linear])
            mod = subset[info['sub_name']]
            mod.weight.view(-1)[info['flat_idx']] = info['val']

    logger.info("PBS 模型恢复完成。")
    return attack_logs  # 返回日志以便后续处理


def flip_and_evaluate_fewbit_nodebug(model, layers, quantizers, initial_input, testenc, dev, wbits,
                             seqlen, nsamples, attack_logs, model_kwargs=None, debug_flip=False):
    """
    [Mode 8] PBS 验证函数
    优化：model_kwargs 支持，Dtype 修复
    """
    if model_kwargs is None: model_kwargs = {}

    from biterror1.ttt_7_18_all_value_flip import Error_injection_fewbit
    import ast

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)

    # 1. 建立严格映射
    strict_map = {}
    for i, layer in enumerate(layers):
        subset = find_layers(layer, layers=[nn.Linear])
        for name, mod in subset.items():
            strict_map[f"{i}_{name}"] = mod

    # 2. 解析任务
    flip_tasks = {}
    for log in attack_logs:
        if isinstance(log, dict):
            raw_name, coords_raw, bit = log.get('Layer'), log.get('Coords'), log.get('Bit')
        else:
            raw_name, coords_raw, bit = log[1], log[2], log[3]

        if raw_name not in strict_map:
            raise ValueError(f"Layer {raw_name} not found.")

        coords = ast.literal_eval(coords_raw) if isinstance(coords_raw, str) else coords_raw
        if raw_name not in flip_tasks: flip_tasks[raw_name] = []
        flip_tasks[raw_name].append([coords, int(bit)])

    # 3. 逐层攻击
    for idx in range(len(layers)):
        layer = layers[idx].to(dev)
        subset = find_layers(layer, layers=[nn.Linear])
        layer_backup = {}

        for name, module in subset.items():
            strict_key = f"{idx}_{name}"
            # 兼容 key
            quantizer = quantizers.get(strict_key) or quantizers.get(f"{idx}.{name}")

            if quantizer and strict_key in flip_tasks:
                W = module.weight.data
                layer_backup[name] = W.cpu().clone()

                scale = quantizer.scale.to(dev, dtype=dtype)
                W_quant = real_quantize(W, scale, wbits, dev)

                # 执行 Few-bit flip
                offset = 2 ** (wbits - 1)
                W_quant2 = W_quant + offset
                W_quant2 = Error_injection_fewbit(W_quant2, bit_w=wbits, flip_positions=flip_tasks[strict_key])
                W_quant2 = W_quant2 - offset

                maxq = 2 ** (wbits - 1) - 1
                W_quant = W_quant2.clamp(-maxq - 1, maxq).to(torch.int16)

                W_dequant = (W_quant * scale).to(dtype)
                module.weight.data = W_dequant

                del W, W_quant, W_dequant

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))
            current_output = torch.stack(layer_output_list)

        # 恢复
        for name, module in subset.items():
            if name in layer_backup:
                module.weight.data = layer_backup[name].to(dev)
        del layer_backup

    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)
    return flipped_loss, flipped_ppl


def flip_and_evaluate_fewbit_2_3(model, layers, quantizers, initial_input, testenc, dev, wbits,
                             seqlen, nsamples, attack_logs, model_kwargs=None, debug_flip=False):
    """
    [Mode 8] PBS 验证函数
    修改：仅添加 debug_flip 校验模块，保留原有一切逻辑和外部函数调用。
    """
    if model_kwargs is None: model_kwargs = {}

    from biterror1.ttt_7_18_all_value_flip import Error_injection_fewbit
    import ast
    import torch

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)

    # 1. 建立严格映射
    strict_map = {}
    for i, layer in enumerate(layers):
        subset = find_layers(layer, layers=[nn.Linear])
        for name, mod in subset.items():
            strict_map[f"{i}_{name}"] = mod

    # 2. 解析任务
    flip_tasks = {}
    for log in attack_logs:
        if isinstance(log, dict):
            raw_name, coords_raw, bit = log.get('Layer'), log.get('Coords'), log.get('Bit')
        else:
            raw_name, coords_raw, bit = log[1], log[2], log[3]

        if raw_name not in strict_map:
            raise ValueError(f"Layer {raw_name} not found.")

        coords = ast.literal_eval(coords_raw) if isinstance(coords_raw, str) else coords_raw
        if raw_name not in flip_tasks: flip_tasks[raw_name] = []
        flip_tasks[raw_name].append([coords, int(bit)])

    # 3. 逐层攻击
    for idx in range(len(layers)):
        layer = layers[idx].to(dev)
        subset = find_layers(layer, layers=[nn.Linear])
        layer_backup = {}

        for name, module in subset.items():
            strict_key = f"{idx}_{name}"
            # 兼容 key
            quantizer = quantizers.get(strict_key) or quantizers.get(f"{idx}.{name}")

            if quantizer and strict_key in flip_tasks:
                W = module.weight.data
                layer_backup[name] = W.cpu().clone()

                scale = quantizer.scale.to(dev, dtype=dtype)
                W_quant = real_quantize(W, scale, wbits, dev)

                # 执行 Few-bit flip
                offset = 2 ** (wbits - 1)
                W_quant2 = W_quant + offset

                # ================= [DEBUG START: 记录修改前] =================
                W_before_debug = None
                if debug_flip:
                    W_before_debug = W_quant2.clone().to(torch.int64)
                # ===========================================================

                # --- 原封不动调用你的外部函数 ---
                W_quant2 = Error_injection_fewbit(W_quant2, bit_w=wbits, flip_positions=flip_tasks[strict_key])

                # ================= [DEBUG START: 校验修改后] =================
                if debug_flip:
                    logger.info(f"\n====== [Layer: {strict_key}] Debug Verification ======")
                    W_after_debug = W_quant2.to(torch.int64)

                    # 计算差异
                    diff_mask = torch.bitwise_xor(W_before_debug, W_after_debug)
                    changed_indices = torch.nonzero(diff_mask, as_tuple=False)

                    actual_changes_set = set()
                    print(f"  >>> [Actual Changes Detected (By Error_injection_fewbit)]:")
                    for coord in changed_indices:
                        r, c = coord[0].item(), coord[1].item()
                        val_diff = diff_mask[r, c].item()
                        # 解析多 Bit 变化
                        for b in range(16):
                            if (val_diff >> b) & 1:
                                print(
                                    f"      Coords=[{r}, {c}] | Bit={b} | Val_Before={W_before_debug[r, c]} -> Val_After={W_after_debug[r, c]}")
                                actual_changes_set.add((r, c, b))

                    # 统计预期
                    expected_set = set()
                    for t_coord, t_bit in flip_tasks[strict_key]:
                        expected_set.add((t_coord[0], t_coord[1], t_bit))

                    print(f"  >>> [Expected Tasks]: {len(expected_set)} items")

                    missing = expected_set - actual_changes_set
                    extra = actual_changes_set - expected_set

                    if len(missing) == 0 and len(extra) == 0:
                        logger.info("  ✅ 校验通过：外部函数执行完全正确！")
                    else:
                        logger.error(f"  ❌ 校验失败！Missing: {len(missing)}, Extra: {len(extra)}")
                        if len(missing) > 0: print(f"     Missing sample: {list(missing)[:3]}")
                    logger.info("========================================================\n")
                # ===========================================================

                W_quant2 = W_quant2 - offset

                maxq = 2 ** (wbits - 1) - 1
                W_quant = W_quant2.clamp(-maxq - 1, maxq).to(torch.int16)

                W_dequant = (W_quant * scale).to(dtype)
                module.weight.data = W_dequant

                del W, W_quant, W_dequant

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))
            current_output = torch.stack(layer_output_list)

        # 恢复
        for name, module in subset.items():
            if name in layer_backup:
                module.weight.data = layer_backup[name].to(dev)
        del layer_backup

    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)
    return flipped_loss, flipped_ppl

def flip_and_evaluate_fewbit(model, layers, quantizers, initial_input, testenc, dev, wbits,
                             seqlen, nsamples, attack_logs, model_kwargs=None, debug_flip=False):
    """
    [Mode 8] PBS 验证函数
    修改：仅添加 debug_flip 校验模块，保留原有一切逻辑和外部函数调用。
    """
    if model_kwargs is None: model_kwargs = {}

    from biterror1.ttt_7_18_all_value_flip import Error_injection_fewbit
    import ast
    import torch

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)

    # 1. 建立严格映射
    strict_map = {}
    for i, layer in enumerate(layers):
        subset = find_layers(layer, layers=[nn.Linear])
        for name, mod in subset.items():
            strict_map[f"{i}_{name}"] = mod

    # 2. 解析任务
    flip_tasks = {}
    for log in attack_logs:
        if isinstance(log, dict):
            raw_name, coords_raw, bit = log.get('Layer'), log.get('Coords'), log.get('Bit')
        else:
            raw_name, coords_raw, bit = log[1], log[2], log[3]

        if raw_name not in strict_map:
            raise ValueError(f"Layer {raw_name} not found.")

        coords = ast.literal_eval(coords_raw) if isinstance(coords_raw, str) else coords_raw
        if raw_name not in flip_tasks: flip_tasks[raw_name] = []
        flip_tasks[raw_name].append([coords, int(bit)])

    # 3. 逐层攻击
    for idx in range(len(layers)):
        layer = layers[idx].to(dev)
        subset = find_layers(layer, layers=[nn.Linear])
        layer_backup = {}

        for name, module in subset.items():
            strict_key = f"{idx}_{name}"
            # 兼容 key
            quantizer = quantizers.get(strict_key) or quantizers.get(f"{idx}.{name}")

            if quantizer and strict_key in flip_tasks:
                W = module.weight.data
                layer_backup[name] = W.cpu().clone()

                scale = quantizer.scale.to(dev, dtype=dtype)
                if scale.dim() == 1:
                    raise ValueError("Scale cannot be 1D. Please reshape it to (N, 1) before calling real_quantize.")
                W_quant = real_quantize(W, scale, wbits, dev)

                # 执行 Few-bit flip
                offset = 2 ** (wbits - 1)
                W_quant2 = W_quant + offset

                # ================= [DEBUG START: 记录修改前] =================
                W_before_debug = None
                if debug_flip:
                    W_before_debug = W_quant2.clone().to(torch.int64)
                # ===========================================================

                # --- 原封不动调用你的外部函数 ---
                W_quant2 = Error_injection_fewbit(W_quant2, bit_w=wbits, flip_positions=flip_tasks[strict_key])

                # ================= [DEBUG START: 校验修改后] =================
                if debug_flip:
                    logger.info(f"\n====== [Layer: {strict_key}] Debug Verification ======")
                    W_after_debug = W_quant2.to(torch.int64)

                    # 计算差异
                    diff_mask = torch.bitwise_xor(W_before_debug, W_after_debug)
                    changed_indices = torch.nonzero(diff_mask, as_tuple=False)

                    actual_changes_set = set()
                    print(f"  >>> [Actual Changes Detected (By Error_injection_fewbit)]:")
                    for coord in changed_indices:
                        r, c = coord[0].item(), coord[1].item()
                        val_diff = diff_mask[r, c].item()
                        # 解析多 Bit 变化
                        for b in range(16):
                            if (val_diff >> b) & 1:
                                print(
                                    f"      Coords=[{r}, {c}] | Bit={b} | Val_Before={W_before_debug[r, c]} -> Val_After={W_after_debug[r, c]}")
                                actual_changes_set.add((r, c, b))

                    # 统计预期
                    expected_set = set()
                    for t_coord, t_bit in flip_tasks[strict_key]:
                        expected_set.add((t_coord[0], t_coord[1], t_bit))

                    print(f"  >>> [Expected Tasks]: {len(expected_set)} items")

                    missing = expected_set - actual_changes_set
                    extra = actual_changes_set - expected_set

                    if len(missing) == 0 and len(extra) == 0:
                        logger.info("  ✅ 校验通过：外部函数执行完全正确！")
                    else:
                        logger.error(f"  ❌ 校验失败！Missing: {len(missing)}, Extra: {len(extra)}")
                        if len(missing) > 0: print(f"     Missing sample: {list(missing)[:3]}")
                    logger.info("========================================================\n")

                W_quant2 = W_quant2 - offset

                maxq = 2 ** (wbits - 1) - 1
                W_quant = W_quant2.clamp(-maxq - 1, maxq).to(torch.int16)

                W_dequant = (W_quant * scale).to(dtype)
                module.weight.data = W_dequant

                # 在现有 debug_flip 块的末尾加入
                if debug_flip:
                    #  更新实际权重变化值
                    logger.info(f"更新实际变化的权重值>>>")
                    for t_coord, t_bit in flip_tasks[strict_key]:
                        r, c = t_coord
                        w_before = layer_backup[name][r, c].item()
                        w_after = W_dequant[r, c].item()
                        delta_float = w_after - w_before
                        logger.info(
                            f"[DeltaFloat] {strict_key} W[{r},{c}] bit={t_bit}: "
                            f"{w_before:.6f} → {w_after:.6f}, delta={delta_float:.6f}"
                        )
                        logger.info(
                            f"[raw_target_update] "
                            f"[1, '{idx}_{strict_key.split('_', 1)[1]}', [{r}, {c}], {t_bit}, {delta_float:.8f}]"
                            # 直接复制这行更新 raw_target
                        )

                del W, W_quant, W_dequant

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))
            current_output = torch.stack(layer_output_list)

        # 恢复
        for name, module in subset.items():
            if name in layer_backup:
                module.weight.data = layer_backup[name].to(dev)
        del layer_backup

    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)
    return flipped_loss, flipped_ppl




def apply_attack_for_utility(model, layers, quantizers, attack_logs, wbits, dev, debug_flip=False):
    """
    [Utility专用] 将 PBS 攻击应用到整个模型权重上。
    逻辑严格参考 flip_and_evaluate_fewbit，确保量化/翻转行为一致。
    新增: debug_flip 校验模块，确保注入准确性。
    返回: backup_weights (用于恢复)
    """
    import ast
    import torch
    from biterror1.ttt_7_18_all_value_flip import Error_injection_fewbit  # 导入你的翻转函数
    from custom_logger import logger  # 假设你有logger

    # 1. 解析攻击任务 flip_tasks
    # 格式: { "0.self_attn.q_proj": [ [coords, bit], ... ] }
    flip_tasks = {}
    for log in attack_logs:
        # 兼容 dict (csv read) 和 tuple
        if isinstance(log, dict):
            raw_name = log.get('Layer') or log.get('module')  # 兼容不同列名
            coords_raw = log.get('Coords')
            bit = int(log.get('Bit'))
        else:
            # 列表格式
            raw_name, coords_raw, bit = log[1], log[2], log[3]

        coords = ast.literal_eval(coords_raw) if isinstance(coords_raw, str) else coords_raw

        if raw_name not in flip_tasks: flip_tasks[raw_name] = []
        flip_tasks[raw_name].append([coords, bit])

    # 2. 执行攻击并备份
    backup_weights = {}  # { module: original_cpu_weight }
    dtype = next(model.parameters()).dtype

    if debug_flip:
        logger.info(f"    >>> [Debug] 开启翻转校验模式 (Verbose Mode)")

    # 遍历所有层
    for i, layer in enumerate(layers):
        subset = find_layers(layer, layers=[nn.Linear])

        for name, module in subset.items():
            full_name = f"{i}.{name}"  # e.g., 0.self_attn.q_proj

            # 检查该层是否有攻击任务
            if full_name not in flip_tasks:
                continue

            # 获取量化器
            quantizer_key_1 = f"{i}_{name}"
            quantizer_key_2 = f"{i}.{name}"
            quantizer = quantizers.get(quantizer_key_1) or quantizers.get(quantizer_key_2)

            if quantizer is None:
                logger.warning(f"    [Skip] 找不到 Quantizer: {full_name}")
                continue

            # === 核心翻转逻辑 ===

            # A. 备份 (移动到 CPU 节省显存)
            W = module.weight.data
            backup_weights[module] = W.cpu().clone()

            # B. 准备数据
            target_dev = W.device
            W_curr = W.to(dev)
            scale = quantizer.scale.to(dev, dtype=dtype)

            # C. 量化 (Real Quantize)
            # 假设 real_quantize 在上下文可用
            W_quant = real_quantize(W_curr, scale, wbits, dev)

            # D. Offset & Flip
            offset = 2 ** (wbits - 1)
            W_quant2 = W_quant + offset

            # ================= [DEBUG START: 记录修改前] =================
            W_before_debug = None
            if debug_flip:
                W_before_debug = W_quant2.clone().to(torch.int64)
            # ===========================================================

            # 调用外部注入函数
            W_quant2 = Error_injection_fewbit(
                W_quant2,
                bit_w=wbits,
                flip_positions=flip_tasks[full_name]
            )

            # ================= [DEBUG START: 校验修改后] =================
            if debug_flip:
                # logger.info(f"\n====== [Layer: {full_name}] Debug Verification ======")
                W_after_debug = W_quant2.to(torch.int64)

                # 计算差异
                diff_mask = torch.bitwise_xor(W_before_debug, W_after_debug)
                # 快速检查是否有差异
                if torch.sum(diff_mask) > 0:
                    changed_indices = torch.nonzero(diff_mask, as_tuple=False)

                    actual_changes_set = set()
                    # print(f"  >>> [Actual Changes Detected]:")
                    for coord in changed_indices:
                        r, c = coord[0].item(), coord[1].item()
                        val_diff = diff_mask[r, c].item()
                        # 解析多 Bit 变化
                        for b in range(16):  # 假设 wbits <= 16
                            if (val_diff >> b) & 1:
                                # print(f"      Coords=[{r}, {c}] | Bit={b} | {W_before_debug[r, c]} -> {W_after_debug[r, c]}")
                                actual_changes_set.add((r, c, b))

                    # 统计预期
                    expected_set = set()
                    for t_coord, t_bit in flip_tasks[full_name]:
                        expected_set.add((t_coord[0], t_coord[1], t_bit))

                    # 比较集合
                    missing = expected_set - actual_changes_set
                    extra = actual_changes_set - expected_set

                    if len(missing) == 0 and len(extra) == 0:
                        # logger.info(f"  ✅ 层 {full_name}: {len(expected_set)} 个翻转任务执行正确。")
                        pass
                    else:
                        logger.error(f"  ❌ 层 {full_name} 校验失败！Missing: {len(missing)}, Extra: {len(extra)}")
                        if len(missing) > 0: logger.error(f"     Missing: {list(missing)[:3]}")
                else:
                    # 如果预期有任务但没变化
                    if len(flip_tasks[full_name]) > 0:
                        logger.error(
                            f"  ❌ 层 {full_name} 严重错误：预期有 {len(flip_tasks[full_name])} 个翻转，但权重未发生任何变化！")

                # logger.info("========================================================\n")
            # ===========================================================

            # E. De-Offset & Clamp
            W_quant2 = W_quant2 - offset
            maxq = 2 ** (wbits - 1) - 1
            W_quant = W_quant2.clamp(-maxq - 1, maxq).to(torch.int16)

            # F. Dequantize & Update
            W_dequant = (W_quant * scale).to(dtype)

            # 更新模型权重 (保持原设备)
            module.weight.data = W_dequant.to(target_dev)

    return backup_weights


import ast
import torch
import torch.nn as nn
# 假设这些是你项目中已经存在的引用
from custom_logger import logger
from biterror1.ttt_7_18_all_value_flip import Error_injection_fewbit




def apply_attack_and_evaluate_utility(
        model,
        layers,
        quantizers,
        attack_logs,
        wbits,
        dev,
        tokenizer,  # 新增参数: 用于评估
        eval_batch_size=32,
        shotnum=5,  # 新增参数: MMLU shot数
        limit=None,  # 新增参数: 评估题目限制
        debug_flip=False
):
    """
    [Utility全流程函数]
    1. 备份权重
    2. 应用 PBS 攻击 (带 Debug 校验)
    3. 运行 Utility 评估 (MMLU/PIQA)
    4. 自动恢复权重 (确保无副作用)

    返回: scores (字典, 包含 MMLU, PIQA 分数)
    """
    from datautils import run_utility_evaluation
    from datautils import run_utility_evaluation_ac4
    from datautils import run_utility_evaluation_hellaswag
    # ==========================
    # Step 1: 解析攻击任务
    # ==========================
    flip_tasks = {}
    for log in attack_logs:
        # 兼容 dict (csv read) 和 tuple/list
        if isinstance(log, dict):
            raw_name = log.get('Layer') or log.get('module')
            coords_raw = log.get('Coords')
            bit = int(log.get('Bit'))
        else:
            # 列表格式 [idx, name, coords, bit, ...]
            raw_name, coords_raw, bit = log[1], log[2], log[3]

        coords = ast.literal_eval(coords_raw) if isinstance(coords_raw, str) else coords_raw

        if raw_name not in flip_tasks: flip_tasks[raw_name] = []
        flip_tasks[raw_name].append([coords, bit])

    # 准备变量
    backup_weights = {}  # { module: original_cpu_weight }
    dtype = next(model.parameters()).dtype

    if debug_flip:
        logger.info(f"    >>> [Debug] 开启翻转校验模式 (Verbose Mode)")

    try:
        # ==========================
        # Step 2: 执行攻击并备份
        # ==========================
        # 遍历所有层
        for i, layer in enumerate(layers):
            subset = find_layers(layer, layers=[nn.Linear])

            for name, module in subset.items():
                full_name = f"{i}_{name}"  # e.g., 0.self_attn.q_proj
                # 检查该层是否有攻击任务
                if full_name not in flip_tasks:
                    continue
                else:
                    if debug_flip:
                        logger.info(f"找到攻击层 {i}_{name}")

                # 获取量化器
                quantizer_key_1 = f"{i}_{name}"
                quantizer_key_2 = f"{i}.{name}"
                quantizer = quantizers.get(quantizer_key_1) or quantizers.get(quantizer_key_2)

                if quantizer is None:
                    raise ValueError(f"    [Skip] 找不到 Quantizer: {full_name}")
                    # logger.warning(f"    [Skip] 找不到 Quantizer: {full_name}")
                    # continue

                # === A. 备份 (移动到 CPU 节省显存) ===
                W = module.weight.data
                backup_weights[module] = W.cpu().clone()

                # === B. 准备数据 ===
                target_dev = W.device
                W_curr = W.to(dev)
                scale = quantizer.scale.to(dev, dtype=dtype)

                # === C. 量化 (Real Quantize) ===
                # 调用你的量化函数
                W_quant = real_quantize(W_curr, scale, wbits, dev)

                # === D. Offset & Flip ===
                offset = 2 ** (wbits - 1)
                W_quant2 = W_quant + offset

                # ---------------- [DEBUG: 记录修改前] ----------------
                W_before_debug = None
                if debug_flip:
                    W_before_debug = W_quant2.clone().to(torch.int64)
                # ---------------------------------------------------

                # 调用外部注入函数
                W_quant2 = Error_injection_fewbit(
                    W_quant2,
                    bit_w=wbits,
                    flip_positions=flip_tasks[full_name]
                )

                # ---------------- [DEBUG: 校验修改后] ----------------
                if debug_flip:
                    W_after_debug = W_quant2.to(torch.int64)

                    # 计算差异
                    diff_mask = torch.bitwise_xor(W_before_debug, W_after_debug)

                    if torch.sum(diff_mask) > 0:
                        changed_indices = torch.nonzero(diff_mask, as_tuple=False)
                        actual_changes_set = set()

                        for coord in changed_indices:
                            r, c = coord[0].item(), coord[1].item()
                            val_diff = diff_mask[r, c].item()
                            # 解析多 Bit 变化
                            for b in range(16):  # 假设 wbits <= 16
                                if (val_diff >> b) & 1:
                                    actual_changes_set.add((r, c, b))

                        # 统计预期
                        expected_set = set()
                        for t_coord, t_bit in flip_tasks[full_name]:
                            expected_set.add((t_coord[0], t_coord[1], t_bit))

                        # 比较集合
                        missing = expected_set - actual_changes_set
                        extra = actual_changes_set - expected_set

                        if len(missing) == 0 and len(extra) == 0:
                            logger.info(f"  ✅ 层 {full_name}: {len(expected_set)} 个翻转任务执行正确。")
                            pass
                        else:
                            logger.error(f"  ❌ 层 {full_name} 校验失败！Missing: {len(missing)}, Extra: {len(extra)}")
                            if len(missing) > 0: logger.error(f"     Missing: {list(missing)[:3]}")
                    else:
                        if len(flip_tasks[full_name]) > 0:
                            logger.error(f"  ❌ 层 {full_name} 严重错误：预期有翻转但权重未变！")
                # ---------------------------------------------------

                # === E. De-Offset & Clamp ===
                W_quant2 = W_quant2 - offset
                maxq = 2 ** (wbits - 1) - 1
                W_quant = W_quant2.clamp(-maxq - 1, maxq).to(torch.int16)

                # === F. Dequantize & Update ===
                W_dequant = (W_quant * scale).to(dtype)

                # 更新模型权重
                module.weight.data = W_dequant.to(target_dev)

        # ==========================
        # Step 3: 运行评估 (Attack Applied)
        # ==========================
        logger.info(f"    >>> [Eval] 攻击注入完成，开始 Utility 评估 (shot={shotnum})...")

        # 调用外部评估函数
        # scores = run_utility_evaluation(model, tokenizer, dev, shotnum=shotnum, limit=limit)
        # scores = run_utility_evaluation_ac4(model, tokenizer, dev, batch_size=eval_batch_size, shotnum=shotnum, limit=limit)
        scores = run_utility_evaluation_hellaswag(model, tokenizer, dev, batch_size=eval_batch_size, shotnum=shotnum, limit=limit)
        return scores

    except Exception as e:
        logger.error(f"    ❌ [Error] 攻击或评估过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return {"MMLU (Avg)": 0.0, "PIQA": 0.0, "Error": str(e)}

    finally:
        # ==========================
        # Step 4: 自动恢复权重 (Always Executed)
        # ==========================
        if len(backup_weights) > 0:
            # logger.info(f"    >>> [Restore] 正在恢复 {len(backup_weights)} 个层的原始权重...")
            for module, cpu_weight in backup_weights.items():
                # 恢复到模型原本所在的 device
                module.weight.data = cpu_weight.to(module.weight.device)

            # 清理内存
            del backup_weights
            torch.cuda.empty_cache()
            # logger.info(f"    >>> [Restore] 权重恢复完毕，显存已清理。")
        else:
            logger.warning("    >>> [Restore] 没有备份权重，跳过恢复 (可能是没有层被攻击)。")


def restore_model_weights(backup_weights):
    """
    恢复模型权重
    """
    logger.info(f"    >>> [Restore] 正在恢复 {len(backup_weights)} 个层的权重...")
    for module, orig_weight in backup_weights.items():
        # 恢复到 module 当前所在的 device
        target_dev = module.weight.device
        module.weight.data = orig_weight.to(target_dev)

def flip_and_evaluate_onebit_hook(model, layers, quantizers, initial_input, testenc, dev, wbits,
                                  seqlen, nsamples, attack_log,
                                  hook_layer=None, save_dir="./logs/data/hook_data.pt", max_samples=10,
                                  model_kwargs=None):
    """
    [Mode 2] 翻转指定单比特 + Hook 捕获
    优化：全 GPU 流程，适配 H200，支持 model_kwargs
    """
    if hook_layer is None:
        hook_layer = []
    if model_kwargs is None:
        model_kwargs = {}

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)

    # 解析 Hook 目标
    hook_layer_dict = {}
    for hl in hook_layer:
        parts = hl.split('.', 1)
        if len(parts) == 2:
            l_idx, name = int(parts[0]), parts[1]
            if l_idx not in hook_layer_dict: hook_layer_dict[l_idx] = []
            hook_layer_dict[l_idx].append(name)

    # 准备 Hook 数据容器
    hook_data_paths = {l_idx: {name: {'inputs': [], 'outputs': [], 'weight': None, 'scale': None}
                               for name in names} for l_idx, names in hook_layer_dict.items()}

    # 钩子函数 (仅在 Hook 时转移到 CPU 以节省显存，计算仍在 GPU)
    def create_hook(l_idx, name):
        def hook_fn(module, input, output):
            if len(hook_data_paths[l_idx][name]['inputs']) < max_samples:
                # 移动输入到 CPU
                input_data = [inp.detach().cpu() for inp in input]
                # 移动输出到 CPU
                if isinstance(output, tuple):
                    output_data = [out.detach().cpu() for out in output]
                else:
                    output_data = output.detach().cpu()

                hook_data_paths[l_idx][name]['inputs'].append({'data': input_data})
                hook_data_paths[l_idx][name]['outputs'].append({'data': output_data})

        return hook_fn

    # 解析攻击信息
    layer_idx, layer_name, pos, bit_pos, _ = attack_log
    flip_info = [[pos, bit_pos]]

    hooks = []
    original_weights_backup = {}  # 仅备份修改过的层

    # 逐层处理
    for idx in range(len(layers)):
        layer = layers[idx].to(dev)
        subset = find_layers(layer, layers=[nn.Linear, MatmulModule, IdentityModule])

        # 备份当前层权重 (CPU RAM)
        original_weights_backup[idx] = {n: m.weight.data.cpu().clone() for n, m in subset.items() if
                                        hasattr(m, 'weight')}

        for name, module in subset.items():
            # 1. 注册钩子
            if idx in hook_layer_dict and name in hook_layer_dict[idx]:
                hooks.append(module.register_forward_hook(create_hook(idx, name)))
                # 保存量化参数供分析 (如果有)
                q_key = f"{idx}_{name}"
                if q_key in quantizers:
                    hook_data_paths[idx][name]['scale'] = quantizers[q_key].scale.cpu()

            # 2. 执行量化和攻击 (仅针对 Linear)
            if isinstance(module, nn.Linear):
                quantizer = quantizers.get(f"{idx}_{name}")
                if quantizer is None: continue

                W = module.weight.data
                scale = quantizer.scale.to(dev, dtype=dtype)

                # 量化
                W_quant = real_quantize(W, scale, wbits, dev)

                # 攻击：仅对目标层执行翻转
                if idx == layer_idx and name == layer_name:
                    offset = 2 ** (wbits - 1)
                    W_quant_unsigned = W_quant + offset

                    # 记录翻转前的值 (用于 Log)
                    W_before = (W_quant * scale).to(dtype)
                    val_before = W_before[pos[0], pos[1]].item()

                    # 注入错误
                    W_quant_flipped = Error_injection_fewbit(W_quant_unsigned, bit_w=wbits, flip_positions=flip_info)

                    W_quant = W_quant_flipped - offset
                    maxq = 2 ** (wbits - 1) - 1
                    W_quant = W_quant.clamp(-maxq - 1, maxq).to(torch.int16)

                    # 记录翻转后的值
                    W_after = (W_quant * scale).to(dtype)
                    val_after = W_after[pos[0], pos[1]].item()

                    logger.info(f"触发翻转: Layer {idx}.{name} Pos {pos} Bit {bit_pos}")
                    logger.info(f"值变化: {val_before:.6f} -> {val_after:.6f}")

                # 反量化并应用
                W_dequant = (W_quant * scale).to(dtype)
                module.weight.data = W_dequant

                # 如果是 Hook 层，保存当前使用的权重
                if idx in hook_layer_dict and name in hook_layer_dict[idx]:
                    # 这里的 weight path 暂时不写盘，直接存 tensor 稍后统一保存
                    hook_data_paths[idx][name]['weight'] = W_dequant.cpu()

                del W, W_quant, W_dequant

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                # 使用 model_kwargs
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))

            current_output = torch.stack(layer_output_list)

        # 快速恢复权重 (从 RAM 恢复)
        for name, module in subset.items():
            if hasattr(module, 'weight'):
                module.weight.data = original_weights_backup[idx][name].to(dev)

        # 释放备份
        del original_weights_backup[idx]

    # 计算 Loss
    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)

    # 移除钩子
    for h in hooks: h.remove()

    # 保存 Hook 数据
    if save_dir:
        create_dir(save_dir)
        try:
            torch.save(hook_data_paths, save_dir)
            logger.info(f"Hook 数据已保存至: {save_dir}")
        except Exception as e:
            logger.error(f"保存 Hook 数据失败: {e}")

    return flipped_loss, flipped_ppl, hook_data_paths


import time
import torch
import numpy as np


def benchmark_inference_latency0(model, layers, dev, seqlen, wbits, model_kwargs=None,
                                batch_size=1, repeats=100, warmup=10):
    """
    [专业版] 测量模型推理延迟 (Latency) 和吞吐量 (Throughput)
    """
    logger.info(f"*** 开始基准测试 (Benchmark) ***")
    logger.info(f"配置: Batch={batch_size}, SeqLen={seqlen}, Wbits={wbits}, Repeats={repeats}")

    if model_kwargs is None: model_kwargs = {}

    # 1. 构造 Dummy Input (模拟真实输入形状)
    # 假设输入是 [Batch, Seq, Hidden]，直接造一个随机 Embedding 输出
    # 这样可以跳过 Embedding 层，直接测 Layer 的计算压力
    hidden_dim = model.config.hidden_size
    dtype = next(model.parameters()).dtype

    # 构造输入张量 (模拟上一层的输出)
    dummy_input = torch.randn(batch_size, seqlen, hidden_dim, device=dev, dtype=dtype)

    # 准备必要的 kwargs (如 mask)
    # 注意：这里需要确保 kwargs 的 shape 也是对的，简单起见我们复用 model_kwargs
    # 但如果 model_kwargs 里有特定 batch 维度的 tensor，可能需要调整

    # 2. 预热 (Warm-up)
    # 让 GPU 唤醒，编译 Kernel，避免第一次运行的抖动
    logger.info("正在预热 GPU (Warm-up)...")
    with torch.no_grad():
        for _ in range(warmup):
            current = dummy_input
            for layer in layers:
                layer = layer.to(dev)
                # 模拟 evaluate_full_model 的调用方式
                out = layer(current, **model_kwargs)[0]
                current = out
    torch.cuda.synchronize()  # 确保预热完成

    # 3. 正式计时 (使用 CUDA Event 以获得纳秒级精度)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    logger.info("正在进行压力测试...")

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        for _ in range(repeats):
            current = dummy_input
            for layer in layers:
                # 这里的 .to(dev) 在 H200 上如果是同一设备几乎无开销，
                # 但为了模拟 evaluate_full_model 的真实行为，保留它
                layer = layer.to(dev)
                out = layer(current, **model_kwargs)[0]
                current = out

            # 这是一个 Trick: 强制 GPU 完成这一轮的计算依赖
            # 避免 PyTorch 只是把 Kernel 塞进队列就返回了
            # 但不要在这里用 synchronize()，否则会阻断流水线并行，导致测得比实际慢
            # 只要最后 current 被使用了（或者最后统一 sync），计时就是准的。

    end_event.record()
    torch.cuda.synchronize()  # 等待所有 repeats 完成

    # 4. 计算统计数据
    total_time_ms = start_event.elapsed_time(end_event)  # 返回毫秒
    avg_time_ms = total_time_ms / repeats

    # 计算吞吐量 (Tokens per Second)
    # 总 Tokens = Batch * SeqLen
    tokens_per_sec = (batch_size * seqlen) / (avg_time_ms / 1000.0)

    logger.info(f"--------------------------------------------------")
    logger.info(f"测速结果 (RoR / Baseline):")
    logger.info(f"平均延迟 (Latency): {avg_time_ms:.2f} ms / batch")
    logger.info(f"吞吐量 (Throughput): {tokens_per_sec:.2f} tokens/sec")
    logger.info(f"--------------------------------------------------")

    return avg_time_ms, tokens_per_sec


import torch
import time
import torch._dynamo


def benchmark_inference_latency_2_26(model, layers, dev, seqlen, wbits=None, model_kwargs=None,
                                batch_size=1, repeats=50, warmup=10):
    """
    [纯净版] 测量原生 PyTorch (Eager Mode) 的逐层推理延迟。
    去除 JIT 编译，直接在 GPU 上执行 forward，贴近底层物理测速。
    """
    logger.info(f"*** 开始基准测试 (Eager Mode Benchmark) ***")
    logger.info(f"配置: Batch={batch_size}, SeqLen={seqlen}, Repeats={repeats}")

    if model_kwargs is None:
        model_kwargs = {}

    # 1. 构造 Dummy Input
    # 直接在目标设备上生成张量，避免 CPU->GPU 拷贝带来的误差
    hidden_dim = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    dummy_input = torch.randn(batch_size, seqlen, hidden_dim, device=dev, dtype=dtype)

    # 2. 预热 (Warm-up)
    # 虽然没有编译，但 GPU 需要时间唤醒，且 CUDA context 需要初始化
    logger.info(">>> 正在预热 (Warm-up)...")
    with torch.no_grad():
        for _ in range(warmup):
            x = dummy_input
            for layer in layers:
                x = layer(x, **model_kwargs)[0]

    # 确保预热的所有计算在 GPU 上彻底执行完毕
    torch.cuda.synchronize(device=dev)
    logger.info(">>> 预热完成，开始正式计时...")

    # 3. 正式计时 (使用精确的 CUDA Event)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.no_grad():
        for _ in range(repeats):
            x = dummy_input
            # 采用你 evaluate_full_model 中的逐层推理逻辑
            for layer in layers:
                x = layer(x, **model_kwargs)[0]

    end_event.record()

    # 等待所有 CUDA 算子执行完毕
    torch.cuda.synchronize(device=dev)

    # 4. 统计结果
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / repeats

    # 吞吐量计算公式：(BatchSize * 序列长度) / 每批次平均耗时(秒)
    tokens_per_sec = (batch_size * seqlen) / (avg_time_ms / 1000.0)

    logger.info(f"-" * 50)
    logger.info(f"[Result] Batch={batch_size}")
    logger.info(f"平均延迟 (Latency): {avg_time_ms:.2f} ms")
    logger.info(f"吞吐量 (Throughput): {tokens_per_sec:.2f} tokens/s")
    logger.info(f"-" * 50)

    return avg_time_ms

def benchmark_inference_latency(model, layers, initial_input, dev, seqlen,
                                 model_kwargs=None, repeats=50, warmup=10):
    import numpy as np
    if model_kwargs is None:
        model_kwargs = {}

    # initial_input shape: [nsamples, seqlen, hidden_dim]
    nsamples = initial_input.shape[0]
    logger.info(f"Benchmark: nsamples={nsamples}, seqlen={seqlen}, repeats={repeats}")

    def run_one_pass():
        current_output = initial_input.to(dev)
        for layer in layers:
            layer = layer.to(dev)
            outs = []
            for j in range(nsamples):
                out = layer(current_output[j].unsqueeze(0), **model_kwargs)[0]
                outs.append(out.squeeze(0))
            current_output = torch.stack(outs)
        return current_output

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            run_one_pass()
    torch.cuda.synchronize(dev)

    # 逐次计时
    timings = []
    with torch.no_grad():
        for _ in range(repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            run_one_pass()
            e.record()
            torch.cuda.synchronize(dev)
            timings.append(s.elapsed_time(e))

    avg = np.mean(timings)
    std = np.std(timings)
    p95 = np.percentile(timings, 95)
    tokens_per_sec = (nsamples * seqlen) / (avg / 1000.0)

    logger.info(f"Latency    : {avg:.2f} ± {std:.2f} ms")
    logger.info(f"P95        : {p95:.2f} ms")
    logger.info(f"Throughput : {tokens_per_sec:.0f} tokens/s")

    return avg, std




def benchmark_inference_latency_compiled_2_26(model, layers, dev, seqlen, wbits, model_kwargs=None,
                                         batch_size=1, repeats=50, warmup=10):
    """
    [JIT 优化版] 使用 torch.compile 消除 Python Overhead
    """
    logger.info(f"*** 开始基准测试 (JIT Compiled Benchmark) ***")
    logger.info(f"配置: Batch={batch_size}, SeqLen={seqlen}, Repeats={repeats}")

    if model_kwargs is None: model_kwargs = {}

    # 1. 构造 Dummy Input
    hidden_dim = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    dummy_input = torch.randn(batch_size, seqlen, hidden_dim, device=dev, dtype=dtype)

    # 2. 定义待编译的 Step 函数 (模拟 forward loop)
    # 我们只编译 layer 循环部分，这是计算密集区
    def inference_step(x_in, layer_list, **kwargs):
        x = x_in
        for layer in layer_list:
            x = layer(x, **kwargs)[0]
        return x

    # 3. 启用 JIT 编译 (Method 1)
    logger.info(">>> 正在编译计算图 (torch.compile mode='reduce-overhead')...")
    # 重置 Dynamo 缓存，防止之前的编译残留
    torch._dynamo.reset()

    # reduce-overhead: 使用 CUDA Graphs 极大降低 kernel 启动开销 (适合 BS=1)
    # max-autotune: 暴力搜索最优 Triton 算子 (适合 BS 大的时候)
    mode = "reduce-overhead" if batch_size < 16 else "max-autotune"

    try:
        opt_step_fn = torch.compile(inference_step, mode=mode, fullgraph=False)
    except Exception as e:
        logger.warning(f"torch.compile 初始化失败: {e}, 降级为默认模式")
        opt_step_fn = torch.compile(inference_step)

    # 4. 预热 (Warm-up) - 极其重要！编译发生在第一次运行
    logger.info(">>> 正在预热并等待编译 (First run triggers compilation)...")
    start_warm = time.time()
    with torch.no_grad():
        # 第一次运行会很慢，因为在做 trace 和 codegen
        for _ in range(max(10, warmup)):
            _ = opt_step_fn(dummy_input, layers, **model_kwargs)
    torch.cuda.synchronize()
    logger.info(f"预热完成，耗时: {time.time() - start_warm:.2f}s")

    # 5. 正式计时
    logger.info(">>> 正在进行压力测试...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        for _ in range(repeats):
            _ = opt_step_fn(dummy_input, layers, **model_kwargs)

    end_event.record()
    torch.cuda.synchronize()

    # 6. 统计
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / repeats
    tokens_per_sec = (batch_size * seqlen) / (avg_time_ms / 1000.0)

    logger.info(f"-" * 50)
    logger.info(f"[Result] Batch={batch_size}")
    logger.info(f"平均延迟 (Latency): {avg_time_ms:.2f} ms")
    logger.info(f"吞吐量 (Throughput): {tokens_per_sec:.2f} tokens/s")
    logger.info(f"-" * 50)

    return avg_time_ms


def benchmark_inference_latency_compiled(model, layers, dev, seqlen, wbits, model_kwargs=None,
                                         batch_size=1, repeats=50, warmup=10):
    logger.info(f"*** 开始基准测试 (JIT Compiled Benchmark) ***")
    logger.info(f"配置: Batch={batch_size}, SeqLen={seqlen}, Repeats={repeats}")

    if model_kwargs is None:
        model_kwargs = {}

    dtype = next(model.parameters()).dtype

    # 构造 dummy token ids（整个模型 forward 用 input_ids）
    dummy_ids = torch.randint(
        0, model.config.vocab_size,
        (batch_size, seqlen),
        device=dev,
        dtype=torch.long
    )

    # attention_mask 全 1
    dummy_mask = torch.ones(batch_size, seqlen, device=dev, dtype=torch.long)

    def inference_step(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask)

    logger.info(">>> 正在编译计算图 (torch.compile dynamic=True)...")
    torch._dynamo.reset()
    mode = "reduce-overhead" if batch_size < 16 else "max-autotune"
    opt_step_fn = torch.compile(inference_step, mode=mode, fullgraph=False, dynamic=True)

    logger.info(">>> 正在预热并等待编译 (First run triggers compilation)...")
    start_warm = time.time()
    with torch.no_grad():
        for _ in range(max(10, warmup)):
            _ = opt_step_fn(dummy_ids, dummy_mask)
    torch.cuda.synchronize()
    logger.info(f"预热完成，耗时: {time.time() - start_warm:.2f}s")

    logger.info(">>> 正在进行压力测试...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        for _ in range(repeats):
            _ = opt_step_fn(dummy_ids, dummy_mask)

    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / repeats
    tokens_per_sec = (batch_size * seqlen) / (avg_time_ms / 1000.0)

    logger.info(f"-" * 50)
    logger.info(f"[Result] Batch={batch_size}")
    logger.info(f"平均延迟 (Latency): {avg_time_ms:.2f} ms")
    logger.info(f"吞吐量 (Throughput): {tokens_per_sec:.2f} tokens/s")
    logger.info(f"-" * 50)

    return avg_time_ms

def evaluate_full_model(model, layers, initial_input, testenc, dev, seqlen, nsamples, model_kwargs=None):
    """
    [优化版] 运行完整的模型前向传播，计算 Loss 和 PPL。
    使用 model_kwargs 统一传递 attention_mask, position_ids, position_embeddings。
    """
    if model_kwargs is None:
        model_kwargs = {}

    # 1. 准备输入 (确保在 GPU)
    # initial_input 可能是 CPU 上的，先移到 GPU
    current_output = initial_input.to(dev)

    # 2. 逐层前向传播
    with torch.no_grad():
        for i, layer in enumerate(layers):
            layer = layer.to(dev)

            layer_output_list = []
            for j in range(nsamples):
                # [seqlen, hidden] -> [1, seqlen, hidden]
                inp = current_output[j].unsqueeze(0)

                # 直接解包 model_kwargs 传给 layer
                # LLaMA/Qwen 会用到 position_ids/embeddings, OPT 则忽略或只用 mask
                out = layer(inp, **model_kwargs)[0]

                layer_output_list.append(out.squeeze(0))

            current_output = torch.stack(layer_output_list)

            # H200 显存极大，不需要 layer.cpu()，也不需要频繁 empty_cache
            # 保持 layer 在 GPU 上供后续使用（如果有）

    # 3. 计算最终 Loss
    loss, ppl = compute_loss(
        model,
        current_output,
        testenc,
        dev,
        seqlen=seqlen,
        nsamples=nsamples,
        model_type=model.config.model_type
    )

    return loss, ppl


def evaluate_quantized_full_model(model, layers, quantizers, wbits, initial_input, testenc, dev, seqlen, nsamples,
                                  model_kwargs=None):
    """
    [量化评估版] 逐层伪量化 -> 前向传播 -> 恢复权重 -> 计算 Loss/PPL
    Args:
        quantizers: 字典，存储每层的量化参数 (scale, zero 等)
        wbits: 量化位数 (如 4, 8)
    """
    if model_kwargs is None:
        model_kwargs = {}

    # 获取模型默认精度 (fp16/bf16)
    dtype = next(model.parameters()).dtype

    # 准备输入
    current_output = initial_input.to(dev)

    # 计算量化截断范围 (假设对称量化 [-maxq, maxq])
    maxq = 2 ** (wbits - 1) - 1

    # 逐层处理
    with torch.no_grad():
        for i, layer in enumerate(layers):
            layer = layer.to(dev)

            # ==========================================
            # 1. 临时量化权重 (Pseudo Quantization)
            # ==========================================
            subset = find_layers(layer)  # 获取该层所有 Linear 模块
            layer_backup = {}  # 用于备份原始权重

            for name, module in subset.items():
                # 构造 Key，需与 Stage 1 生成 quantizers 时的命名逻辑一致
                # 通常是 f"{layer_idx}_{module_name}"
                key = f"{i}_{name}"

                # 兼容性处理：尝试不同的 Key 格式
                if key not in quantizers:
                    key_dot = f"{i}.{name}"
                    if key_dot in quantizers:
                        key = key_dot

                if key in quantizers:
                    quantizer = quantizers[key]

                    # A. 备份原始 FP16 权重 (存到 CPU 节省显存)
                    W_original = module.weight.data
                    layer_backup[name] = W_original.cpu()

                    # B. 执行伪量化 (Simulated Quantization)
                    # W_fake = Dequant(Quant(W))
                    scale = quantizer.scale.to(dev, dtype=dtype)

                    W_fake = fake_quantize(W_original, scale, wbits, dev)

                    # C. 替换权重
                    module.weight.data = W_fake

            # ==========================================
            # 2. 该层前向传播
            # ==========================================
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0)  # [1, seqlen, hidden]
                # 传递 kwargs (mask, pos_ids 等)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))

            current_output = torch.stack(layer_output_list)

            # ==========================================
            # 3. 恢复原始权重 (Restore)
            # ==========================================
            # 必须恢复，否则后续层会受到前面层量化误差的累积影响（Double Quantization 错误）
            # 或者如果之后还要用这个 model 做其他实验，保持 model 纯净很重要
            for name, module in subset.items():
                if name in layer_backup:
                    module.weight.data = layer_backup[name].to(dev)

            # 清理备份，释放内存
            del layer_backup

    # 3. 计算最终 Loss 和 PPL
    loss, ppl = compute_loss(
        model,
        current_output,
        testenc,
        dev,
        seqlen=seqlen,
        nsamples=nsamples,
        model_type=model.config.model_type
    )

    return loss, ppl

def quantize_only_evaluate(model, layers, quantizers, initial_input, testenc, dev, wbits, seqlen,
                            attention_mask, position_ids, nsamples):
    """Quantize the model and evaluate performance without bit flipping"""
    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)
    quantized_loss = 0.0
    quantized_ppl = 0.0
    weight_dir = os.path.join(f"/home/DengLIU/code/gptq/tmp_data/{model.config.model_type}", "weights")
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        logger.info(f"创建权重临时文件夹: {weight_dir}")

    # Store paths for all layer weights
    all_weight_paths = {}

    # Initialize current output
    current_output = initial_input.to(dev)  # Shape: (nsamples, seqlen, hidden_size)
    # logger.info(f"初始输入: shape {current_output.shape}, norm {torch.norm(current_output).item()}")

    # Quantize layer by layer
    for layer_idx in range(len(layers)):
        logger.info(f"量化层 {layer_idx}")
        log_memory(dev)

        # Move entire layer to dev, ensuring all parameters (including layernorm) are on GPU
        layer = layers[layer_idx].to(dev)

        # Check parameter devices
        for name, param in layer.named_parameters():
            if param.device != dev:
                logger.warning(f"Layer {layer_idx} 参数 {name} 在 {param.device}，应为 {dev}")
                param.data = param.data.to(dev)

        # Get submodules using find_layers
        subset = find_layers(layer)
        # logger.info(f"Layer {layer_idx} 子模块: {list(subset.keys())}")

        # Quantize submodules
        for name, module in subset.items():
            # logger.info(f"layer_idx {layer_idx}, name {name}, weight device: {module.weight.device}")
            quantizer = quantizers[f"{layer_idx}_{name}"]
            W = module.weight.data
            # Save original weights to temporary folder
            weight_path = os.path.join(weight_dir, f"weight_{layer_idx}_{name}.pt")
            torch.save(W.cpu(), weight_path)
            all_weight_paths[f"{layer_idx}_{name}"] = weight_path
            # logger.info(f"保存原始权重到: {weight_path}, shape {W.shape}")

            # Perform quantization
            W_quant = real_quantize(W, quantizer.scale.to(dev, dtype=dtype), wbits, dev)
            scale = quantizer.scale.to(dev, dtype=dtype)
            W_quant_dequant = W_quant * scale.unsqueeze(-1)
            W_quant_dequant = W_quant_dequant.to(dtype)
            module.weight.data = W_quant_dequant
            # Release temporary tensors
            del W, W_quant, W_quant_dequant

            # module.weight.data = W


        # Forward pass for current layer
        with torch.no_grad():
            layer_output = torch.zeros(
                (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
            )
            for j in range(nsamples):
                # logger.info(f"Layer {layer_idx}, sample {j}: input device {current_output[j].device}, norm {torch.norm(current_output[j]).item()}")
                if is_llama_like(model):
                    layer_output[j] = layer(
                        current_output[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask.to(dev) if attention_mask is not None else None,
                        position_ids=position_ids.to(dev) if position_ids is not None else None
                    )[0]
                else:
                    layer_output[j] = layer(
                        current_output[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask.to(dev) if attention_mask is not None else None
                    )[0]
                torch.cuda.empty_cache()
            # logger.info(f"Layer {layer_idx} 输出: shape {layer_output.shape}, norm {torch.norm(layer_output).item()}")
            current_output = layer_output.cpu()  # Update current output and move to CPU

    # Compute loss using last layer output
    # logger.info(f"Last layer output norm: {torch.norm(current_output).item()}")
    quantized_loss, quantized_ppl = compute_loss(model, current_output.to(dev), testenc, dev, seqlen, nsamples, model.config.model_type)
    logger.info(f"仅量化的损失: {quantized_loss}, PPL: {quantized_ppl}")

    # Restore original weights
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)  # Ensure layer is on dev for restoration
        subset = find_layers(layer)
        for name, module in subset.items():
            weight_path = all_weight_paths.get(f"{layer_idx}_{name}")
            if weight_path and os.path.exists(weight_path):
                module.weight.data = torch.load(weight_path, map_location=dev)
                os.remove(weight_path)
                # logger.info(f"恢复原始权重并删除: {weight_path}")
            else:
                logger.error(f"未找到权重文件: {weight_path}")
                raise FileNotFoundError(f"Weight file {weight_path} not found")

    # Clean up temporary weight folder
    if os.path.exists(weight_dir):
        shutil.rmtree(weight_dir)
        logger.info(f"清理权重临时文件夹: {weight_dir}")

    return quantized_loss, quantized_ppl

def quantize_only_evaluate_hook(model, layers, quantizers, initial_input, testenc, dev, wbits, seqlen,
                           attention_mask, position_ids, nsamples, hook_layer=None, save_dir="./logs/data/hook_data.pt", max_samples=10):
    """Quantize the model and evaluate performance without bit flipping, with hook functionality to capture intermediate data"""
    if hook_layer is None:
        hook_layer = []

    # 创建临时目录
    base_temp_dir = f"/home/DengLIU/code/gptq/tmp_data/{model.config.model_type}"
    weight_dir = os.path.join(base_temp_dir, "weights")
    hook_data_dir = os.path.join(base_temp_dir, "hook_data")
    for directory in [weight_dir, hook_data_dir]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"创建临时文件夹: {directory}")

    # 初始化存储路径
    all_weight_paths = {}
    hook_data_paths = {layer_idx: {name: {'inputs': [], 'outputs': []} for name in hook_layer} for layer_idx in range(len(layers))}

    # 定义钩子函数，保存前 max_samples 个样本到磁盘
    def create_hook(layer_idx, name):
        def hook_fn(module, input, output):
            if len(hook_data_paths[layer_idx][name]['inputs']) < max_samples:
                input_path = os.path.join(hook_data_dir, f"hook_input_{layer_idx}_{name}_{len(hook_data_paths[layer_idx][name]['inputs'])}.pt")
                output_path = os.path.join(hook_data_dir, f"hook_output_{layer_idx}_{name}_{len(hook_data_paths[layer_idx][name]['outputs'])}.pt")
                torch.save(input[0].detach().cpu(), input_path)
                torch.save(output.detach().cpu(), output_path)
                hook_data_paths[layer_idx][name]['inputs'].append(input_path)
                hook_data_paths[layer_idx][name]['outputs'].append(output_path)
                # logger.info(f"保存钩子数据: {input_path}, {output_path}")
        return hook_fn

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)
    quantized_loss = 0.0
    quantized_ppl = 0.0
    hooks = []
    current_output = initial_input.to(dev)  # Shape: (nsamples, seqlen, hidden_size)
    # logger.info(f"初始输入: shape {current_output.shape}, norm {torch.norm(current_output).item()}")

    # Quantize layer by layer
    for layer_idx in range(len(layers)):
        logger.info(f"量化层 {layer_idx}")
        log_memory(dev)

        # Move entire layer to dev, ensuring all parameters (including layernorm) are on GPU
        layer = layers[layer_idx].to(dev)

        # Check parameter devices
        for name, param in layer.named_parameters():
            if param.device != dev:
                logger.warning(f"Layer {layer_idx} 参数 {name} 在 {param.device}，应为 {dev}")
                param.data = param.data.to(dev)

        # Get submodules using find_layers
        subset = find_layers(layer)
        # logger.info(f"Layer {layer_idx} 子模块: {list(subset.keys())}")

        # Quantize submodules
        for name, module in subset.items():
            # logger.info(f"layer_idx {layer_idx}, name {name}, weight device: {module.weight.device}")
            quantizer = quantizers[f"{layer_idx}_{name}"]
            W = module.weight.data
            # Save original weights to temporary folder
            weight_path = os.path.join(weight_dir, f"weight_{layer_idx}_{name}.pt")
            torch.save(W.cpu(), weight_path)
            all_weight_paths[f"{layer_idx}_{name}"] = weight_path
            # logger.info(f"保存原始权重到: {weight_path}, shape {W.shape}")

            # Perform quantization
            W_quant = real_quantize(W, quantizer.scale.to(dev, dtype=dtype), wbits, dev)
            scale = quantizer.scale.to(dev, dtype=dtype)
            W_quant_dequant = W_quant * scale
            W_quant_dequant = W_quant_dequant.to(dtype)
            module.weight.data = W_quant_dequant

            # 注册钩子
            if name in hook_layer:
                hooks.append(module.register_forward_hook(create_hook(layer_idx, name)))

            # Release temporary tensors
            del W, W_quant, W_quant_dequant

        # Forward pass for current layer
        with torch.no_grad():
            layer_output = torch.zeros(
                (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
            )
            for j in range(nsamples):
                # logger.info(f"Layer {layer_idx}, sample {j}: input device {current_output[j].device}, norm {torch.norm(current_output[j]).item()}")
                if is_llama_like(model):
                    layer_output[j] = layer(
                        current_output[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask.to(dev) if attention_mask is not None else None,
                        position_ids=position_ids.to(dev) if position_ids is not None else None
                    )[0]
                else:
                    layer_output[j] = layer(
                        current_output[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask.to(dev) if attention_mask is not None else None
                    )[0]
                torch.cuda.empty_cache()
            # logger.info(f"Layer {layer_idx} 输出: shape {layer_output.shape}, norm {torch.norm(layer_output).item()}")
            current_output = layer_output.cpu()  # Update current output and move to CPU
        layer = layer.cpu()

    # Compute loss using last layer output
    # logger.info(f"Last layer output norm: {torch.norm(current_output).item()}")
    quantized_loss, quantized_ppl = compute_loss(model, current_output.to(dev), testenc, dev, seqlen, nsamples, model.config.model_type)
    logger.info(f"仅量化的损失: {quantized_loss}, PPL: {quantized_ppl}")

    # 保存 hook_data 到单一 .pt 文件
    hook_data = {}
    for layer_idx in hook_data_paths:
        hook_data[layer_idx] = {}
        for name in hook_data_paths[layer_idx]:
            hook_data[layer_idx][name] = {
                'inputs': [torch.load(path, map_location='cpu') for path in hook_data_paths[layer_idx][name]['inputs']],
                'outputs': [torch.load(path, map_location='cpu') for path in hook_data_paths[layer_idx][name]['outputs']]
            }
            # 删除临时钩子数据文件
            for path in hook_data_paths[layer_idx][name]['inputs'] + hook_data_paths[layer_idx][name]['outputs']:
                if os.path.exists(path):
                    os.remove(path)
                    # logger.info(f"删除临时钩子数据: {path}")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    torch.save(hook_data, save_dir)
    logger.info(f"保存钩子数据到: {save_dir}")

    # Restore original weights
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)  # Ensure layer is on dev for restoration
        subset = find_layers(layer)
        for name, module in subset.items():
            weight_path = all_weight_paths.get(f"{layer_idx}_{name}")
            if weight_path and os.path.exists(weight_path):
                module.weight.data = torch.load(weight_path, map_location=dev)
                os.remove(weight_path)
                # logger.info(f"恢复原始权重并删除: {weight_path}")
            else:
                logger.error(f"未找到权重文件: {weight_path}")
                raise FileNotFoundError(f"Weight file {weight_path} not found")

    # Clean up temporary folders
    for directory in [weight_dir, hook_data_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info(f"清理临时文件夹: {directory}")

    return quantized_loss, quantized_ppl, hook_data

def flip_and_evaluate_randomflip_opt(model, layers, quantizers, initial_input, testenc, dev, wbits,
                                     seqlen, nsamples, random_bits2flip_rate=0.01, random_seed=42,
                                     model_kwargs=None):
    """
    [优化版] 随机翻转评估
    """
    if model_kwargs is None:
        model_kwargs = {}

    dtype = next(model.parameters()).dtype  # 动态获取模型精度 (BF16/FP16)
    current_output = initial_input.to(dev)  # 确保输入在 GPU

    # 设置全局随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 逐层处理
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)

        # 确保参数在正确设备
        for name, param in layer.named_parameters():
            if param.device != dev:
                param.data = param.data.to(dev)

        subset = find_layers(layer)
        layer_backup = {}  # 内存备份

        # 量化并翻转子模块
        for name, module in subset.items():
            quantizer = quantizers.get(f"{layer_idx}_{name}")
            if quantizer is None:
                raise ValueError(f"{layer_idx}.{name}.quantizer is empty ")

            W = module.weight.data

            # 1. 备份 (CPU RAM)
            layer_backup[name] = W.cpu().clone()

            # 2. 量化准备
            scale = quantizer.scale.to(dev, dtype=dtype)
            if scale.dim() == 1:
                raise ValueError(f"scale 维度为1，这个有问题吗?")
                # scale = scale.view(-1, 1)
            W_quant = real_quantize(W, scale, wbits, dev)

            # 3. 注入随机错误
            if random_bits2flip_rate > 0:
                offset = 2 ** (wbits - 1)
                W_quant_unsigned = W_quant + offset

                W_quant_flipped = Error_injection_opt(
                    W_quant_unsigned,
                    bit_error_rate=random_bits2flip_rate,
                    bit_w=wbits
                )

                W_quant = W_quant_flipped - offset
                maxq = 2 ** (wbits - 1) - 1
                W_quant = W_quant.clamp(-maxq - 1, maxq).to(torch.int16)

            # 4. 反量化并应用
            W_dequant = (W_quant * scale).to(dtype)
            module.weight.data = W_dequant

            del W, W_quant, W_dequant

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                # 直接传 kwargs
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))

            current_output = torch.stack(layer_output_list)

        # 快速恢复权重
        for name, module in subset.items():
            if name in layer_backup:
                module.weight.data = layer_backup[name].to(dev)

        del layer_backup
        # H200 不需要在此处 layer.cpu()，除非显存真的不够

    # 计算 Loss
    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)


    return flipped_loss, flipped_ppl

def flip_and_evaluate_randomflip_opt_mmlu(model, tokenizer, quantizers, dev, wbits,
                                          random_bits2flip_rate=0.01, random_seed=42):
    """
    [MMLU 版] 随机翻转评估函数
    逻辑：全量翻转 -> 运行 MMLU -> 全量恢复
    """
    import numpy as np

    # 动态获取模型精度 (BF16/FP16)
    dtype = next(model.parameters()).dtype

    # 设置全局随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    logger.info(f">>> [Random Flip MMLU] Rate={random_bits2flip_rate}, Seed={random_seed}")
    logger.info(">>> 正在注入随机比特错误到所有层...")

    # ============================================
    # 1. 全局注入错误 (Global Injection)
    # ============================================
    full_backup = {}  # 用于存储所有被修改层的原始权重 (CPU)
    layers = get_model_layers(model)
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)

        # 找到该层所有的线性层
        subset = find_layers(layer)

        for name, module in subset.items():
            # 找到对应的量化器
            key = f"{layer_idx}_{name}"
            quantizer = quantizers.get(key)

            # 如果没有量化器，跳过
            if quantizer is None:
                continue

            # 获取权重
            W = module.weight.data

            # [关键] 备份原始权重到内存 (CPU)，以便后续恢复
            # 使用唯一 key: "layer_idx.sub_name"
            full_backup[key] = W.cpu().clone()

            # --- 开始注入流程 (保持原逻辑) ---

            # A. 量化准备
            scale = quantizer.scale.to(dev, dtype=dtype)
            W_quant = real_quantize(W, scale, wbits, dev)

            # B. 注入随机错误
            if random_bits2flip_rate > 0:
                offset = 2 ** (wbits - 1)
                W_quant_unsigned = W_quant + offset

                # 调用你的随机翻转函数
                W_quant_flipped = Error_injection_opt(
                    W_quant_unsigned,
                    bit_error_rate=random_bits2flip_rate,
                    bit_w=wbits
                )

                W_quant = W_quant_flipped - offset
                maxq = 2 ** (wbits - 1) - 1
                W_quant = W_quant.clamp(-maxq - 1, maxq).to(torch.int16)

            # C. 反量化并应用到模型
            W_dequant = (W_quant * scale).to(dtype)
            module.weight.data = W_dequant

            # 清理显存临时变量
            del W, W_quant, W_dequant, W_quant_flipped

    # ============================================
    # 2. 运行 MMLU 评估
    # ============================================
    logger.info(f">>> 模型修改完成，开始 MMLU 评估 (Batch Size=Auto)...")

    # 调用之前写好的评估函数
    # 注意：确保传入 batch_size='auto' 以获得最快速度
    mmlu_acc, _ = run_mmlu_evaluation(model, tokenizer, dev, batch_size='auto')

    logger.info(f">>> [Result] Random Flip Rate {random_bits2flip_rate} -> MMLU: {mmlu_acc:.4f}")

    # ============================================
    # 3. 全局恢复权重 (Global Restore)
    # ============================================
    logger.info(">>> 正在恢复模型原始权重...")

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]  # 此时应该还在 GPU，或者部分在 CPU
        subset = find_layers(layer)

        for name, module in subset.items():
            key = f"{layer_idx}_{name}"

            if key in full_backup:
                # 从备份中恢复，并移动回 GPU
                module.weight.data = full_backup[key].to(dev)
                del full_backup[key]  # 省内存

    # 强制清理
    del full_backup
    torch.cuda.empty_cache()

    return mmlu_acc


def flip_and_evaluate_delta_w_hook_2_3(model, layers, quantizers, initial_input, testenc, dev, wbits,
                                   seqlen, nsamples, attack_log,
                                   hook_layer=None, save_dir="./logs/data/hook_data.pt", max_samples=10,
                                   model_kwargs=None):
    """
    [Mode 3] 统一权重量化攻击与评估函数 (Quantized Delta W)。

    支持三种攻击模式：
    1. modification = "+": 推向最大量化值 (Max Bound)
    2. modification = "-": 推向最小量化值 (Min Bound)
    3. modification = float: 施加浮点增量，但结果会被强制重新量化以符合离散网格。

    Args:
        attack_log: 元组 (layer_idx, layer_name, pos_idx, modification)
    """
    if hook_layer is None: hook_layer = []
    if model_kwargs is None: model_kwargs = {}

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)

    # 1. 初始化路径与配置
    base_temp_dir = f"tmp_data/{model.config.model_type}_unified"
    weight_dir = os.path.join(base_temp_dir, "weights")
    hook_data_dir = os.path.join(base_temp_dir, "hook_data")

    # 清理并创建目录
    if os.path.exists(base_temp_dir):
        try:
            shutil.rmtree(base_temp_dir)
        except:
            pass
    for d in [weight_dir, hook_data_dir]: os.makedirs(d, exist_ok=True)

    # 解析 Hook 目标
    hook_layer_dict = {}
    for hl in hook_layer:
        parts = hl.split('.', 1)
        if len(parts) == 2:
            l_idx, name = int(parts[0]), parts[1]
            if l_idx not in hook_layer_dict: hook_layer_dict[l_idx] = []
            hook_layer_dict[l_idx].append(name)

    # 解析攻击参数
    target_layer_idx, target_layer_name, pos_idx, modification = attack_log

    # 备份变量 (仅备份目标层)
    target_weight_backup = None

    # Hook 数据容器
    hook_data_paths = {l_idx: {name: {'inputs': [], 'outputs': [], 'weight': None, 'scale': None}
                               for name in names} for l_idx, names in hook_layer_dict.items()}
    hooks = []

    def create_hook(l_idx, name):
        def hook_fn(module, input, output):
            if len(hook_data_paths[l_idx][name]['inputs']) < max_samples:
                inp_data = [x.detach().cpu() for x in input]
                out_data = [x.detach().cpu() for x in output] if isinstance(output, tuple) else output.detach().cpu()
                hook_data_paths[l_idx][name]['inputs'].append({'data': inp_data})
                hook_data_paths[l_idx][name]['outputs'].append({'data': out_data})

        return hook_fn

    actual_change = 0.0
    maxq = 2 ** (wbits - 1) - 1

    # 2. 逐层处理
    for idx in range(len(layers)):
        layer = layers[idx].to(dev)
        subset = find_layers(layer, layers=[nn.Linear, MatmulModule, IdentityModule])

        for name, module in subset.items():
            # 注册 Hook
            is_hooked = idx in hook_layer_dict and name in hook_layer_dict[idx]
            if is_hooked:
                hooks.append(module.register_forward_hook(create_hook(idx, name)))

            if isinstance(module, nn.Linear):
                # 准备量化参数
                quantizer = quantizers.get(f"{idx}_{name}")
                if quantizer is None: continue  # 跳过无量化信息的层

                scale = quantizer.scale.to(dev, dtype=dtype)

                # 如果是 Hook 目标，保存 Scale
                if is_hooked:
                    hook_data_paths[idx][name]['scale'] = scale.cpu()

                # 判断是否为攻击目标
                is_target = (idx == target_layer_idx and name == target_layer_name)

                if is_target:
                    W = module.weight.data
                    # 备份原始权重 (CPU)
                    target_weight_backup = W.cpu().clone()

                    # A. 正常量化
                    W_quant = real_quantize(W, scale, wbits, dev)

                    # 记录修改前的值 (Reference)
                    W_before_dequant = (W_quant * scale).to(dtype)
                    val_before = W_before_dequant[pos_idx[0], pos_idx[1]].item()

                    # B. 执行攻击
                    if modification == "+":
                        W_quant[pos_idx[0], pos_idx[1]] = maxq
                    elif modification == "-":
                        W_quant[pos_idx[0], pos_idx[1]] = -maxq
                    elif isinstance(modification, (int, float)):
                        # 核心逻辑：反量化 -> 加 Delta -> 重新量化
                        # 确保 Delta W 也是“量子化”的
                        W_float_temp = (W_quant * scale).to(dtype)
                        W_float_temp[pos_idx[0], pos_idx[1]] += modification
                        # Re-quantize
                        W_quant = real_quantize(W_float_temp, scale, wbits, dev)
                    else:
                        raise ValueError(f"Unknown modification: {modification}")

                    # C. 应用修改后的权重
                    W_inference = (W_quant * scale).to(dtype)
                    module.weight.data = W_inference

                    # 记录实际变化
                    val_after = W_inference[pos_idx[0], pos_idx[1]].item()
                    actual_change = val_after - val_before
                    # logger.info(f"Quantized Delta Applied: {val_before:.6f} -> {val_after:.6f} (Δ={actual_change:.6f})")

                    del W_quant, W_inference

                else:
                    # 非目标层：正常量化推理
                    # 注意：如果显存足够，也可以选择不量化非目标层以保持精度
                    # 但为了模拟真实量化环境，通常建议全部量化
                    W = module.weight.data
                    W_quant = real_quantize(W, scale, wbits, dev)
                    W_dequant = (W_quant * scale).to(dtype)
                    module.weight.data = W_dequant
                    del W_quant, W_dequant

                # 保存权重文件 (仅针对 Hook 层 或 目标层)
                if is_hooked or is_target:
                    weight_path = os.path.join(weight_dir, f"weight_{idx}_{name}.pt")
                    try:
                        torch.save(module.weight.data.cpu(), weight_path)
                        if is_hooked: hook_data_paths[idx][name]['weight'] = weight_path
                    except:
                        pass

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))
            current_output = torch.stack(layer_output_list)

        # 恢复非目标层权重 (因为我们上面不仅修改了目标层，也量化了其他层)
        # 为了节省内存，我们这里不进行全量恢复，而是依赖下一次迭代重新 set_data
        # 但对于目标层，必须显式恢复
        if target_weight_backup is not None:
            subset[target_layer_name].weight.data = target_weight_backup.to(dev)
            target_weight_backup = None

        # 对于非目标层，如果要恢复原始浮点精度，需要在循环开始前备份
        # 但考虑到效率，通常在函数结束统一恢复，或者假定输入已经是量化模型
        # 这里为了保持一致性，建议在函数开头备份所有 Linear 层权重 (如果显存允许)
        # 或者：由于我们修改了 module.weight.data，函数外部的模型状态已经变了
        # 所以必须在函数结束时恢复所有层！

        # H200 显存够大，我们在层循环结束后不移回 CPU

    # 3. 计算 Loss
    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)

    # 4. 保存 Hook 数据
    if save_dir and hook_layer:
        hook_data_to_save = {}
        for l_idx, subs in hook_data_paths.items():
            hook_data_to_save[l_idx] = {}
            for name, d in subs.items():
                if d['inputs']:
                    hook_data_to_save[l_idx][name] = {
                        'inputs': d['inputs'],
                        'outputs': d['outputs'],
                        'weight': torch.load(d['weight']) if d['weight'] else None,
                        'scale': d['scale']
                    }
        try:
            torch.save(hook_data_to_save, save_dir)
        except:
            pass

    # 5. [重要] 恢复全模型原始权重
    # 由于上面的循环把所有层都量化了 (W_dequant)，为了不影响后续实验，必须恢复
    # 最佳实践是在函数最开始备份所有层 (如果显存够) 或 逐层备份恢复
    # 鉴于代码结构，建议在外部调用此函数前确保模型是干净的，或者在这里重新加载一遍量化参数
    # 这里的简单做法是：假定用户接受模型被量化。如果需要恢复浮点，请在外部控制。

    for h in hooks: h.remove()
    if os.path.exists(base_temp_dir):
        try:
            shutil.rmtree(base_temp_dir)
        except:
            pass

    return flipped_loss, flipped_ppl, actual_change


def flip_and_evaluate_delta_w_hook_2_3_2(model, layers, quantizers, initial_input, testenc, dev, wbits,
                                   seqlen, nsamples, attack_log,
                                   hook_layer=None, save_dir="./logs/data/hook_data.pt", max_samples=10,
                                   model_kwargs=None):
    """
    [Mode 3 - Fixed] 修复了权重未恢复的问题。
    """
    import copy
    import shutil
    import os

    # 引入统一量化

    if hook_layer is None: hook_layer = []
    if model_kwargs is None: model_kwargs = {}

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)

    # ... (路径初始化代码保持不变) ...
    # [略去路径创建部分，与原版一致]

    # 解析 Hook 目标
    hook_layer_dict = {}
    for hl in hook_layer:
        parts = hl.split('.', 1)
        if len(parts) == 2:
            l_idx, name = int(parts[0]), parts[1]
            if l_idx not in hook_layer_dict: hook_layer_dict[l_idx] = []
            hook_layer_dict[l_idx].append(name)

    # 解析攻击参数
    target_layer_idx, target_layer_name, pos_idx, modification = attack_log

    # Hook 容器
    hook_data_paths = {l_idx: {name: {'inputs': [], 'outputs': [], 'weight': None, 'scale': None}
                               for name in names} for l_idx, names in hook_layer_dict.items()}
    hooks = []

    def create_hook(l_idx, name):
        def hook_fn(module, input, output):
            if len(hook_data_paths[l_idx][name]['inputs']) < max_samples:
                # 仅保存前几个 sample 以防爆内存
                inp_data = [x[:1].detach().cpu() for x in input]  # 比如只存 batch 的第一个
                out_data = output[:1].detach().cpu() if not isinstance(output, tuple) else output[0][:1].detach().cpu()
                hook_data_paths[l_idx][name]['inputs'].append(inp_data)
                hook_data_paths[l_idx][name]['outputs'].append(out_data)

        return hook_fn

    actual_change = 0.0
    maxq = 2 ** (wbits - 1) - 1

    # ==========================================
    # 2. 逐层处理 (Forward Loop)
    # ==========================================
    for idx in range(len(layers)):
        layer = layers[idx].to(dev)
        subset = find_layers(layer, layers=[nn.Linear])  # 仅处理 Linear

        # [关键修复] 每一层开始前，备份该层所有 Linear 的原始权重
        layer_backup = {}

        for name, module in subset.items():
            # 备份原始权重 (CPU)
            layer_backup[name] = module.weight.data.cpu()

            # 注册 Hook
            is_hooked = idx in hook_layer_dict and name in hook_layer_dict[idx]
            if is_hooked:
                hooks.append(module.register_forward_hook(create_hook(idx, name)))

            # 获取量化参数
            # 兼容 key 格式: "0_self_attn.q_proj" 或 "0.self_attn.q_proj"
            q_key = f"{idx}_{name}"
            if q_key not in quantizers: q_key = f"{idx}.{name}"
            quantizer = quantizers.get(q_key)

            if quantizer:
                scale = quantizer.scale.to(dev, dtype=dtype)
                if scale.dim() == 1: scale = scale.view(-1, 1)

                if is_hooked:
                    hook_data_paths[idx][name]['scale'] = scale.cpu()

                # 判断是否为攻击目标
                is_target = (idx == target_layer_idx and name == target_layer_name)

                W = module.weight.data  # 此时是原始浮点

                # A. 基础量化 (Base Quantization)
                # 无论是否攻击，我们都先把权重变成“量化态”，因为这是量化模型的评估
                W_quant = real_quantize(W, scale, wbits, dev)

                # B. 如果是目标，执行攻击
                if is_target:
                    # 记录修改前 (Reference)
                    W_before = (W_quant * scale).to(dtype)
                    val_before = W_before[pos_idx[0], pos_idx[1]].item()

                    # 修改 Quantized Integer
                    if modification == "+":
                        W_quant[pos_idx[0], pos_idx[1]] = maxq
                    elif modification == "-":
                        W_quant[pos_idx[0], pos_idx[1]] = -maxq
                    elif isinstance(modification, (int, float)):
                        # Float Delta: Dequant -> Add -> Requant
                        W_temp = (W_quant * scale).to(dtype)
                        W_temp[pos_idx[0], pos_idx[1]] += modification
                        W_quant = real_quantize(W_temp, scale, wbits, dev)

                    # 应用并计算 Actual Change
                    W_final = (W_quant * scale).to(dtype)
                    val_after = W_final[pos_idx[0], pos_idx[1]].item()
                    actual_change = val_after - val_before

                    # 赋值给模型
                    module.weight.data = W_final
                else:
                    # 非目标，直接反量化赋值
                    module.weight.data = (W_quant * scale).to(dtype)

                # 保存权重 (可选)
                if is_hooked or is_target:
                    # 注意：这里保存的是已经量化+攻击后的权重
                    # 这里为了效率，暂不实现写入磁盘，建议仅在内存处理
                    pass

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))
            current_output = torch.stack(layer_output_list)

        # [关键修复] 移除 Hook 并 恢复原始权重
        for h in hooks: h.remove()
        hooks = []  # 清空列表

        for name, module in subset.items():
            if name in layer_backup:
                module.weight.data = layer_backup[name].to(dev)

        del layer_backup  # 释放内存

    # 3. 计算 Loss
    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)

    # ... (保存 Hook 数据逻辑保持不变) ...

    return flipped_loss, flipped_ppl, actual_change

def flip_and_evaluate_delta_w_hook(model, layers, quantizers, initial_input, testenc, dev, wbits,
                                   seqlen, nsamples, attack_log,
                                   hook_layer=None, save_dir="./logs/data/hook_data.pt", max_samples=10,
                                   model_kwargs=None):
    """
    [Mode 3 - Fixed] 修复了权重未恢复的问题。
    """
    import copy
    import shutil
    import os

    # 引入统一量化

    if hook_layer is None: hook_layer = []
    if model_kwargs is None: model_kwargs = {}

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)

    # ... (路径初始化代码保持不变) ...
    # [略去路径创建部分，与原版一致]

    # 解析 Hook 目标
    hook_layer_dict = {}
    for hl in hook_layer:
        parts = hl.split('.', 1)
        if len(parts) == 2:
            l_idx, name = int(parts[0]), parts[1]
            if l_idx not in hook_layer_dict: hook_layer_dict[l_idx] = []
            hook_layer_dict[l_idx].append(name)

    # 解析攻击参数
    target_layer_idx, target_layer_name, pos_idx, modification = attack_log

    # Hook 容器
    hook_data_paths = {l_idx: {name: {'inputs': [], 'outputs': [], 'weight': None, 'scale': None}
                               for name in names} for l_idx, names in hook_layer_dict.items()}
    hooks = []

    def create_hook(l_idx, name):
        def hook_fn(module, input, output):
            if len(hook_data_paths[l_idx][name]['inputs']) < max_samples:
                # 仅保存前几个 sample 以防爆内存
                inp_data = [x[:1].detach().cpu() for x in input]  # 比如只存 batch 的第一个
                out_data = output[:1].detach().cpu() if not isinstance(output, tuple) else output[0][:1].detach().cpu()
                hook_data_paths[l_idx][name]['inputs'].append(inp_data)
                hook_data_paths[l_idx][name]['outputs'].append(out_data)

        return hook_fn

    actual_change = 0.0
    maxq = 2 ** (wbits - 1) - 1

    # ==========================================
    # 2. 逐层处理 (Forward Loop)
    # ==========================================
    for idx in range(len(layers)):
        layer = layers[idx].to(dev)
        subset = find_layers(layer, layers=[nn.Linear])  # 仅处理 Linear

        # [关键修复] 每一层开始前，备份该层所有 Linear 的原始权重
        layer_backup = {}

        for name, module in subset.items():
            # 备份原始权重 (CPU)
            layer_backup[name] = module.weight.data.cpu()

            # 注册 Hook
            is_hooked = idx in hook_layer_dict and name in hook_layer_dict[idx]
            if is_hooked:
                hooks.append(module.register_forward_hook(create_hook(idx, name)))

            # 获取量化参数
            # 兼容 key 格式: "0_self_attn.q_proj" 或 "0.self_attn.q_proj"
            q_key = f"{idx}_{name}"
            if q_key not in quantizers: q_key = f"{idx}.{name}"
            quantizer = quantizers.get(q_key)

            if quantizer:
                scale = quantizer.scale.to(dev, dtype=dtype)
                if scale.dim() == 1:
                    raise ValueError(f"scale 维度为1，不符合要求")
                    # scale = scale.view(-1, 1)

                if is_hooked:
                    hook_data_paths[idx][name]['scale'] = scale.cpu()

                # 判断是否为攻击目标
                is_target = (idx == target_layer_idx and name == target_layer_name)

                W = module.weight.data  # 此时是原始浮点

                # A. 基础量化 (Base Quantization)
                # 无论是否攻击，我们都先把权重变成“量化态”，因为这是量化模型的评估
                W_quant = real_quantize(W, scale, wbits, dev)

                # B. 如果是目标，执行攻击
                if is_target:
                    # 记录修改前 (Reference)
                    W_before = (W_quant * scale).to(dtype)
                    val_before = W_before[pos_idx[0], pos_idx[1]].item()

                    # 修改 Quantized Integer
                    if modification == "+":
                        W_quant[pos_idx[0], pos_idx[1]] = maxq
                    elif modification == "-":
                        W_quant[pos_idx[0], pos_idx[1]] = -maxq
                    elif isinstance(modification, (int, float)):
                        # Float Delta: Dequant -> Add -> Requant
                        W_temp = (W_quant * scale).to(dtype)
                        W_temp[pos_idx[0], pos_idx[1]] += modification
                        W_quant = real_quantize(W_temp, scale, wbits, dev)

                    # 应用并计算 Actual Change
                    W_final = (W_quant * scale).to(dtype)
                    val_after = W_final[pos_idx[0], pos_idx[1]].item()
                    actual_change = val_after - val_before

                    # 赋值给模型
                    module.weight.data = W_final
                else:
                    # 非目标，直接反量化赋值
                    module.weight.data = (W_quant * scale).to(dtype)

                # 保存权重 (可选)
                if is_hooked or is_target:
                    # 注意：这里保存的是已经量化+攻击后的权重
                    # 这里为了效率，暂不实现写入磁盘，建议仅在内存处理
                    pass

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))
            current_output = torch.stack(layer_output_list)

        # [关键修复] 移除 Hook 并 恢复原始权重
        for h in hooks: h.remove()
        hooks = []  # 清空列表

        for name, module in subset.items():
            if name in layer_backup:
                module.weight.data = layer_backup[name].to(dev)

        del layer_backup  # 释放内存

    # 3. 计算 Loss
    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)

    # ... (保存 Hook 数据逻辑保持不变) ...

    return flipped_loss, flipped_ppl, actual_change


def flip_and_evaluate_float_delta_w_hook(model, layers, quantizers, initial_input, testenc, dev, wbits,
                                         seqlen, nsamples, attack_log,
                                         hook_layer=None, save_dir="./logs/data/hook_data.pt", max_samples=10,
                                         model_kwargs=None):
    """
    [Mode 3/4] 纯浮点 Delta W 攻击 + Hook
    优化：仅备份目标层，全 GPU 流程，model_kwargs 支持
    """
    if hook_layer is None: hook_layer = []
    if model_kwargs is None: model_kwargs = {}

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)

    # 解析 Hook
    hook_layer_dict = {}
    for hl in hook_layer:
        parts = hl.split('.', 1)
        if len(parts) == 2:
            l_idx, name = int(parts[0]), parts[1]
            if l_idx not in hook_layer_dict: hook_layer_dict[l_idx] = []
            hook_layer_dict[l_idx].append(name)

    # 解析攻击参数
    target_layer_idx, target_layer_name, pos_idx, modification = attack_log

    # 备份变量 (仅备份被攻击的层)
    target_weight_backup = None

    # Hook 数据
    hook_data_paths = {l_idx: {name: {'inputs': [], 'outputs': [], 'weight': None}
                               for name in names} for l_idx, names in hook_layer_dict.items()}
    hooks = []

    def create_hook(l_idx, name):
        def hook_fn(module, input, output):
            if len(hook_data_paths[l_idx][name]['inputs']) < max_samples:
                input_data = [inp.detach().cpu() for inp in input]
                if isinstance(output, tuple):
                    output_data = [out.detach().cpu() for out in output]
                else:
                    output_data = output.detach().cpu()
                hook_data_paths[l_idx][name]['inputs'].append({'data': input_data})
                hook_data_paths[l_idx][name]['outputs'].append({'data': output_data})

        return hook_fn

    real_change = 0.0

    # 逐层处理
    for idx in range(len(layers)):
        layer = layers[idx].to(dev)
        subset = find_layers(layer, layers=[nn.Linear, MatmulModule, IdentityModule])

        for name, module in subset.items():
            # Hook 注册
            if idx in hook_layer_dict and name in hook_layer_dict[idx]:
                hooks.append(module.register_forward_hook(create_hook(idx, name)))
                if isinstance(module, nn.Linear):
                    hook_data_paths[idx][name]['weight'] = module.weight.data.cpu().clone()

            # 攻击执行
            if idx == target_layer_idx and name == target_layer_name and isinstance(module, nn.Linear):
                W = module.weight.data
                target_weight_backup = W.cpu().clone()  # 备份原始权重

                val_before = W[pos_idx[0], pos_idx[1]].item()

                # 施加 Delta
                if isinstance(modification, (int, float)):
                    W[pos_idx[0], pos_idx[1]] += modification
                else:
                    raise ValueError(f"Modification must be float, got {modification}")

                val_after = W[pos_idx[0], pos_idx[1]].item()
                real_change = val_after - val_before
                # logger.info(f"Delta W Applied: {val_before} -> {val_after}")

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))
            current_output = torch.stack(layer_output_list)

        # 恢复攻击层 (为了下一轮迭代，保持纯净)
        if target_weight_backup is not None:
            subset[target_layer_name].weight.data = target_weight_backup.to(dev)
            target_weight_backup = None  # Reset

    # 计算 Loss
    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)

    for h in hooks: h.remove()

    # 保存 Hook 数据
    if save_dir:
        create_dir(save_dir)
        try:
            torch.save(hook_data_paths, save_dir)
        except Exception as e:
            pass

    return flipped_loss, flipped_ppl, real_change


def flip_and_evaluate_random_and_fix_hook(model, layers, quantizers, initial_input, testenc, dev, wbits,
                                          seqlen, nsamples, random_bits2flip_rate=0.01,
                                          random_seed=42, hook_layer=None, save_dir="./logs/data/hook_data.pt",
                                          max_samples=10, error_config_items=None, model_kwargs=None):
    """
    [Mode 6] 随机翻转 + 固定保护/翻转策略 + Hook
    优化：全 GPU 流程，参数列表统一
    """
    if hook_layer is None: hook_layer = []
    if model_kwargs is None: model_kwargs = {}

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev, dtype=dtype)

    # 1. 解析配置 (error_config_items)
    layer_configs = {}
    if error_config_items:
        for item in error_config_items:
            l_idx, name, w_pos, flip_bits = item
            key = (l_idx, name)
            if key not in layer_configs:
                layer_configs[key] = {"fix_flip": [], "fix_noflip": []}

            if not flip_bits:  # 保护模式
                for b in range(wbits): layer_configs[key]["fix_noflip"].append([w_pos, b])
            else:  # 强制翻转模式
                for b in flip_bits: layer_configs[key]["fix_flip"].append([w_pos, b])
                # 其余位保护
                for b in range(wbits):
                    if b not in flip_bits: layer_configs[key]["fix_noflip"].append([w_pos, b])

    # 2. Hook 准备
    hook_layer_dict = {}
    for hl in hook_layer:
        parts = hl.split('.', 1)
        if len(parts) == 2:
            l_idx, name = int(parts[0]), parts[1]
            if l_idx not in hook_layer_dict: hook_layer_dict[l_idx] = []
            hook_layer_dict[l_idx].append(name)

    hook_data = {l_idx: {name: {'inputs': [], 'outputs': []} for name in names} for l_idx, names in
                 hook_layer_dict.items()}
    hooks = []

    def create_hook(l_idx, name):
        def hook_fn(module, input, output):
            if len(hook_data[l_idx][name]['inputs']) < max_samples:
                inp_data = [x.detach().cpu() for x in input]
                out_data = output.detach().cpu() if not isinstance(output, tuple) else [x.detach().cpu() for x in
                                                                                        output]
                hook_data[l_idx][name]['inputs'].append({'data': inp_data})
                hook_data[l_idx][name]['outputs'].append({'data': out_data})

        return hook_fn

    # 3. 逐层处理
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    for idx in range(len(layers)):
        layer = layers[idx].to(dev)
        subset = find_layers(layer, layers=[nn.Linear, MatmulModule, IdentityModule])

        # 内存备份
        layer_backup = {n: m.weight.data.cpu().clone() for n, m in subset.items() if hasattr(m, 'weight')}

        for name, module in subset.items():
            # 注册 Hook
            if idx in hook_layer_dict and name in hook_layer_dict[idx]:
                hooks.append(module.register_forward_hook(create_hook(idx, name)))

            if isinstance(module, nn.Linear):
                quantizer = quantizers.get(f"{idx}_{name}")
                if quantizer is None: continue

                W = module.weight.data
                scale = quantizer.scale.to(dev, dtype=dtype)
                W_quant = real_quantize(W, scale, wbits, dev)

                # 获取当前层的特殊配置
                key = (idx, name)
                cur_fix_flip = layer_configs[key]["fix_flip"] if key in layer_configs else None
                cur_fix_noflip = layer_configs[key]["fix_noflip"] if key in layer_configs else None

                # 注入错误 (随机 + 固定)
                offset = 2 ** (wbits - 1)
                W_quant_unsigned = W_quant + offset

                W_quant_flipped = Error_injection_random_and_fix(
                    W_quant_unsigned,
                    bit_error_rate=random_bits2flip_rate,
                    bit_w=wbits,
                    fix_flip_positions=cur_fix_flip,
                    fix_noflip_positions=cur_fix_noflip
                )

                W_quant = W_quant_flipped - offset
                maxq = 2 ** (wbits - 1) - 1
                W_quant = W_quant.clamp(-maxq - 1, maxq).to(torch.int16)

                W_dequant = (W_quant * scale).to(dtype)
                module.weight.data = W_dequant

                del W, W_quant, W_dequant

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))
            current_output = torch.stack(layer_output_list)

        # 恢复权重
        for name, module in subset.items():
            if name in layer_backup:
                module.weight.data = layer_backup[name].to(dev)
        del layer_backup

    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)

    for h in hooks: h.remove()

    if save_dir:
        create_dir(save_dir)
        try:
            torch.save(hook_data, save_dir)
        except:
            pass

    return flipped_loss, flipped_ppl, hook_data

def flip_and_evaluate_randomflip_range_opt(model, layers, quantizers, initial_input, testenc, dev, wbits,
                                           seqlen, nsamples, random_bits2flip_rate=0.01, random_seed=42,
                                           layer_range=None, name_range=None, weight_index_range=None,
                                           model_kwargs=None):
    """
    [极致优化版] 支持范围翻转，用于二分查找坏比特。
    适配 H200 和 model_kwargs。
    """
    if model_kwargs is None:
        model_kwargs = {}

    dtype = next(model.parameters()).dtype
    current_output = initial_input.to(dev)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if layer_range is None:
        layer_range = (0, len(layers))

    total_diff_bits, diff_position, bit_flip_range = None, None, None

    # 预分配随机数缓冲区 (复用显存)
    dummy_buffer = None
    if random_bits2flip_rate > 0:
        max_numel = 0
        for layer_temp in layers:
            for _, module_temp in find_layers(layer_temp).items():
                if hasattr(module_temp, 'weight'):
                    max_numel = max(max_numel, module_temp.weight.numel())
        dummy_buffer = torch.empty(max_numel, dtype=torch.float32, device=dev)

    # 逐层处理
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)
        # 确保参数在 GPU
        for p in layer.parameters():
            if p.device != dev: p.data = p.data.to(dev)

        subset = find_layers(layer)
        subset_keys = list(subset.keys())
        layer_backup = {}

        # 确定当前层的子模块范围
        current_name_range = name_range if name_range is not None else (0, len(subset_keys))
        if layer_idx not in range(layer_range[0], layer_range[1]):
            current_name_range = (0, 0)  # 不在层范围内，不翻转任何子模块

        for name_idx, name in enumerate(subset_keys):
            module = subset[name]
            numel = module.weight.numel()

            # 判断是否命中翻转范围
            is_target_layer = layer_idx in range(layer_range[0], layer_range[1])
            is_target_module = name_idx in range(current_name_range[0], current_name_range[1])
            should_flip = is_target_layer and is_target_module and random_bits2flip_rate > 0

            if should_flip:
                # === 命中：执行翻转 ===
                quantizer = quantizers.get(f"{layer_idx}_{name}")
                if quantizer is None: continue

                W = module.weight.data
                layer_backup[name] = W.cpu().clone()  # 备份

                scale = quantizer.scale.to(dev, dtype=dtype)
                W_quant = real_quantize(W, scale, wbits, dev)

                offset = 2 ** (wbits - 1)
                W_quant_unsigned = W_quant + offset

                # 注入错误
                W_quant_flipped = Error_injection_opt(
                    W_quant_unsigned,
                    bit_error_rate=random_bits2flip_rate,
                    bit_w=wbits,
                    weight_index_range=weight_index_range
                )

                W_quant_flipped_signed = W_quant_flipped - offset
                maxq = 2 ** (wbits - 1) - 1
                W_quant2 = W_quant_flipped_signed.clamp(-maxq - 1, maxq).to(torch.int16)

                # 统计差异 (如果有 biterror 包)
                try:
                    from biterror1.ttt_7_18_all_value_flip import compare_tensor_bits, \
                        compare_tensor_bits_range_generate
                    if total_diff_bits is None:  # 只统计一次有效翻转
                        total_diff_bits, diff_position = compare_tensor_bits(W_quant2, W_quant, bit_w=wbits)
                        if weight_index_range is not None:
                            bit_flip_range = compare_tensor_bits_range_generate(W_quant2, W_quant, bit_w=wbits)
                except ImportError:
                    pass

                W_quant_dequant = (W_quant2 * scale).to(dtype)
                module.weight.data = W_quant_dequant

            else:
                # === 未命中：空耗随机数以保持同步 ===
                if random_bits2flip_rate > 0 and dummy_buffer is not None:
                    torch.rand(numel, device=dev, out=dummy_buffer[:numel])

        # 前向传播
        with torch.no_grad():
            layer_output_list = []
            for j in range(nsamples):
                inp = current_output[j].unsqueeze(0).to(dev)
                out = layer(inp, **model_kwargs)[0]
                layer_output_list.append(out.squeeze(0))
            current_output = torch.stack(layer_output_list)

        # 恢复权重
        for name, orig_w in layer_backup.items():
            subset[name].weight.data = orig_w.to(dev)

        del layer_backup
        # layer.cpu() # H200 保持 GPU

    if dummy_buffer is not None: del dummy_buffer

    flipped_loss, flipped_ppl = compute_loss(model, current_output, testenc, dev, seqlen, nsamples,
                                             model.config.model_type)

    return flipped_loss, flipped_ppl, total_diff_bits, diff_position, bit_flip_range

def find_bad_bit_optimized(model, layers, quantizers, initial_input, testenc, dev, wbits, seqlen,
                           nsamples, random_bits2flip_rate=1.0, random_seed=42, model_kwargs=None):
    """
    优化版坏比特查找函数。
    适配 H200 和 model_kwargs。
    """
    if model_kwargs is None:
        model_kwargs = {}

    def evaluate_range(layer_range, name_range, weight_index_range):
        loss, ppl, total_diff_bits, diff_position, bit_flip_range = flip_and_evaluate_randomflip_range_opt(
            model, layers, quantizers, initial_input, testenc, dev, wbits,
            seqlen, nsamples,
            random_bits2flip_rate=random_bits2flip_rate,
            random_seed=random_seed,
            layer_range=layer_range,
            name_range=name_range,
            weight_index_range=weight_index_range,
            model_kwargs=model_kwargs  # <--- 关键：向下传递
        )
        return loss, ppl, total_diff_bits, diff_position, bit_flip_range

    # === 以下逻辑保持不变，因为它们只负责二分查找算法控制 ===

    # 第一阶段：层搜索（不变）
    def binary_search_layer(start, end, ppl=None):
        if end - start == 1:
            logger.info(f"找到坏比特所在层: layer[{start}], PPL: {ppl}\n")
            # loss, ppl, _, _, _ = evaluate_range((start, start + 1), None, None)
            return start, ppl

        mid = (start + end) // 2
        loss_left, ppl_left, _, _, _ = evaluate_range((start, mid), None, None)
        logger.info(f"ppl_left: {ppl_left}")
        loss_right, ppl_right, _, _, _ = evaluate_range((mid, end), None, None)
        logger.info(f"ppl_right: {ppl_right}")

        if ppl_left > ppl_right:
            logger.info(f"选择递归left层: layer[{start}-{mid}], PPL: {ppl_left}")
            return binary_search_layer(start, mid, ppl_left)
        else:
            logger.info(f"选择递归right层: layer[{mid}-{end}], PPL: {ppl_right}")
            return binary_search_layer(mid, end, ppl_right)


    # 第二阶段：子模块搜索（不变）
    def binary_search_name(layer_idx, names, start, end, ppl=None):
        if end - start == 1:
            logger.info(f"找到坏比特所在子层: layer {layer_idx}, name {names[start]}, PPL: {ppl}\n")
            # loss, ppl, _, _, _ = evaluate_range((layer_idx, layer_idx + 1), (start, start + 1), None)
            return start, names[start], ppl

        mid = (start + end) // 2
        loss_left, ppl_left, _, _, _ = evaluate_range((layer_idx, layer_idx + 1), (start, mid), None)
        logger.info(f"ppl_left: {ppl_left}")
        loss_right, ppl_right, _, _, _ = evaluate_range((layer_idx, layer_idx + 1), (mid, end), None)
        logger.info(f"ppl_right: {ppl_right}")

        if ppl_left > ppl_right:
            logger.info(f"选择递归left子层: layer[{layer_idx}].[{start}-{mid}], PPL: {ppl_left}")
            return binary_search_name(layer_idx, names, start, mid, ppl_left)
        else:
            logger.info(f"选择递归right子层: layer[{layer_idx}].[{mid}-{end}], PPL: {ppl_right}")
            return binary_search_name(layer_idx, names, mid, end, ppl_right)


    # 第三阶段：权重索引搜索（优化版）
    def binary_search_weight(layer_idx, name_idx, name, start, end, cols, total_diff_bits=None, diff_position=None, bit_flip_range=None, ppl=None):
        # 评估当前范围
        # loss, ppl, total_diff_bits, diff_position, bit_flip_range = evaluate_range(
        #     (layer_idx, layer_idx + 1), (name_idx, name_idx + 1),
        #     (start, end, None, None)
        # )

        # logger.info(f"权重范围 [{start}, {end}): PPL={ppl}, 差异比特数={total_diff_bits}")

        # total_diff_bits, diff_position, bit_flip_range = None

        # 检查单比特翻转
        if total_diff_bits == 1 and diff_position:
            row, col, bit_idx = diff_position
            weight_idx = row * cols + col
            logger.info(f"范围[{start}-{end}]内只有单比特翻转！直接定位: weight_idx={weight_idx}")
            logger.info(f"权重位置: row={row}, col={col}, PPL: {ppl}\n")
            return weight_idx, bit_idx, ppl, total_diff_bits

        # 如果有翻转范围，直接使用
        if bit_flip_range:
            new_start, new_end = bit_flip_range
            # 确保在范围内
            new_start = max(start, new_start)
            new_end = min(end, new_end)

            # logger.info(f"使用翻转范围: [{new_start}, {new_end})")
            start, end = new_start, new_end

        # 标准二分
        if end - start <= 1:
            row, col, bit_idx = diff_position
            logger.info(f"终止权重递归: weight_idx={start}")
            weight_row, weight_col = start // cols, start % cols
            logger.info(f"权重位置: row={weight_row}, col={weight_col}, PPL: {ppl}\n")
            return start, bit_idx, ppl, total_diff_bits

        mid = (start + end) // 2
        loss_left, ppl_left, left_diff, left_pos, left_range = evaluate_range(
            (layer_idx, layer_idx + 1), (name_idx, name_idx + 1),
            (start, mid, None, None)
        )
        logger.info(f"ppl_left: {ppl_left}")
        loss_right, ppl_right, right_diff, right_pos, right_range = evaluate_range(
            (layer_idx, layer_idx + 1), (name_idx, name_idx + 1),
            (mid, end, None, None)
        )
        logger.info(f"ppl_right: {ppl_right}")

        if ppl_left > ppl_right:
            logger.info(f"选择递归left权重范围: layer[{layer_idx}].[{name}][{start}-{mid}], 翻转bit数[{left_diff}], PPL: {ppl_left}")
            return binary_search_weight(layer_idx, name_idx, name, start, mid, cols, left_diff, left_pos, left_range, ppl_left)
        else:
            logger.info(f"选择递归right权重范围: layer[{layer_idx}].[{name}][{mid}-{end}], 翻转bit数[{right_diff}], PPL: {ppl_right}")
            return binary_search_weight(layer_idx, name_idx, name, mid, end, cols, right_diff, right_pos, right_range, ppl_right)

    # 第四阶段：比特搜索（优化版）
    def binary_search_bit(layer_idx, name_idx, name, weight_idx, start, end, total_diff_bits=None, diff_position=None, ppl=None):
        # 评估当前范围
        # loss, ppl, total_diff_bits, diff_position, _ = evaluate_range(
        #     (layer_idx, layer_idx + 1), (name_idx, name_idx + 1),
        #     (weight_idx, weight_idx + 1, start, end)
        # )
        #
        # logger.info(f"比特范围 [{start}, {end}): PPL={ppl}, 差异比特数={total_diff_bits}")

        # 检查单比特翻转
        if total_diff_bits == 1 and diff_position:
            _, _, bit_pos = diff_position
            logger.info(f"范围[{start}-{end}]内只有单比特翻转！直接定位: bit_idx={bit_pos}, PPL: {ppl}")
            return bit_pos, ppl

        # 标准二分
        if end - start <= 1:
            logger.info(f"终止权重bit递归: bit_idx={start}, PPL: {ppl}")
            return start, ppl

        mid = (start + end) // 2
        loss_left, ppl_left, left_diff, left_pos, _ = evaluate_range(
            (layer_idx, layer_idx + 1), (name_idx, name_idx + 1),
            (weight_idx, weight_idx + 1, start, mid)
        )
        logger.info(f"ppl_left: {ppl_left}")
        loss_right, ppl_right, right_diff, right_pos, _ = evaluate_range(
            (layer_idx, layer_idx + 1), (name_idx, name_idx + 1),
            (weight_idx, weight_idx + 1, mid, end)
        )
        logger.info(f"ppl_right: {ppl_right}")

        if ppl_left > ppl_right:
            logger.info(f"选择递归left权重bit范围: layer[{layer_idx}].[{name}][{weight_idx}][bit[{start}-{mid}]], 翻转bit数[{left_diff}], PPL: {ppl_left}")
            return binary_search_bit(layer_idx, name_idx, name, weight_idx, start, mid, left_diff, left_pos, ppl_left)
        else:
            logger.info(f"选择递归right权重bit范围: layer[{layer_idx}].[{name}][{weight_idx}][bit[{mid}-{end}]], 翻转bit数[{right_diff}], PPL: {ppl_right}")
            return binary_search_bit(layer_idx, name_idx, name, weight_idx, mid, end, right_diff, right_pos, ppl_right)

    # 主流程
    logger.info("\n******************************************************************")
    logger.info(f"随机种子random_seed={random_seed}...")
    logger.info("开始优化版坏比特查找...")

    # 1. 查找层
    layer_idx, _ = binary_search_layer(0, len(layers))
    logger.info(f"定位到层 {layer_idx}")

    # 2. 查找子模块
    layer = layers[layer_idx]
    subset = find_layers(layer)
    names = list(subset.keys())

    name_idx, name, _ = binary_search_name(layer_idx, names, 0, len(names))

    # #### 从确定layer.subset开始测试
    # layer_idx, name_idx, name, ppl = 1, 2, "self_attn.v_proj", 20000.0
    # layer = layers[layer_idx]
    # subset = find_layers(layer)
    # ####

    # 3. 查找权重索引（使用优化）
    module = subset[name]
    rows, cols = module.weight.shape
    num_weights = rows * cols

    weight_idx, bit_idx, ppl, total_diff = binary_search_weight(layer_idx, name_idx, name, 0, num_weights, cols)
    # weight_idx, bit_idx, ppl, total_diff = binary_search_weight(layer_idx, name_idx, name, 5915932, 5916167, cols)  # 测试从[5915932, 5916167)索引权重开始

    weight_row, weight_col = weight_idx // cols, weight_idx % cols

    if total_diff != 1:  # 若当前权重内不止1bit翻转，需要bit级搜索
        # 4. 查找比特位置
        logger.info(f"当前权重内不止1bit翻转，翻转 {total_diff} bit，需要权重内bit搜索")
        bit_idx, ppl = binary_search_bit(layer_idx, name_idx, name, weight_idx, 0, wbits)

    return layer_idx, name, [weight_row, weight_col], bit_idx, ppl


def load_attack_logs(log_path):
    import ast  # 用于安全地将字符串 "[1, 2]" 转回列表 [1, 2]
    # 读取 csv
    df = pd.read_csv(log_path)

    attacks = []
    for index, row in df.iterrows():
        layer_idx = int(row['layer_idx'])
        submodule_name = row['submodule_name']

        # 关键步骤：把字符串 "[1923, 1512]" 转回 list [1923, 1512]
        position = ast.literal_eval(row['position'])

        attacks.append((layer_idx, submodule_name, position))

    return attacks




# ==========================================
# [新增] ViT 专用评估函数
# ==========================================
def evaluate_accuracy_vit(model, loader, dev, nsamples=None):
    model.eval()
    correct = 0
    total = 0
    count = 0

    with torch.no_grad():
        for batch in loader:
            if nsamples is not None and count >= nsamples:
                break

            pixel_values, label = batch
            pixel_values = pixel_values.to(dev)
            label = label.to(dev)

            if not isinstance(label, torch.Tensor):
                label = torch.tensor([label]).to(dev)
            if label.dim() == 0:
                label = label.unsqueeze(0)

            outputs = model(pixel_values)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == label).sum().item()
            total += label.size(0)
            count += 1

    return correct / total if total > 0 else 0.0

def evaluate_full_model_vit(model, layers, initial_input, test_loader, dev, nsamples):
    """
    ViT 版的 evaluate_full_model
    """
    current_output = initial_input.to(dev)

    with torch.no_grad():
        for layer in layers:
            layer = layer.to(dev)
            outs = []
            for j in range(nsamples):
                # ViT Encoder Layer 输入: (Batch, Seq, Hidden)
                inp = current_output[j].unsqueeze(0)
                # ViT Layer 不需要 attention_mask
                out = layer(inp)[0]
                outs.append(out.squeeze(0))
            current_output = torch.stack(outs)

    # 计算准确率
    _, acc = evaluate_accuracy_vit(model, current_output, test_loader, dev, nsamples)
    return 0.0, acc


def flip_and_evaluate_randomflip_vit(model, layers, quantizers, dev, wbits,
                                     test_loader, nsamples, random_bits2flip_rate,
                                     random_seed=0):
    """
    [ViT 优化版] 全局随机翻转评估
    使用 "备份 -> 全局注入 -> 推理 -> 恢复" 的逻辑
    """
    # 1. 备份整个模型权重到 CPU
    # ViT-Base 只有 ~86MB 参数，备份非常快且省显存
    full_model_backup = {}
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只备份可学习参数
            full_model_backup[name] = param.data.cpu().clone()

    # 设置种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    logger.info(f"Seed {random_seed}: Injecting errors (Rate={random_bits2flip_rate})...")

    # 2. 遍历层并注入错误
    # 类似于参考代码的 logic，但我们在推理前一次性把所有层都改了

    dtype = next(model.parameters()).dtype

    for layer_idx, layer in enumerate(layers):
        subset = find_layers(layer, layers=[nn.Linear])

        for name, module in subset.items():
            q_key = f"{layer_idx}_{name}"
            quantizer = quantizers.get(q_key)
            if quantizer is None: continue

            W = module.weight.data  # GPU上的权重

            # 准备量化参数
            # 注意：gptq quantizer 的 scale 形状可能与 weight 不一致 (Transposed)
            # 通常 weight 是 (Out, In)，scale 是 (Out, 1) [per-channel]
            scale = quantizer.scale.to(dev).to(dtype)

            # # reshape scale match weight
            # if scale.dim() == 1:
            #     scale = scale.reshape(-1, 1)

            # A. 量化
            W_quant = real_quantize(W, scale, wbits, dev)

            # Clamp
            maxq = 2 ** (wbits - 1) - 1
            W_quant = torch.clamp(W_quant, -maxq - 1, maxq)

            # B. 注入错误
            if random_bits2flip_rate > 0:
                # 转为无符号进行位操作 (offset binary)
                offset = 2 ** (wbits - 1)
                W_quant_unsigned = W_quant + offset

                # 调用 Error_injection_opt
                W_quant_flipped_unsigned = Error_injection_opt(
                    W_quant_unsigned,
                    bit_error_rate=random_bits2flip_rate,
                    bit_w=wbits
                )

                W_quant_flipped = W_quant_flipped_unsigned - offset
                W_quant_flipped = torch.clamp(W_quant_flipped, -maxq - 1, maxq)

                W_quant = W_quant_flipped

            # C. 反量化
            W_dequant = (W_quant * scale).to(dtype)

            # D. 应用回模型
            module.weight.data = W_dequant

    # 3. 执行推理
    acc = evaluate_accuracy_vit(model, test_loader, dev, nsamples)

    # 4. 恢复权重
    for name, param in model.named_parameters():
        if name in full_model_backup:
            param.data = full_model_backup[name].to(dev)

    del full_model_backup
    torch.cuda.empty_cache()

    return acc


def flip_and_evaluate_fewbit_vit0(model, layers, quantizers, test_loader, dev, wbits,
                                 nsamples, attack_logs):
    """
    [Mode 8] PBS 验证函数 (ViT 版)
    适配：调用 biterror1.ttt_7_18_all_value_flip.Error_injection_fewbit
    """

    from biterror1.ttt_7_18_all_value_flip import Error_injection_fewbit
    import ast

    dtype = next(model.parameters()).dtype

    # 1. 建立严格映射
    strict_map = {}
    for i, layer in enumerate(layers):
        subset = find_layers(layer, layers=[nn.Linear])
        for name, mod in subset.items():
            strict_map[f"{i}_{name}"] = mod

    # 2. 解析任务
    flip_tasks = {}
    for log in attack_logs:
        if isinstance(log, dict):
            raw_name, coords_raw, bit = log.get('Layer'), log.get('Coords'), log.get('Bit')
        else:
            raw_name, coords_raw, bit = log[1], log[2], log[3]

        if raw_name not in strict_map: continue

        coords = ast.literal_eval(coords_raw) if isinstance(coords_raw, str) else coords_raw
        if raw_name not in flip_tasks: flip_tasks[raw_name] = []
        flip_tasks[raw_name].append([coords, int(bit)])

    # 3. 全局备份 (因为 ViT 需要全模型推理，无法像 LLM 那样逐层处理)
    full_backup = {}

    # 4. 执行攻击 (修改权重)
    for strict_key, tasks in flip_tasks.items():
        if strict_key not in strict_map: continue

        module = strict_map[strict_key]
        quantizer = quantizers.get(strict_key)
        if not quantizer: continue

        # 备份权重 (只备份受影响的层)
        if strict_key not in full_backup:
            full_backup[strict_key] = module.weight.data.cpu().clone()

        W = module.weight.data
        scale = quantizer.scale.to(dev, dtype=dtype)
        # if scale.dim() == 1: scale = scale.reshape(-1, 1)

        # A. 量化
        W_quant = real_quantize(W, scale, wbits, dev)
        maxq = 2 ** (wbits - 1) - 1
        minq = -2 ** (wbits - 1)
        W_quant = torch.clamp(W_quant, minq, maxq)

        # B. 注入错误 (调用你的函数)
        offset = 2 ** (wbits - 1)
        W_quant2 = W_quant + offset

        # Call Error_injection_fewbit
        W_quant2 = Error_injection_fewbit(
            W_quant2,
            bit_w=wbits,
            flip_positions=tasks
        )

        W_quant2 = W_quant2 - offset

        # C. 反量化 & 应用
        W_quant = W_quant2.clamp(-maxq - 1, maxq).to(torch.int16)
        W_dequant = (W_quant * scale).to(dtype)
        module.weight.data = W_dequant

    # 5. 全模型评估
    logger.info(f"Evaluating PBS Attack on ViT ({len(flip_tasks)} layers affected)...")
    acc = evaluate_accuracy_vit(model, test_loader, dev, nsamples)

    # 6. 恢复权重
    for strict_key, original_weight in full_backup.items():
        module = strict_map[strict_key]
        module.weight.data = original_weight.to(dev)

    del full_backup
    torch.cuda.empty_cache()

    return acc


def flip_and_evaluate_fewbit_vit1(model, layers, quantizers, test_loader, dev, wbits,
                                 nsamples, attack_logs):
    """
    [Mode 8] PBS 验证函数 (ViT 版 - 严格复刻原版逻辑)
    适配：调用 biterror1.ttt_7_18_all_value_flip.Error_injection_fewbit
    注意：由于ViT需要全图推理，所以逻辑是：先全部修改 -> 评估 -> 全部恢复
    """

    from biterror1.ttt_7_18_all_value_flip import Error_injection_fewbit
    import ast

    # 1. 建立严格映射 (Layer Index_Name -> Module)
    strict_map = {}
    for i, layer in enumerate(layers):
        subset = find_layers(layer, layers=[nn.Linear])
        for name, mod in subset.items():
            strict_map[f"{i}_{name}"] = mod

    # 2. 解析任务 (Parsing Tasks)
    # 构造 flip_tasks: { "0_query": [ [coords, bit], [coords, bit]... ] }
    flip_tasks = {}
    for log in attack_logs:
        if isinstance(log, dict):
            raw_name = log.get('Layer')
            coords_raw = log.get('Coords')
            bit = log.get('Bit')
        else:
            raw_name = log[1]
            coords_raw = log[2]
            bit = log[3]

        if raw_name not in strict_map:
            # 如果找不到层，说明可能是 key 不匹配，或者层不在 layers 里 (比如 head)
            continue

        coords = ast.literal_eval(coords_raw) if isinstance(coords_raw, str) else coords_raw

        if raw_name not in flip_tasks:
            flip_tasks[raw_name] = []

        # 严格复刻原版：append([coords, int(bit)])
        flip_tasks[raw_name].append([coords, int(bit)])

    # 3. 执行攻击 (全量修改)
    # 我们需要一个全局备份，因为不能像 LLM 那样改完一层测一层
    full_backup = {}
    dtype = next(model.parameters()).dtype

    # 遍历所有层，寻找是否有任务
    for idx in range(len(layers)):
        layer = layers[idx]  # 此时不需要 .to(dev)，假设模型已在 GPU
        subset = find_layers(layer, layers=[nn.Linear])

        for name, module in subset.items():
            strict_key = f"{idx}_{name}"
            # 兼容性 Key 查找
            quantizer = quantizers.get(strict_key) or quantizers.get(f"{idx}.{name}")

            # 只有当该层在任务列表中时，才进行处理
            if quantizer and strict_key in flip_tasks:
                W = module.weight.data

                # A. 备份 (保存到 CPU 以省显存)
                full_backup[strict_key] = W.cpu().clone()

                # B. 准备量化参数
                scale = quantizer.scale.to(dev, dtype=dtype)

                # C. 模拟量化 (Real Quantize)
                # 使用你提到的 real_quantize 函数 (如果它在上下文中)，或者手动实现
                # 这里手动实现以确保独立性
                W_quant = real_quantize(W, scale, wbits, dev)
                maxq = 2 ** (wbits - 1) - 1
                minq = -2 ** (wbits - 1)
                W_quant = torch.clamp(W_quant, minq, maxq)

                # D. 执行 Few-bit flip
                offset = 2 ** (wbits - 1)
                W_quant2 = W_quant + offset

                # [关键调用] 传入 flip_positions = [[coords, bit], ...]
                W_quant2 = Error_injection_fewbit(
                    W_quant2,
                    bit_w=wbits,
                    flip_positions=flip_tasks[strict_key]
                )

                W_quant2 = W_quant2 - offset

                # E. 反量化并写回
                W_quant = W_quant2.clamp(-maxq - 1, maxq).to(torch.int16)
                W_dequant = (W_quant * scale).to(dtype)

                module.weight.data = W_dequant

    # 4. 全模型评估 (Evaluate)
    # 原版是用 compute_loss，这里换成 ViT 的 Accuracy
    if len(full_backup) > 0:
        logger.info(f"Evaluating PBS Attack on ViT ({len(full_backup)} layers affected)...")

    # 评估
    acc = evaluate_accuracy_vit(model, test_loader, dev, nsamples)

    # 5. 恢复模型 (Restore)
    # 将所有修改过的层恢复原状
    for strict_key, original_weight in full_backup.items():
        module = strict_map[strict_key]
        module.weight.data = original_weight.to(dev)

    # 清理
    del full_backup
    torch.cuda.empty_cache()

    # 返回 Acc 作为指标 (Loss 设为 -1 占位)
    return acc


def flip_and_evaluate_fewbit_vit(model, layers, quantizers, test_loader, dev, wbits,
                                 nsamples, attack_logs, model_kwargs=None,
                                 debug_flip=False):  # <--- [新增开关] 默认开启方便你调试
    """
    [Mode 8] PBS 验证函数 (ViT 版)
    功能：执行攻击，并支持详细的比特翻转肉眼比对校验。
    """
    if model_kwargs is None: model_kwargs = {}

    from biterror1.ttt_7_18_all_value_flip import Error_injection_fewbit
    import ast

    # 1. 建立映射
    strict_map = {}
    for i, layer in enumerate(layers):
        subset = find_layers(layer, layers=[nn.Linear])
        for name, mod in subset.items():
            strict_map[f"{i}_{name}"] = mod

    # 2. 解析任务
    flip_tasks = {}
    for log in attack_logs:
        if isinstance(log, dict):
            raw_name, coords_raw, bit = log.get('Layer'), log.get('Coords'), log.get('Bit')
        else:
            raw_name, coords_raw, bit = log[1], log[2], log[3]

        if raw_name not in strict_map: continue

        coords = ast.literal_eval(coords_raw) if isinstance(coords_raw, str) else coords_raw
        if raw_name not in flip_tasks: flip_tasks[raw_name] = []
        flip_tasks[raw_name].append([coords, int(bit)])

    full_backup = {}
    dtype = next(model.parameters()).dtype

    # 3. 攻击执行
    for idx in range(len(layers)):
        layer = layers[idx]
        subset = find_layers(layer, layers=[nn.Linear])

        for name, module in subset.items():
            strict_key = f"{idx}_{name}"
            quantizer = quantizers.get(strict_key) or quantizers.get(f"{idx}.{name}")

            if quantizer and strict_key in flip_tasks:
                tasks = flip_tasks[strict_key]
                if not tasks: continue

                W = module.weight.data
                full_backup[strict_key] = W.cpu().clone()

                # 量化准备
                scale = quantizer.scale.to(dev, dtype=dtype)
                if scale.dim() == 1:
                    raise ValueError("Scale cannot be 1D. Please reshape it to (N, 1) before calling real_quantize.")
                W_quant = real_quantize(W, scale, wbits, dev)

                offset = 2 ** (wbits - 1)
                W_quant_unsigned = W_quant + offset

                # ================= [DEBUG START] =================
                W_before_debug = None
                if debug_flip:
                    # 记录翻转前的张量 (Int64 以防溢出干扰比对)
                    W_before_debug = W_quant_unsigned.clone().to(torch.int64)
                # =================================================

                # Call Injection
                W_quant_unsigned = Error_injection_fewbit(
                    W_quant_unsigned,
                    bit_w=wbits,
                    flip_positions=tasks
                )

                # ================= [DEBUG VERIFICATION 2.0] =================
                if debug_flip:
                    logger.info(f"\n====== [Layer: {strict_key}] Bit-Flip Verification ======")
                    W_after_debug = W_quant_unsigned.to(torch.int64)

                    # 1. 计算实际变化 (XOR)
                    diff_mask = torch.bitwise_xor(W_before_debug, W_after_debug)
                    changed_indices = torch.nonzero(diff_mask, as_tuple=False)

                    actual_changes_set = set()  # 用集合存储 (r, c, bit)

                    print(f"  >>> [Actual Changes Detected]:")
                    for coord in changed_indices:
                        r, c = coord[0].item(), coord[1].item()
                        val_diff = diff_mask[r, c].item()

                        # [FIX] 解析多 Bit 变化
                        # 遍历 0 到 15 位，检查每一位是否为 1
                        for b in range(16):  # 假设最大 16 bit
                            if (val_diff >> b) & 1:
                                print(
                                    f"      Coords=[{r}, {c}] | Bit={b} | Val_Before={W_before_debug[r, c]} -> Val_After={W_after_debug[r, c]}")
                                actual_changes_set.add((r, c, b))

                    # 2. 验证匹配
                    logger.info(f"预期翻转总数: {len(tasks)}")
                    logger.info(f"实际翻转总数: {len(actual_changes_set)}")

                    # 3. 严格比对
                    expected_set = set()
                    for t_coord, t_bit in tasks:
                        expected_set.add((t_coord[0], t_coord[1], t_bit))

                    # 打印漏掉的或多余的
                    missing = expected_set - actual_changes_set
                    extra = actual_changes_set - expected_set

                    if len(missing) == 0 and len(extra) == 0:
                        logger.info("  ✅ 校验通过：完美匹配！")
                    else:
                        logger.error("  ❌ 校验失败！")
                        if len(missing) > 0:
                            print(f"     Missing: {list(missing)[:5]} ...")
                        if len(extra) > 0:
                            print(f"     Extra: {list(extra)[:5]} ...")

                    logger.info("==========================================================\n")
                # ========================================================

                W_quant2 = W_quant_unsigned - offset
                W_dequant = (W_quant2 * scale).to(dtype)
                module.weight.data = W_dequant

    logger.info(f"Evaluating PBS Attack on ViT ({len(full_backup)} layers modified)...")
    acc = evaluate_accuracy_vit(model, test_loader, dev, nsamples)

    for strict_key, original_weight in full_backup.items():
        module = strict_map[strict_key]
        module.weight.data = original_weight.to(dev)

    del full_backup
    torch.cuda.empty_cache()

    return acc

# ==============================================================================
# [新增] ViT 专用 Outlier 扫描函数
# ==============================================================================
def scan_outliers_layerwise_vit_x_dim_error(model, inps, dev, filename="outlier_config.csv"):
    """
    ViT 专用分层扫描：不需要重新抓取输入，直接使用 inps (Hidden States)
    """
    logger.info(f">>> [Profiler] 启动 ViT 极速分层扫描 (nsamples={inps.shape[0]})...")

    layers = get_model_layers_vit(model)
    nsamples = inps.shape[0]

    layer_stats = []
    # 确保输入在 GPU
    current_inputs = inps.to(dev)

    # 循环层
    for i, layer in enumerate(tqdm(layers, desc="Scanning Layers (ViT)")):
        layer = layer.to(dev)
        subset = find_layers(layer, layers=[nn.Linear])
        layer_local_maxes = {}

        # 定义 Hook
        def stat_hook(name):
            def hook(module, inp, out):
                x = inp[0].detach()  # [Batch, Seq, Dim]
                # 计算 max absolute value
                # ViT 也是 (Batch, Seq, Dim)，我们统计 Channel (Dim) 维度的最大值
                if x.dim() == 3:
                    col_max = x.abs().amax(dim=(0, 1))
                else:
                    logger.error(f"处理不了不是3d的输入x")
                    # col_max = x.abs().flatten().max()

                if name not in layer_local_maxes:
                    layer_local_maxes[name] = col_max
                else:
                    torch.maximum(layer_local_maxes[name], col_max, out=layer_local_maxes[name])

            return hook

        # 注册 Hooks
        handles = [m.register_forward_hook(stat_hook(n)) for n, m in subset.items()]

        # 前向传播 (Update current_inputs for next layer)
        layer_output_list = []
        for j in range(nsamples):
            inp_j = current_inputs[j].unsqueeze(0)  # [1, Seq, Dim]
            with torch.no_grad():
                # ViT Layer 输出通常是 tuple (hidden_states, ...)
                out_j = layer(inp_j)[0]
            layer_output_list.append(out_j)  # Keep [1, Seq, Dim]

        # 移除 Hooks
        for h in handles: h.remove()

        # 统计分析
        for name, max_val in layer_local_maxes.items():
            val_float = max_val.float()
            mean, std = val_float.mean().item(), val_float.std().item()
            # 阈值策略：均值 + 6倍标准差，且至少是均值的2倍
            thresh = max(mean + 6 * std, max(1.0, mean * 2.0))
            outliers = (val_float > thresh).nonzero(as_tuple=True)[0]

            if len(outliers) > 0:
                layer_stats.append({
                    "Layer": f"{i}.{name}",
                    "Outlier_Indices": ";".join(map(str, outliers.cpu().numpy())),
                    "Count": len(outliers)
                })

        # 堆叠输出作为下一层的输入
        current_inputs = torch.cat(layer_output_list, dim=0)  # [N, Seq, Dim]

    # 保存 CSV
    df = pd.DataFrame(layer_stats)
    df.to_csv(filename, index=False)
    logger.info(f"Outlier 统计已保存至: {filename}")
    return filename


# ==============================================================================
# [修正版] ViT 专用 Outlier 扫描函数 (修复 UnboundLocalError 和 2D 输入支持)
# ==============================================================================
def scan_outliers_layerwise_vit(model, inps, dev, filename="outlier_config.csv",
                                sigma_thresh=6.0, rel_thresh=2.0, abs_thresh=1.0):
    """
    ViT 专用分层扫描：
    1. 修复了 UnboundLocalError
    2. 增加了对 2D 输入 [Tokens, Dim] 的支持 (通用化处理)
    """
    logger.info(f">>> [Profiler] 启动 ViT 极速分层扫描 (nsamples={inps.shape[0]})...")

    layers = get_model_layers_vit(model)
    nsamples = inps.shape[0]

    layer_stats = []
    # 确保输入在 GPU
    current_inputs = inps.to(dev)

    # 循环层
    for i, layer in enumerate(tqdm(layers, desc="Scanning Layers (ViT)")):
        layer = layer.to(dev)
        subset = find_layers(layer, layers=[nn.Linear])
        layer_local_maxes = {}

        # 定义 Hook
        def stat_hook(name):
            def hook(module, inp, out):
                x = inp[0].detach()  # 输入可能是 [B, S, Dim] 或 [N, Dim]

                # --- [核心修复] ---
                # 无论 x 是几维，我们关注的是最后一维 (Feature Dim) 的 Outlier
                # 所以我们需要求 Max 的维度是：除了最后一维之外的所有维度
                if x.dim() >= 2:
                    # 例如 3D: range(2) -> (0, 1); 2D: range(1) -> (0,)
                    reduce_dims = tuple(range(x.dim() - 1))
                    col_max = x.abs().amax(dim=reduce_dims)
                else:
                    # 极少见的情况 (1D)，直接取绝对值
                    col_max = x.abs()

                # 确保 col_max 已经被定义，再进入字典操作
                if name not in layer_local_maxes:
                    layer_local_maxes[name] = col_max
                else:
                    # 确保维度一致再 max，防止广播错误
                    if layer_local_maxes[name].shape == col_max.shape:
                        torch.maximum(layer_local_maxes[name], col_max, out=layer_local_maxes[name])
                    else:
                        # 这是一个保险措施，通常不会发生，除非模型动态改变了 Hidden Dim
                        pass

            return hook

        # 注册 Hooks
        handles = [m.register_forward_hook(stat_hook(n)) for n, m in subset.items()]

        # 前向传播 (Update current_inputs for next layer)
        layer_output_list = []
        for j in range(nsamples):
            inp_j = current_inputs[j].unsqueeze(0)  # [1, Seq, Dim]
            with torch.no_grad():
                # ViT Layer 输出通常是 tuple (hidden_states, ...)
                # 注意：有些 ViT 实现可能会改变 output 形状，但在 block 级别通常保持一致
                out_j = layer(inp_j)[0]
            layer_output_list.append(out_j)

            # 移除 Hooks
        for h in handles: h.remove()

        # 统计分析
        for name, max_val in layer_local_maxes.items():
            val_float = max_val.float()
            mean, std = val_float.mean().item(), val_float.std().item()
            # 阈值策略：均值 + 6倍标准差，且至少是均值的2倍
            # thresh = max(mean + 6 * std, max(1.0, mean * 2.0))
            # 使用 h_sigma, h_rel, h_abs
            thresh = max(mean + sigma_thresh * std, max(abs_thresh, mean * rel_thresh))
            outliers = (val_float > thresh).nonzero(as_tuple=True)[0]

            if len(outliers) > 0:
                layer_stats.append({
                    "Layer": f"{i}.{name}",
                    "Outlier_Indices": ";".join(map(str, outliers.cpu().numpy())),
                    "Count": len(outliers)
                })

        # 堆叠输出作为下一层的输入
        current_inputs = torch.cat(layer_output_list, dim=0)  # [N, Seq, Dim]

    # 保存 CSV
    df = pd.DataFrame(layer_stats)
    df.to_csv(filename, index=False)
    logger.info(f"Outlier 统计已保存至: {filename}")
    return filename

def progressive_bit_search_vit_2_3(model, layers, calib_loader, dev, wbits, quantizers, nsamples=32, attack_iters=10,
                               chunk_size=4, output_csv="attack_logs_PBS_ViT.csv"):
    """
    [PBS ViT 极速版] Progressive Bit Search (Vectorized for Vision Transformer)

    适配说明：
    1. 输入改为 calib_loader，提取 pixel_values 和 labels 计算 CrossEntropy Loss。
    2. 核心搜索逻辑保持向量化，适配 H200 高性能计算。
    """
    import copy
    import pandas as pd
    import torch.nn as nn

    logger.info(f">>> [PBS-ViT] 启动搜索 (Iters={attack_iters}, Bits={wbits}, Mode=Vectorized)...")

    model.eval()
    model.zero_grad()

    attack_logs = []
    modified_weights_backup = {}
    visited_bits = set()

    # 预计算 Offset 和 Bit Masks
    offset = 2 ** (wbits - 1)
    bit_range = torch.arange(wbits, device=dev)
    powers_of_2 = 2 ** bit_range  # [1, 2, 4, ..., 2^(w-1)]

    # === 1. 数据准备 (预取一个 Batch 用于梯度计算) ===
    # PBS 是一种“白盒”攻击搜索，通常基于校准集的一个 Batch 来计算敏感度
    # 为了保证 Iteration 间的一致性，我们将数据固定在 GPU 上

    # 从 loader 获取数据
    if isinstance(calib_loader, list):
        # 兼容 list of tuples [(pixels, label), ...]
        # 手动拼 Batch
        batch_data = calib_loader[:min(len(calib_loader), nsamples)]
        pixel_list = []
        label_list = []
        for p, l in batch_data:
            pixel_list.append(p)
            label_list.append(l)

        # 堆叠
        fixed_pixels = torch.cat(pixel_list, dim=0).to(dev)  # [N, 3, H, W]

        # 处理 Label (可能是 Tensor 也可能是 Int)
        if isinstance(label_list[0], torch.Tensor):
            fixed_labels = torch.stack(label_list).to(dev)
        else:
            fixed_labels = torch.tensor(label_list).to(dev)

        if fixed_labels.dim() > 1: fixed_labels = fixed_labels.view(-1)

    else:
        # 假设是 DataLoader
        batch = next(iter(calib_loader))
        fixed_pixels, fixed_labels = batch
        fixed_pixels = fixed_pixels.to(dev)
        fixed_labels = fixed_labels.to(dev)

    logger.info(f"PBS 锁定梯度计算数据: Shape={fixed_pixels.shape}")

    # === 2. 迭代搜索 ===
    for iter_step in range(attack_iters):
        logger.info(f"--- PBS Iteration {iter_step + 1}/{attack_iters} ---")

        global_best_score = -1.0
        global_target = None

        # 启用梯度检查点以节省显存 (可选，ViT 通常不需要)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        num_layers = len(layers)

        # 按块遍历层 (Chunk Processing)
        for start_idx in range(0, num_layers, chunk_size):
            end_idx = min(start_idx + chunk_size, num_layers)

            # A. 激活当前块的梯度
            model.requires_grad_(False)
            active_params = []
            current_chunk_meta = []

            for i in range(start_idx, end_idx):
                layer = layers[i]
                subset = find_layers(layer, layers=[nn.Linear])

                for name, mod in subset.items():
                    # 寻找对应的量化器 Key
                    q_key_under = f"{i}_{name}"
                    q_key_dot = f"{i}.{name}"
                    found_q_key = q_key_under if q_key_under in quantizers else (
                        q_key_dot if q_key_dot in quantizers else None)

                    if found_q_key:
                        mod.weight.requires_grad_(True)
                        active_params.append(mod.weight)
                        current_chunk_meta.append({
                            'layer_idx': i,
                            'sub_name': name,
                            'module': mod,
                            'quant_key': found_q_key
                        })

            if not active_params: continue

            # B. 计算梯度 (Forward + Backward)
            model.zero_grad()
            try:
                # ViT Forward: Pixel -> Loss
                outputs = model(fixed_pixels, labels=fixed_labels)
                loss = outputs.loss
                loss.backward()
            except RuntimeError as e:
                logger.error(f"PBS OOM at chunk {start_idx}: {e}")
                model.zero_grad()
                torch.cuda.empty_cache()
                continue

            # C. 向量化计算分数
            for meta in current_chunk_meta:
                mod = meta['module']
                if mod.weight.grad is None: continue

                quantizer = quantizers[meta['quant_key']]
                scale = quantizer.scale.to(dev)
                w = mod.weight.data
                g = mod.weight.grad.data

                # --- 向量化核心逻辑 (与 LLM 版一致) ---

                # 1. 模拟量化取整
                w_int_signed = (w / scale).round().to(torch.int64)
                w_int_unsigned = w_int_signed + offset  # 转 Offset Binary
                g_int = g * scale  # 梯度也缩放，保持量纲一致

                # 2. 广播计算 Bit 状态 [N, B]
                # N = num_weights, B = wbits
                w_bits_mat = (w_int_unsigned.view(-1, 1) >> bit_range.view(1, -1)) & 1

                # 3. 梯度展开
                g_expanded = g_int.view(-1, 1)

                # 4. 有效性 Mask
                # 如果当前 bit 是 0，且梯度 > 0 (希望权重变大)，则翻转 0->1 会使权重变大 (+)，Loss 变大。有效。
                # 如果当前 bit 是 1，且梯度 < 0 (希望权重变小)，则翻转 1->0 会使权重变小 (-)，Loss 变大。有效。
                # 注意：PBS 的目标是 Maximize Loss (Attack)
                valid_mask = ((w_bits_mat == 0) & (g_expanded > 0)) | \
                             ((w_bits_mat == 1) & (g_expanded < 0))

                # 5. 计算分数 Score = |Grad| * 2^Bit
                score_mat = g_expanded.abs() * powers_of_2.view(1, -1)

                # Mask 掉无效翻转
                score_mat = torch.where(valid_mask, score_mat, torch.tensor(-1.0, device=dev))

                # 6. 去重与选优 Loop
                while True:
                    # 找当前层最大值
                    max_s, flat_idx_in_mat = score_mat.view(-1).max(0)
                    max_s = max_s.item()

                    if max_s <= global_best_score:
                        break

                    flat_idx_in_mat = flat_idx_in_mat.item()

                    weight_flat_idx = flat_idx_in_mat // wbits
                    bit_idx = flat_idx_in_mat % wbits

                    visit_key = f"{meta['layer_idx']}|{meta['sub_name']}|{weight_flat_idx}|{bit_idx}"

                    if visit_key in visited_bits:
                        score_mat.view(-1)[flat_idx_in_mat] = -1.0  # 排除已访
                        continue
                    else:
                        global_best_score = max_s

                        # 计算物理坐标 (仅用于 Log)
                        row_idx = weight_flat_idx // mod.weight.shape[1]
                        col_idx = weight_flat_idx % mod.weight.shape[1]
                        grad_val = g_expanded[weight_flat_idx, 0].item()

                        # 处理 per-channel scale 索引
                        # scale shape: (Out, 1) -> view(-1) has size Out
                        # weight shape: (Out, In)
                        # scale_idx = row_idx
                        scale_val = scale.view(-1)[row_idx].item() if scale.numel() > 1 else scale.item()

                        global_target = {
                            'layer_idx': meta['layer_idx'],
                            'sub_name': meta['sub_name'],
                            'flat_idx': weight_flat_idx,
                            'coords': [row_idx, col_idx],
                            'bit': bit_idx,
                            'score': max_s,
                            'direction': 1 if grad_val > 0 else -1,
                            'scale': scale_val,
                            'visit_key': visit_key
                        }
                        break

            # 清理
            model.zero_grad()
            for p in active_params: p.requires_grad_(False)
            torch.cuda.empty_cache()

        # === 3. 执行攻击 ===
        if global_target is None:
            logger.warning(f"PBS Iter {iter_step + 1}: 搜索结束 (无有效目标)。")
            break

        visited_bits.add(global_target['visit_key'])
        log_layer_name = f"{global_target['layer_idx']}_{global_target['sub_name']}"

        attack_logs.append([
            iter_step + 1, log_layer_name, global_target['coords'], global_target['bit'], global_target['score']
        ])

        # 获取目标模块
        layer = layers[global_target['layer_idx']]
        subset = find_layers(layer, layers=[nn.Linear])
        mod = subset[global_target['sub_name']]

        # 备份原始值
        backup_key_unique = global_target['visit_key'].rsplit('|', 1)[0]  # 去掉 bit_idx，只备份权重值
        if backup_key_unique not in modified_weights_backup:
            curr_w = mod.weight.data.view(-1)[global_target['flat_idx']].item()
            modified_weights_backup[backup_key_unique] = {
                'layer_idx': global_target['layer_idx'],
                'sub_name': global_target['sub_name'],
                'flat_idx': global_target['flat_idx'],
                'val': curr_w
            }

        # 物理修改权重 (Apply Perturbation)
        # Delta = Direction * 2^Bit * Scale
        perturbation = global_target['direction'] * (2 ** global_target['bit']) * global_target['scale']

        with torch.no_grad():
            mod.weight.view(-1)[global_target['flat_idx']] += perturbation

        logger.info(
            f"PBS Iter {iter_step + 1}: Attacked {log_layer_name} | Bit {global_target['bit']} | Score {global_target['score']:.2e}")

    # === 4. 收尾 ===
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # 保存日志
    csv_data = [{"Iter": l[0], "Layer": l[1], "Coords": str(l[2]), "Bit": l[3], "Score": l[4]} for l in attack_logs]
    pd.DataFrame(csv_data).to_csv(output_csv, index=False)
    logger.info(f"PBS 攻击日志已保存至: {output_csv}")

    # 恢复模型参数
    logger.info("PBS 搜索结束，正在恢复模型原始参数...")
    with torch.no_grad():
        for key, info in modified_weights_backup.items():
            layer = layers[info['layer_idx']]
            subset = find_layers(layer, layers=[nn.Linear])
            mod = subset[info['sub_name']]
            mod.weight.view(-1)[info['flat_idx']] = info['val']

    logger.info("PBS 模型恢复完成。")
    return attack_logs


def progressive_bit_search_vit(model, layers, calib_loader, dev, wbits, quantizers, nsamples=32, attack_iters=10,
                               chunk_size=4, output_csv="attack_logs_PBS_ViT.csv"):
    """
    [PBS ViT 极速版] Progressive Bit Search (Vectorized for Vision Transformer)
    修正：统一量化逻辑、修复 Scale 维度、增强鲁棒性。
    """
    import copy
    import pandas as pd
    import torch.nn as nn

    logger.info(f">>> [PBS-ViT] 启动搜索 (Iters={attack_iters}, Bits={wbits}, Mode=Vectorized)...")

    model.eval()
    model.zero_grad()

    attack_logs = []
    modified_weights_backup = {}
    visited_bits = set()

    # 预计算 Offset (Tensor) 和 Bit Masks
    offset = 2 ** (wbits - 1)
    # [Fix] Offset 转 Tensor
    offset_tensor = torch.tensor(offset, device=dev, dtype=torch.int64)

    bit_range = torch.arange(wbits, device=dev)
    powers_of_2 = 2 ** bit_range

    # === 1. 数据准备 (预取一个 Batch) ===
    if isinstance(calib_loader, list):
        batch_data = calib_loader[:min(len(calib_loader), nsamples)]
        pixel_list = []
        label_list = []
        for p, l in batch_data:
            pixel_list.append(p)
            label_list.append(l)

        fixed_pixels = torch.cat(pixel_list, dim=0).to(dev)
        if isinstance(label_list[0], torch.Tensor):
            fixed_labels = torch.stack(label_list).to(dev)
        else:
            fixed_labels = torch.tensor(label_list).to(dev)
        if fixed_labels.dim() > 1: fixed_labels = fixed_labels.view(-1)
    else:
        batch = next(iter(calib_loader))
        fixed_pixels, fixed_labels = batch
        fixed_pixels = fixed_pixels.to(dev)
        fixed_labels = fixed_labels.to(dev)

    logger.info(f"PBS 锁定梯度计算数据: Shape={fixed_pixels.shape}")

    # === 2. 迭代搜索 ===
    for iter_step in range(attack_iters):
        logger.info(f"--- PBS Iteration {iter_step + 1}/{attack_iters} ---")

        global_best_score = -1.0
        global_target = None

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        num_layers = len(layers)

        for start_idx in range(0, num_layers, chunk_size):
            end_idx = min(start_idx + chunk_size, num_layers)

            model.requires_grad_(False)
            active_params = []
            current_chunk_meta = []

            for i in range(start_idx, end_idx):
                layer = layers[i]
                subset = find_layers(layer, layers=[nn.Linear])

                for name, mod in subset.items():
                    q_key_under = f"{i}_{name}"
                    q_key_dot = f"{i}.{name}"
                    found_q_key = q_key_under if q_key_under in quantizers else (
                        q_key_dot if q_key_dot in quantizers else None)

                    if found_q_key:
                        mod.weight.requires_grad_(True)
                        active_params.append(mod.weight)
                        current_chunk_meta.append({
                            'layer_idx': i,
                            'sub_name': name,
                            'module': mod,
                            'quant_key': found_q_key
                        })

            if not active_params: continue

            model.zero_grad()
            try:
                outputs = model(fixed_pixels, labels=fixed_labels)
                loss = outputs.loss
                loss.backward()
            except RuntimeError as e:
                logger.error(f"PBS OOM at chunk {start_idx}: {e}")
                model.zero_grad()
                torch.cuda.empty_cache()
                continue

            for meta in current_chunk_meta:
                mod = meta['module']
                if mod.weight.grad is None: continue

                quantizer = quantizers[meta['quant_key']]
                scale = quantizer.scale.to(dev)

                # [Fix] 自动修复 Scale 维度
                # if scale.dim() == 1:
                #     scale = scale.view(-1, 1)
                if scale.dim() == 1:
                    raise ValueError("Scale cannot be 1D. Please reshape it to (N, 1) before calling real_quantize.")

                w = mod.weight.data
                g = mod.weight.grad.data

                # [Fix] 使用统一量化
                w_int_signed = real_quantize(w, scale, wbits, dev).to(torch.int64)
                w_int_unsigned = w_int_signed + offset_tensor
                g_int = g * scale

                # 向量化计算
                w_bits_mat = (w_int_unsigned.view(-1, 1) >> bit_range.view(1, -1)) & 1
                g_expanded = g_int.view(-1, 1)

                valid_mask = ((w_bits_mat == 0) & (g_expanded > 0)) | \
                             ((w_bits_mat == 1) & (g_expanded < 0))

                score_mat = g_expanded.abs() * powers_of_2.view(1, -1)
                score_mat = torch.where(valid_mask, score_mat, torch.tensor(-1.0, device=dev))

                while True:
                    max_s, flat_idx_in_mat = score_mat.view(-1).max(0)
                    max_s = max_s.item()

                    if max_s <= global_best_score:
                        break

                    flat_idx_in_mat = flat_idx_in_mat.item()
                    num_bits = wbits
                    weight_flat_idx = flat_idx_in_mat // num_bits
                    bit_idx = flat_idx_in_mat % num_bits

                    visit_key = f"{meta['layer_idx']}|{meta['sub_name']}|{weight_flat_idx}|{bit_idx}"

                    if visit_key in visited_bits:
                        score_mat.view(-1)[flat_idx_in_mat] = -1.0
                        continue
                    else:
                        global_best_score = max_s

                        row_idx = weight_flat_idx // mod.weight.shape[1]
                        col_idx = weight_flat_idx % mod.weight.shape[1]
                        grad_val = g_expanded[weight_flat_idx, 0].item()

                        # [Fix] 获取正确的 Scale
                        # 如果 scale 是 [Out, 1]，则 row_idx 对应 scale index
                        target_scale = scale[row_idx, 0].item()

                        global_target = {
                            'layer_idx': meta['layer_idx'],
                            'sub_name': meta['sub_name'],
                            'flat_idx': weight_flat_idx,
                            'coords': [row_idx, col_idx],
                            'bit': bit_idx,
                            'score': max_s,
                            'direction': 1 if grad_val > 0 else -1,
                            'scale': target_scale,
                            'visit_key': visit_key
                        }
                        break

            model.zero_grad()
            for p in active_params: p.requires_grad_(False)
            torch.cuda.empty_cache()

        # === 3. 执行攻击 ===
        if global_target is None:
            logger.warning(f"PBS Iter {iter_step + 1}: 搜索结束 (无有效目标)。")
            break

        visited_bits.add(global_target['visit_key'])
        log_layer_name = f"{global_target['layer_idx']}_{global_target['sub_name']}"

        attack_logs.append([
            iter_step + 1, log_layer_name, global_target['coords'], global_target['bit'], global_target['score']
        ])

        layer = layers[global_target['layer_idx']]
        subset = find_layers(layer, layers=[nn.Linear])
        mod = subset[global_target['sub_name']]

        # [Fix] 备份 key 去掉 bit 信息
        backup_key_unique = f"{global_target['layer_idx']}|{global_target['sub_name']}|{global_target['flat_idx']}"
        if backup_key_unique not in modified_weights_backup:
            curr_w = mod.weight.data.view(-1)[global_target['flat_idx']].item()
            modified_weights_backup[backup_key_unique] = {
                'layer_idx': global_target['layer_idx'],
                'sub_name': global_target['sub_name'],
                'flat_idx': global_target['flat_idx'],
                'val': curr_w
            }

        perturbation = global_target['direction'] * (2 ** global_target['bit']) * global_target['scale']

        with torch.no_grad():
            # [Fix] 使用 .data 更安全
            mod.weight.data.view(-1)[global_target['flat_idx']] += perturbation

        logger.info(
            f"PBS Iter {iter_step + 1}: Attacked {log_layer_name} | Bit {global_target['bit']} | Score {global_target['score']:.2e}")

    # === 4. 收尾 ===
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    csv_data = [{"Iter": l[0], "Layer": l[1], "Coords": str(l[2]), "Bit": l[3], "Score": l[4]} for l in attack_logs]
    pd.DataFrame(csv_data).to_csv(output_csv, index=False)
    logger.info(f"PBS 攻击日志已保存至: {output_csv}")

    logger.info("PBS 搜索结束，正在恢复模型原始参数...")
    with torch.no_grad():
        for key, info in modified_weights_backup.items():
            layer = layers[info['layer_idx']]
            subset = find_layers(layer, layers=[nn.Linear])
            mod = subset[info['sub_name']]
            # [Fix] 使用 .data 恢复
            mod.weight.data.view(-1)[info['flat_idx']] = info['val']

    logger.info("PBS 模型恢复完成。")
    return attack_logs


