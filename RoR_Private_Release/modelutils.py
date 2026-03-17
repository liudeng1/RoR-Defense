# bit_flip_utils.py

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from custom_logger import logger
DEV = torch.device('cuda:0')

# ==========================================
# 1. 统一模型加载器 (The One Loader)
# ==========================================
def get_model_unified(model_path, device="cpu"):
    """
    通用加载函数：自动识别 OPT, Llama, Qwen, Mistral
    """
    logger.info(f"正在加载模型: {model_path} ...")

    # 1. 自动加载配置
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.error(f"加载 Config 失败: {e}")
        raise

    # 2. 确定精度 (H200 推荐 BF16)
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 3. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,  # 必须开启，为了 Qwen
        device_map=None  # 我们手动控制设备
    )

    # 4. 统一 seqlen (处理不同模型配置名不一样的问题)
    if hasattr(config, "max_position_embeddings"):
        model.seqlen = config.max_position_embeddings
    elif hasattr(config, "seq_length"):  # Qwen 可能用这个名字
        model.seqlen = config.seq_length
    else:
        model.seqlen = 2048  # 默认值

    model.eval()
    return model


# ==========================================
# 2. 统一层获取器 (The Layer Finder)
# ==========================================
def get_model_layers(model):
    """
    从模型中自动提取 Transformer Layers 列表，
    无需再写 if type == 'opt'...
    """
    # 常见结构 1: model.model.layers (Llama, Qwen, Mistral)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # 常见结构 2: model.model.decoder.layers (OPT)
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder.layers

    # 常见结构 3: 直接在 decoder 下 (裸 OPT)
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return model.decoder.layers

    raise ValueError(f"无法自动定位模型层结构: {type(model)}")

# ==========================================
# 3. 统一 RoPE 计算器
# ==========================================
def precompute_rope(model, dev, seqlen):
    """
    Llama/Qwen/Mistral 通用的 RoPE 预计算
    OPT 会自动返回 None
    """
    # 检查是否有 rotary_emb 模块
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        logger.info(f"预计算 RoPE (适用 Llama/Qwen/Mistral)...")
        rotary_emb = model.model.rotary_emb.to(dev)

        # 构造辅助变量
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)
        dummy_x = torch.zeros(1, seqlen, head_dim, device=dev, dtype=model.dtype)

        # 计算
        try:
            cos, sin = rotary_emb(dummy_x, position_ids)
        except TypeError:
            cos, sin = rotary_emb(dummy_x, seqlen)
        return (cos, sin)

    return None  # OPT 返回 None

# ==========================================
# 3. catcher捕获
# ==========================================
def prepare_model_inputs0(model, testenc, dev, seqlen, nsamples):
    """
    预处理并捕获模型第一层的输入 (Hidden States) 及相关参数 (Mask, PIDs)。

    Args:
        model: LLM 模型实例
        testenc: 测试数据 (Tensor 或 BatchEncoding)
        dev: 设备 (cuda device)
        seqlen: 序列长度
        nsamples: 采样数量

    Returns:
        inps: (nsamples, seqlen, hidden_size) 的 Tensor，位于 dev 上
        model_kwargs: 包含 attention_mask, position_ids 等参数的字典
    """
    # 1. 获取模型层列表 (假设你有 get_model_layers 函数，如果没有需替换为通用逻辑)
    layers = get_model_layers(model)

    # 2. 数据格式标准化
    if not isinstance(testenc, torch.Tensor):
        if hasattr(testenc, 'input_ids'):
            testenc = testenc.input_ids
        else:
            raise ValueError("testenc must be a Tensor or BatchEncoding")

    # 3. 初始化存储容器
    dtype = next(model.parameters()).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )

    # 缓存字典
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    # 4. 定义拦截器类 (Catcher)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            # 捕获关键参数
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError  # 拿到数据就报错跳出，节省计算

    # 5. 替换第一层并运行前向传播
    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass  # 预期的中断

    # 6. 还原第一层
    layers[0] = layers[0].module

    # 7. 构建 model_kwargs
    model_kwargs = {}

    # 处理 Attention Mask
    if cache['attention_mask'] is not None:
        model_kwargs['attention_mask'] = cache['attention_mask'].to(dev)

    # 处理 Position IDs (包含 LLaMA/Qwen 的兜底逻辑)
    if cache['position_ids'] is not None:
        model_kwargs['position_ids'] = cache['position_ids'].to(dev)
    else:
        # 如果模型需要 position_ids 但 Catcher 没抓到 (常见于部分 HuggingFace 版本)
        # 这里进行兜底生成
        model_kwargs['position_ids'] = torch.arange(
            seqlen, dtype=torch.long, device=dev
        ).unsqueeze(0)

    # 清理显存 (可选)
    torch.cuda.empty_cache()

    return inps, model_kwargs


def prepare_model_inputs(model, testenc, dev, seqlen, nsamples):
    """
    预处理并捕获模型第一层的输入 (Hidden States) 及相关参数 (Mask, PIDs)。
    修复了 Qwen2/Mistral 等模型检查 layer 属性导致的崩溃问题。

    Args:
        model: LLM 模型实例
        testenc: 测试数据 (Tensor 或 BatchEncoding)
        dev: 设备 (cuda device)
        seqlen: 序列长度
        nsamples: 采样数量

    Returns:
        inps: (nsamples, seqlen, hidden_size) 的 Tensor，位于 dev 上
        model_kwargs: 包含 attention_mask, position_ids 等参数的字典
    """
    # 1. 获取模型层列表
    layers = get_model_layers(model)

    # 2. 数据格式标准化
    if not isinstance(testenc, torch.Tensor):
        if hasattr(testenc, 'input_ids'):
            testenc = testenc.input_ids
        else:
            raise ValueError("testenc must be a Tensor or BatchEncoding")

    # 3. 初始化存储容器
    dtype = next(model.parameters()).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )

    # 缓存字典
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    # 4. 定义拦截器类 (Catcher)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

            # ================= [修复关键] =================
            # Qwen2/Mistral 在 model.forward 循环中会检查这些属性
            # 必须从原 layer 复制过来，否则会报 AttributeError
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type
            if hasattr(module, "self_attn"):
                self.self_attn = module.self_attn
            # =============================================

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            # 捕获关键参数
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError  # 拿到数据就报错跳出

    # 5. 替换第一层并运行前向传播
    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass  # 预期的中断

    # 6. 还原第一层
    layers[0] = layers[0].module

    # 7. 构建 model_kwargs
    model_kwargs = {}

    # 处理 Attention Mask
    if cache['attention_mask'] is not None:
        model_kwargs['attention_mask'] = cache['attention_mask'].to(dev)

    # 处理 Position IDs (包含 LLaMA/Qwen 的兜底逻辑)
    if cache['position_ids'] is not None:
        model_kwargs['position_ids'] = cache['position_ids'].to(dev)
    else:
        # 如果模型需要 position_ids 但 Catcher 没抓到 (常见于部分 HuggingFace 版本)
        # 这里进行兜底生成
        model_kwargs['position_ids'] = torch.arange(
            seqlen, dtype=torch.long, device=dev
        ).unsqueeze(0)

    # 清理显存 (可选)
    torch.cuda.empty_cache()

    return inps, model_kwargs


# ==========================================
# [新增] ViT 专用模型工具
# ==========================================

def get_model_vit(model_path, device="cpu"):
    """
    加载视觉模型 (AutoModelForImageClassification)
    """
    from transformers import AutoModelForImageClassification, AutoConfig

    logger.info(f"正在加载 ViT 模型: {model_path} ...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 自动选择 Image Classification 头
    model = AutoModelForImageClassification.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,  # ViT 对 FP16 有时敏感，建议先用 FP32，或者根据 GPU 自动选
        trust_remote_code=True,
        device_map=None
    )
    model.eval()
    return model


def get_model_layers_vit(model):
    """
    提取 ViT 的 Transformer Encoder Layers
    """
    # 常见结构: model.vit.encoder.layer (Google ViT, DeiT, BEiT)
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        return model.vit.encoder.layer

    # Swin Transformer: model.swin.encoder.layers (结构较复杂，由 Blocks 组成)
    # 这里先只支持标准 ViT

    raise ValueError(f"无法定位 ViT 层结构: {type(model)}")


def prepare_model_inputs_vit(model, test_loader, dev, nsamples):
    """
    [ViT版 Catcher]
    捕获第一层的输入。
    注意：ViT 的输入是 pixel_values，经过 Embeddings 层后变成 (Batch, SeqLen, Hidden)。
    """
    layers = get_model_layers_vit(model)
    dtype = next(model.parameters()).dtype

    # 确定 Embedding 后的序列长度 (SeqLen = NumPatches + 1 [CLS])
    # 我们先跑一个样本看看形状
    dummy_pixel = test_loader[0][0].to(dev, dtype=dtype)  # [1, 3, 224, 224]

    # 定义 Catcher
    cache = {'inps': [], 'count': 0}

    class VitCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            # hidden_states shape: [Batch, NumPatches, Hidden]
            # 我们把它存下来。为了节省显存，可以先存 CPU，或者如果是 H200 直接存 GPU
            cache['inps'].append(hidden_states.detach())  # 保持在 GPU
            cache['count'] += 1
            raise ValueError  # 中断

    # 替换
    layers[0] = VitCatcher(layers[0])

    # 运行
    for i in range(nsamples):
        pixel_values, _ = test_loader[i]
        pixel_values = pixel_values.to(dev, dtype=dtype)
        try:
            model(pixel_values)
        except ValueError:
            pass

    # 还原
    layers[0] = layers[0].module

    # 拼接结果
    # cache['inps'] 是 list of [1, Seq, Dim]，拼成 [nsamples, Seq, Dim]
    inps = torch.cat(cache['inps'], dim=0)

    # ViT 通常不需要 attention_mask (全关注) 或 position_ids (已包含)
    # 所以 model_kwargs 可以为空，或者包含 head_mask 等
    model_kwargs = {}

    return inps, model_kwargs