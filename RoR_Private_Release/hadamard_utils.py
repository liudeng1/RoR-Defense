import torch
import torch.nn as nn
import scipy.linalg
import numpy as np
import pandas as pd
import os
from custom_logger import logger


# ==========================================
# Part 1: 数学工具函数
# ==========================================

def load_householder_config_from_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        logger.warning(f"CSV文件未找到或未指定: {csv_path}，将跳过自动加载。")
        return {}

    try:
        df = pd.read_csv(csv_path)
        config = {}
        for _, row in df.iterrows():
            layer_name = row['Layer']
            indices_str = str(row['Outlier_Indices'])
            if ";" in indices_str:
                indices = [int(x) for x in indices_str.split(";") if x.strip()]
            else:
                try:
                    if "[" in indices_str:
                        import ast
                        indices = ast.literal_eval(indices_str)
                    else:
                        indices = [int(indices_str)]
                except:
                    continue
            config[layer_name] = indices
        logger.info(f">>> 已从 CSV 加载 {len(config)} 个层的异常列配置。")
        return config
    except Exception as e:
        logger.error(f"读取 CSV 失败: {e}")
        return {}

def get_random_hadamard_matrix(size, seed=None):
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.seed()

    if (size & (size - 1)) != 0 and size != 0:
        raise ValueError(f"Size {size} must be power of 2")

    H = torch.from_numpy(scipy.linalg.hadamard(size)).float()
    H = H / torch.sqrt(torch.tensor(size).float())

    s = torch.randint(0, 2, (size,), generator=rng).float() * 2 - 1
    return H @ torch.diag(s)


def get_greedy_mixed_rotation_matrix(target_dim, seed=42):
    remaining_dim = target_dim
    blocks = []
    current_seed = seed

    curr_pow = 1
    while curr_pow * 2 <= target_dim:
        curr_pow *= 2

    while remaining_dim > 0:
        if curr_pow <= remaining_dim:
            block = get_random_hadamard_matrix(curr_pow, seed=current_seed)
            blocks.append(block)
            remaining_dim -= curr_pow
            current_seed += 1
        else:
            curr_pow //= 2

    return torch.block_diag(*blocks)


def generate_orthogonal_targets(dim, num_targets):
    targets = []
    base_size = 32
    while base_size < num_targets:
        base_size *= 2
        base_size *= 2

    H_base = torch.from_numpy(scipy.linalg.hadamard(base_size)).float()
    for i in range(num_targets):
        pattern = H_base[i]
        repeats = (dim + base_size - 1) // base_size
        target = pattern.repeat(repeats)[:dim]
        target = target / target.norm()
        targets.append(target)

    return torch.stack(targets)


def compute_householder_vector(target_dim, outlier_idx, target_vector=None):
    e = torch.zeros(target_dim)
    e[outlier_idx] = 1.0

    if target_vector is None:
        u = torch.ones(target_dim) / (target_dim ** 0.5)
    else:
        u = target_vector
        if abs(u.norm().item() - 1.0) > 1e-3:
            u = u / u.norm()

    v = e - u
    if v.norm() > 1e-6:
        v = v / v.norm()
    return v

# ==========================================
# Part 2: 自定义层类
# ==========================================

class RotatedLinear(nn.Linear):
    def __init__(self, original_linear, rotation_matrix):
        super().__init__(original_linear.in_features, original_linear.out_features,
                         bias=original_linear.bias is not None)
        self.weight.data = original_linear.weight.data
        if original_linear.bias is not None:
            self.bias.data = original_linear.bias.data
        self.register_buffer('rotation_matrix', rotation_matrix)

    def forward(self, x):
        x_rotated = torch.matmul(x, self.rotation_matrix.to(x.dtype))
        return nn.functional.linear(x_rotated, self.weight, self.bias)

class HouseholderLinear(nn.Linear):
    """
    [Householder] 混合精度加速版 (Fix: Weight Update Orientation)
    """

    def __init__(self, original_linear, outlier_indices, use_stack_acceleration=True):
        super().__init__(original_linear.in_features, original_linear.out_features,
                         bias=original_linear.bias is not None)

        self.in_features = original_linear.in_features
        self.use_wy_acceleration = use_stack_acceleration

        # 1. 生成向量 V (Float64)
        num_outliers = len(outlier_indices)

        # KEY FIX: Return early if empty!
        if num_outliers == 0:
            self.register_buffer('vs', torch.zeros(0, self.in_features))
            self.register_buffer('ts', torch.zeros(0, 0))

            # [修复] 显式确保权重类型跟随原始层，但为了保险，forward里还会做一次check
            self.weight.data = original_linear.weight.data
            if original_linear.bias is not None:
                self.bias.data = original_linear.bias.data
            return

        target_vectors = generate_orthogonal_targets(self.in_features, num_outliers).double()

        vectors = []
        for i, idx in enumerate(outlier_indices):
            target_u = target_vectors[i].double()
            v = compute_householder_vector(self.in_features, idx, target_vector=target_u).double()
            vectors.append(v)

        V = torch.stack(vectors).t()  # (N, K) Float64
        K = num_outliers

        # 2. 计算 T 矩阵 (Float64)
        if self.use_wy_acceleration:
            T = torch.zeros((K, K), dtype=torch.float64, device=V.device)
            tau = 2.0

            for i in range(K):
                T[i, i] = tau
            for i in range(1, K):
                v_curr = V[:, i].view(-1, 1)
                V_prev = V[:, :i]
                proj = torch.matmul(V_prev.t(), v_curr)
                term = torch.matmul(T[:i, :i], proj)
                T[:i, i] = -tau * term.squeeze(1)

            self.register_buffer('ts', T.float())

            # 【修复Bug】T 不是对称矩阵，离线计算 W_proj 必须乘以 T 的转置 (T.t())
            # 因为 forward 中我们用的是 W_proj.t()，即 (V @ T.t()).t() = T @ V.t()
            W_proj = torch.matmul(V.float(), T.float().t())
            self.register_buffer('w_proj', W_proj)
        else:
            self.register_buffer('ts', torch.zeros(0, 0))

        self.register_buffer('vs', V.float().t())  # (K, N)

        # 3. 离线旋转权重 W (使用 FP64)
        with torch.no_grad():
            W_orig = original_linear.weight.data.double()
            if original_linear.bias is not None:
                self.bias.data = original_linear.bias.data.clone()

            device = W_orig.device
            V_mat = self.vs.t().to(device).double()  # (N, K)

            if self.use_wy_acceleration:
                T_mat = self.ts.to(device).double()  # (K, K)

                # W_new = W @ Q^T ???
                # 我们需要: (x Q) (W_new)^T = x W^T
                # => x Q W_new^T = x W^T => W_new = W Q
                # Q = I - V T V^T
                # W_new = W (I - V T V^T) = W - W V T V^T

                A = torch.matmul(W_orig, V_mat)

                # 【核心修复】这里不要转置 T！
                # 我们需要应用 Q (Forward用的那个)，而不是 Q^T
                B = torch.matmul(A, T_mat)

                correction = torch.matmul(B, V_mat.t())
                W_new = W_orig - correction
            else:
                # 串行模式
                W_curr = W_orig
                for i in range(K):
                    v = V_mat[:, i]
                    w_dot = torch.matmul(W_curr, v)
                    correction = torch.outer(w_dot, v)
                    W_curr -= 2.0 * correction
                W_new = W_curr

            self.weight.data = W_new.to(original_linear.weight.dtype)

    def forward(self, x):
        # [优化 Method 3] 全程无 FP32 转换，减少显存带宽压力


        # 2. 准备旋转矩阵
        # x.dtype 可能是 float16 或 bfloat16
        # 我们将 V 和 T 强转为与 x 一致，避免 PyTorch 自动向上转型到 fp32
        dtype = x.dtype
        device = x.device



        # logger.info(f"forward>>>")
        # 1. 快速路径：如果没有 Outlier，直接由底层 C++ Kernel 执行 Linear
        baseline_speed_test = False  # 设置基准，不进入下面的W_Y操作，可作为对照组
        if baseline_speed_test == True:  # self.vs.shape[0] == 0
            # logger.info(f"只进入不旋转！")
            w_cast = self.weight.to(dtype=x.dtype)
            b_cast = self.bias.to(dtype=x.dtype) if self.bias is not None else None

            return nn.functional.linear(x, w_cast, b_cast)



        # vs 在 buffer 中是 [K, N]，转置为 [N, K]
        V = self.vs.t().to(device=device, dtype=dtype)

        if self.use_wy_acceleration:
            T = self.ts.to(device=device, dtype=dtype)

            # [低秩更新公式] X_new = X - (X @ V) @ T @ V.T
            # X: [B, S, N]
            # V: [N, K]

            # Step A: 降维 [B, S, N] -> [B, S, K]
            # 这一步计算量最大，但因为 K 很小，比全矩阵乘法快得多
            X_V = torch.matmul(x, V)

            # Step B: 核心混合 [B, S, K] -> [B, S, K]
            X_T = torch.matmul(X_V, T)

            # Step C: 升维 [B, S, K] -> [B, S, N]
            correction = torch.matmul(X_T, V.t())

            # Step D: 修正
            # 注意：不使用 in-place 操作 (x -= ...)，防止影响残差连接
            x_new = x - correction

        else:
            # 串行备用模式 (一般不用)
            x_curr = x
            for i in range(self.vs.shape[0]):
                v = V[:, i]
                dot = torch.matmul(x_curr, v)
                correction = 2.0 * dot.unsqueeze(-1) * v
                x_curr = x_curr - correction
            x_new = x_curr

        # 3. 线性变换
        return nn.functional.linear(x_new, self.weight, self.bias)

def generate_and_evaluate_ror_exact_whitebox_fp(model, layers, initial_input, testenc,
                                                dev, seqlen, nsamples, raw_target,
                                                model_kwargs=None):
    """
    RoR 精确白盒攻击（全精度版，不涉及量化）。

    目标：在 RoR 模型的 W' 上施加修改，精确复现原始单 bit 在原模型上的攻击效果。

    数学：
        W' = W @ Q，前向 output = (x @ Q) @ W'^T = x @ W^T
        原始攻击：W[row, col] += δ，造成 Δoutput[:, row] = x[:, col] * δ
        等价白盒：ΔW'[row, :] = δ * Q[col, :]
        低秩展开：Q[col, :] = e_col - (V[col,:] @ T) @ V^T

    raw_target: [seq_no, layer_name_raw, [row, col], bit, delta]
        delta = W[row,col] 翻转后的浮点权重变化，从 baseline 日志里读取
    """
    if model_kwargs is None:
        model_kwargs = {}

    dtype = next(model.parameters()).dtype

    # ── 解析 raw_target ──────────────────────────────────────────
    seq_no, layer_name_raw, (row, col), bit, delta_original = raw_target

    if delta_original == 0:
        raise ValueError("raw_target[4] 为 0，请先跑 baseline 获取 delta 后填入。")

    layer_idx = int(layer_name_raw.split('_')[0])
    layer_i = layer_idx
    sub_path = layer_name_raw[len(f"{layer_idx}_"):]

    logger.info(f"[RoR FP Attack] layer={sub_path}, W[{row},{col}], δ={delta_original:.6f}")

    # ── 找目标模块 ────────────────────────────────────────────────
    target_layer = layers[layer_i]
    subset = find_layers(target_layer, layers=[nn.Linear])
    if sub_path not in subset:
        raise ValueError(f"'{sub_path}' not found. Available: {list(subset.keys())}")
    module = subset[sub_path]

    if not isinstance(module, HouseholderLinear):
        raise ValueError(f"模块 {sub_path} 不是 HouseholderLinear，是 {type(module).__name__}。")

    # ── 计算 direction = Q[col, :]（Q 的第 col 行）────────────────
    # Q = I - V T V^T，V = vs.t(): [N, K]，T = ts: [K, K]
    # Q[col, :] = e_col - (V[col,:] @ T) @ V^T
    V = module.vs.t().float().to(dev)  # [N, K]
    T_mat = module.ts.float().to(dev)  # [K, K]
    in_features = module.in_features

    e_col = torch.zeros(in_features, dtype=torch.float32, device=dev)
    e_col[col] = 1.0

    e_col_V = V[col, :]  # [K]
    # e_col_V_T = e_col_V @ T_mat  # [K]
    # correction = e_col_V_T @ V.t()  # [N]
    # direction = e_col - correction  # [N]，= Q[col, :]
    e_col_V_T = T_mat.t() @ e_col_V  # T^T V^T e_col → Q 的列
    correction = V @ e_col_V_T
    direction = e_col - correction

    logger.info(f"[RoR FP Attack] direction_norm={direction.norm().item():.6f} (应 ≈ 1.0)")

    # ── 施加修改（全精度，直接加到 W'[row, :]）───────────────────
    delta_row = (delta_original * direction).to(dtype).to(dev)  # [N]
    logger.info(f"[RoR FP Attack] ||delta_row||={delta_row.norm().item():.6f} (应 ≈ |δ|={abs(delta_original):.6f})")

    original_row = module.weight.data[row].clone().cpu()

    # with torch.no_grad():
    #     module.weight.data[row] = (module.weight.data[row].float() + delta_row).to(dtype)

    # 施加修改之前
    logger.info(f"[验证] 修改前 W'[{row},{col}]={module.weight.data[row, col].item():.6f}")
    logger.info(f"[验证] 修改前 W'[{row},0:5]={module.weight.data[row, :5].tolist()}")

    with torch.no_grad():
        module.weight.data[row] = (module.weight.data[row].float() + delta_row).to(dtype)

    # 施加修改之后
    logger.info(f"[验证] 修改后 W'[{row},{col}]={module.weight.data[row, col].item():.6f}")
    logger.info(f"[验证] 修改后 W'[{row},0:5]={module.weight.data[row, :5].tolist()}")
    logger.info(f"[验证] delta_row[:5]={delta_row[:5].tolist()}")
    logger.info(f"[验证] delta_row[{col}]={delta_row[col].item():.6f}")
    logger.info(f"[验证] direction[{col}]={direction[col].item():.6f}  (应≈1.0，因为Q正交e_col@Q[col]=1)")



    logger.info(f"[RoR FP Attack] 已施加修改到 W'[{row}, :] of {sub_path}")

    # ================= 增加：精确统计改变的权重数与真实 Bit 翻转数 =================
    import numpy as np

    # 1. 统计改变了多少个权重 (设定 1e-9 为浮点误差阈值)
    changed_weights_count = torch.sum(torch.abs(delta_row) > 1e-9).item()

    # 2. 计算底层真实的 Bit 翻转数 (Bitwise Hamming Distance)
    # [修复]: 解决 bfloat16 无法直接 .numpy() 的问题
    # 策略：在 PyTorch 层面直接 view 成整型，再转 numpy
    dtype_orig = original_row.dtype

    try:
        if dtype_orig == torch.float32:
            orig_view = original_row.view(torch.int32).cpu().numpy()
            new_view = module.weight.data[row].view(torch.int32).cpu().numpy()
        elif dtype_orig in [torch.float16, torch.bfloat16]:
            orig_view = original_row.view(torch.int16).cpu().numpy()
            new_view = module.weight.data[row].view(torch.int16).cpu().numpy()
        else:
            orig_view = original_row.cpu().numpy().view(np.uint8)
            new_view = module.weight.data[row].cpu().numpy().view(np.uint8)

        # 计算异或 (XOR) 并统计结果中二进制 '1' 的个数
        xor_diff = orig_view ^ new_view
        actual_flipped_bits = np.unpackbits(xor_diff.view(np.uint8)).sum()

        logger.info(f"   >>> [统计] 理论破坏范围：改变了 {changed_weights_count} 个权重的值。")
        logger.info(f"   >>> [统计] 底层物理代价：IEEE表示法下真实翻转了 {actual_flipped_bits} 个 Bit！")
    except Exception as e:
        logger.warning(f"   >>> [统计] 计算 Bit 翻转数失败，但这不影响攻击逻辑: {e}")
    # ==============================================================================



    # ── 前向传播（全精度，不量化）────────────────────────────────
    current_output = initial_input.to(dev, dtype=dtype)

    try:
        with torch.no_grad():
            for layer in layers:
                layer = layer.to(dev)
                layer_output_list = []
                for j in range(nsamples):
                    inp = current_output[j].unsqueeze(0).to(dev)
                    out = layer(inp, **model_kwargs)[0]
                    layer_output_list.append(out.squeeze(0))
                current_output = torch.stack(layer_output_list)

        loss, ppl = compute_loss(
            model, current_output, testenc, dev,
            seqlen, nsamples, model.config.model_type
        )
        logger.info(f"[RoR FP Attack] Loss={loss:.6f}, PPL={ppl:.6f}")

    finally:
        with torch.no_grad():
            module.weight.data[row] = original_row.to(dev).to(dtype)
        logger.info(f"[RoR FP Attack] 已恢复 W'[{row}, :] of {sub_path}。")

    return loss, ppl


# ==========================================
# Part 3: 注入函数 (不变)
# ==========================================

def inject_rotation_into_model(model, seed=42, rotation_config=None):
    if rotation_config is None: rotation_config = {}

    logger.info(">>> 开始注入旋转矩阵 (Mixed Precision WY + Fix)...")

    mode = rotation_config.get("hadamard_mode", "None")
    hadamard_list = rotation_config.get("hadamard_target_layers", [])

    use_wy = rotation_config.get("use_stack_acceleration", False)

    if use_wy:
        logger.info(">>> [Config] 模式: Mixed-Precision WY (快速、精确、已修复转置Bug)。")
    else:
        logger.info(">>> [Config] 模式: Serial Loop (串行基准)。")

    householder_map = rotation_config.get("householder_config", {}).copy()
    csv_path = rotation_config.get("householder_csv_path", None)

    if csv_path:
        csv_config = load_householder_config_from_csv(csv_path)
        householder_map.update(csv_config)

    count_householder = 0
    count_hadamard = 0

    torch.manual_seed(seed)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            if isinstance(module, (RotatedLinear, HouseholderLinear)): continue

            is_householder_target = False
            target_outliers = []

            for config_name, outliers in householder_map.items():
                if name.endswith(config_name):
                    is_householder_target = True
                    target_outliers = outliers
                    break

            if is_householder_target:
                new_layer = HouseholderLinear(module, target_outliers, use_stack_acceleration=use_wy)
                replace_module(model, name, new_layer)

                mode_str = "WY-Mixed" if use_wy else "Serial"
                logger.info(f"   [Householder] 层 {name} 命中，消除 {len(target_outliers)} 列 ({mode_str})")

                count_householder += 1
                continue

            should_apply_hadamard = False
            in_hadamard_list = any(name.endswith(suffix) for suffix in hadamard_list)

            if mode == "All":
                should_apply_hadamard = True
            elif mode == "Use":
                if in_hadamard_list: should_apply_hadamard = True
            elif mode == "Nouse":
                if not in_hadamard_list: should_apply_hadamard = True

            if should_apply_hadamard:
                in_dim = module.in_features
                R = get_greedy_mixed_rotation_matrix(in_dim, seed=seed)
                new_layer = RotatedLinear(module, R.to(module.weight.device))
                replace_module(model, name, new_layer)
                with torch.no_grad():
                    new_layer.weight.data = torch.matmul(module.weight.data,
                                                         R.to(module.weight.device).to(module.weight.dtype))
                count_hadamard += 1
                seed += 1

    logger.info(f">>> 注入完成！统计: Householder={count_householder}, Hadamard={count_hadamard}")
    return model

def inject_rotation_into_model_debug(model, seed=42, rotation_config=None):
    """
    [DEBUG ONLY] Forces every Linear layer to be a HouseholderLinear with 0 outliers.
    This guarantees execution of the 'if self.vs.shape[0] == 0' branch.
    """
    logger.info("=" * 60)
    logger.info(">>> [DEBUG] Forcing Empty HouseholderLinear Injection...")
    logger.info(">>> [DEBUG] GOAL: Verify Zero-Overhead Fast Path")
    logger.info("=" * 60)

    count = 0
    # Use list() to avoid iterator modification issues during replacement
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            # Skip if already replaced
            if isinstance(module, (HouseholderLinear, RotatedLinear)): continue

            # Force empty outliers
            target_outliers = []

            # Create layer (will hit the early return in __init__)
            new_layer = HouseholderLinear(module, target_outliers, use_stack_acceleration=True)

            # Replace
            replace_module(model, name, new_layer)
            count += 1

    logger.info(f">>> [DEBUG] Replaced {count} layers with Empty HouseholderLinear.")
    return model

def replace_module(model, name, new_module):
    if '.' in name:
        parent_name = name.rsplit('.', 1)[0]
        child_name = name.rsplit('.', 1)[1]
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, new_module)
    else:
        setattr(model, name, new_module)


"""
RoR 精确白盒攻击：在旋转空间 W' 上精确复现原始单 bit 攻击效果。

数学原理：
    原始模型存储 W，RoR 模型存储 W' = W @ Q（Q 为 Householder 正交矩阵）
    原始攻击：翻转 W[row, col] 的 bit，造成数值变化 δ_original
    等价白盒攻击：在 W' 第 row 行施加 ΔW'[row, j] = δ_original * Q[col, j]
    数学保证：两者产生完全相同的输出误差（对所有输入 x 成立）
"""

# ============================================================
#  核心函数 1：生成精确攻击参数
# ============================================================
from bit_flip_utils import get_model_layers, find_layers, compute_loss
from quant import real_quantize
from biterror1.ttt_7_18_all_value_flip import Error_injection_fewbit
def generate_and_evaluate_ror_exact_whitebox(model, layers, quantizers, initial_input, testenc,
                                              dev, wbits, seqlen, nsamples, raw_target,
                                              model_kwargs=None):
    """
    RoR 精确白盒攻击，量化路径与 flip_and_evaluate_fewbit 完全一致。
    delta_original 从 raw_target[4] 读取（先跑 baseline 获得）。

    数学原理：
        W' = W @ Q，Q = I - V T V^T
        前向：output = (x @ Q) @ W'^T = x @ W^T
        原始攻击翻转 W[row, col]，造成 Δoutput[:, row] = x[:, col] * δ
        等价白盒：ΔW'[row, :] = δ * Q[col, :]  （Q 的第 col 行）
        低秩展开：Q[col, :] = e_col - (V[col,:] @ T) @ V^T
    """
    if model_kwargs is None:
        model_kwargs = {}

    dtype = next(model.parameters()).dtype

    # ── 解析 raw_target ──────────────────────────────────────────
    seq_no, layer_name_raw, (row, col), bit, delta_original = raw_target

    if delta_original == 0:
        raise ValueError("raw_target[4] 为 0，请先跑 baseline 获取 delta 后填入。")

    # 从 layer_name_raw 解析层索引（raw_target[0] 是序号，不是层索引）
    layer_idx = int(layer_name_raw.split('_')[0])   # '1_self_attn.v_proj' → 1
    layer_i = layer_idx                         # 0-based
    sub_path = layer_name_raw[len(f"{layer_idx}_"):]  # 'self_attn.v_proj'
    strict_key = f"{layer_i}_{sub_path}"

    logger.info(
        f"[RoR ExactAttack] seq_no={seq_no}, layer_idx={layer_idx}, "
        f"sub_path={sub_path}, δ={delta_original:.6f}"
    )

    # ── 找目标模块（复用 find_layers）────────────────────────────
    target_layer = layers[layer_i]
    subset = find_layers(target_layer, layers=[nn.Linear])
    if sub_path not in subset:
        raise ValueError(f"'{sub_path}' not found. Available: {list(subset.keys())}")
    module = subset[sub_path]

    if not isinstance(module, HouseholderLinear):
        raise ValueError(f"模块 {sub_path} 不是 HouseholderLinear，是 {type(module).__name__}。")

    # ── 获取 quantizer ────────────────────────────────────────────
    quantizer = quantizers.get(strict_key) or quantizers.get(f"{layer_i}.{sub_path}")
    if quantizer is None:
        raise ValueError(f"Quantizer not found for '{strict_key}'.")

    # ── 计算 direction = Q[col, :]（Q 的第 col 行，低秩展开）──────
    # Q = I - V T V^T，V = vs.t(): [N, K]，T = ts: [K, K]
    # Q[col, :] = e_col @ (I - V T V^T) = e_col - (e_col @ V) @ T @ V^T
    #           = e_col - V[col, :] @ T @ V^T
    V = module.vs.t().float().to(dev)      # [N, K]
    T_mat = module.ts.float().to(dev)      # [K, K]
    in_features = module.in_features

    e_col = torch.zeros(in_features, dtype=torch.float32, device=dev)
    e_col[col] = 1.0

    e_col_V = V[col, :]                    # e_col @ V，shape [K]
    e_col_V_T = e_col_V @ T_mat            # [K]
    correction_vec = e_col_V_T @ V.t()     # [N]
    direction = e_col - correction_vec     # = Q[col, :]，Q 的第 col 行  ✅

    direction_norm = direction.norm().item()
    logger.info(f"[RoR ExactAttack] direction_norm={direction_norm:.6f} (应 ≈ 1.0)")

    delta_row = (delta_original * direction).to(dtype)   # [N]
    logger.info(
        f"[RoR ExactAttack] ||delta_row||={delta_row.norm().item():.6f} "
        f"(应 ≈ |δ|={abs(delta_original):.6f})"
    )

    # ── 量化路径（与 flip_and_evaluate_fewbit 一致）──────────────
    scale = quantizer.scale.to(dev, dtype=dtype)
    if scale.dim() == 1:
        scale = scale.unsqueeze(1)

    W_prime = module.weight.data                          # 原始浮点 W'
    layer_backup = W_prime.cpu().clone()                  # 备份整个矩阵

    W_quant = real_quantize(W_prime, scale, wbits, dev)   # 量化为整型
    W_dequant = (W_quant * scale).to(dtype)               # dequant 回浮点

    # [新增]：备份施加误差前的反量化权重，用于统计翻转了多少 bit
    original_dequant_row = W_dequant[row].clone().cpu()

    # 在 dequant 域的第 row 行施加 delta_row
    W_dequant[row] = W_dequant[row] + delta_row.to(dev)
    module.weight.data = W_dequant

    logger.info(f"[RoR ExactAttack] 已施加修改到 W'[{row}, :] of {sub_path} (layer {layer_idx})")

    # ================= 增加：精确统计改变的权重数与真实 Bit 翻转数 =================
    import numpy as np

    # 1. 统计改变了多少个权重 (设定 1e-9 为浮点误差阈值)
    changed_weights_count = torch.sum(torch.abs(delta_row) > 1e-9).item()

    # 2. 计算底层真实的 Bit 翻转数 (Bitwise Hamming Distance)
    # 针对反量化后的连续浮点数统计物理代价
    new_dequant_row = W_dequant[row].cpu()
    dtype_orig = original_dequant_row.dtype

    try:
        # 使用 view 完美绕过 BFloat16 到 numpy 的转换限制
        if dtype_orig == torch.float32:
            orig_view = original_dequant_row.view(torch.int32).numpy()
            new_view = new_dequant_row.view(torch.int32).numpy()
        elif dtype_orig in [torch.float16, torch.bfloat16]:
            orig_view = original_dequant_row.view(torch.int16).numpy()
            new_view = new_dequant_row.view(torch.int16).numpy()
        else:
            orig_view = original_dequant_row.numpy().view(np.uint8)
            new_view = new_dequant_row.numpy().view(np.uint8)

        # 计算异或 (XOR) 并统计结果中二进制 '1' 的个数
        xor_diff = orig_view ^ new_view
        actual_flipped_bits = np.unpackbits(xor_diff.view(np.uint8)).sum()

        logger.info(f"   >>> [统计] 理论破坏范围：改变了 {changed_weights_count} 个权重的值。")
        logger.info(f"   >>> [统计] 等效浮点代价：底层物理表示下真实翻转了 {actual_flipped_bits} 个 Bit！")
    except Exception as e:
        logger.warning(f"   >>> [统计] 计算 Bit 翻转数失败: {e}")
    # ==============================================================================


    # ── 完整前向传播 ─────────────────────────────────────────────
    current_output = initial_input.to(dev, dtype=dtype)

    try:
        with torch.no_grad():
            for layer in layers:
                layer = layer.to(dev)
                layer_output_list = []
                for j in range(nsamples):
                    inp = current_output[j].unsqueeze(0).to(dev)
                    out = layer(inp, **model_kwargs)[0]
                    layer_output_list.append(out.squeeze(0))
                current_output = torch.stack(layer_output_list)

        loss, ppl = compute_loss(
            model, current_output, testenc, dev,
            seqlen, nsamples, model.config.model_type
        )
        logger.info(f"[RoR ExactAttack] Loss={loss:.6f}, PPL={ppl:.6f}")

    finally:
        with torch.no_grad():
            module.weight.data = layer_backup.to(dev)
        logger.info(f"[RoR ExactAttack] 已恢复 W' of {sub_path}。")

    return loss, ppl


