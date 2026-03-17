import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_logger import logger
from quant import real_quantize  # [关键] 导入项目统一的真量化函数


class RADARLinear(nn.Linear):
    """
    [RADAR Defense Layer]
    Paper: RADAR: Run-time Adversarial Weight Attack Detection and Accuracy Recovery

    Features:
    - Replaces standard nn.Linear
    - Implements Checksum-based Detection (Signature)
    - Implements Interleaving & Masking
    - Implements Zero-out Recovery
    - [Aligned] Strictly uses `real_quantize` for integer conversion
    """

    def __init__(self, original_linear: nn.Linear, scale: torch.Tensor,
                 G=512, interleave=True, secret_seed=42, wbits=8, debug=False):

        super().__init__(
            original_linear.in_features,
            original_linear.out_features,
            bias=original_linear.bias is not None
        )

        # 1. 继承权重 (从原始模型复制)
        self.weight.data = original_linear.weight.data.clone()
        if original_linear.bias is not None:
            self.bias.data = original_linear.bias.data.clone()

        # 2. 保存配置
        self.G = G
        self.interleave = interleave
        self.seed = secret_seed
        self.wbits = wbits
        self.debug = debug

        # 3. 注册量化参数 (Scale)
        # [关键] real_quantize 要求 Per-Channel Scale 必须是 [Out, 1] 形状
        if scale.dim() == 1:
            scale = scale.view(-1, 1)
        self.register_buffer('scale', scale)

        # 4. 注册 RADAR 状态 Buffer
        self.register_buffer('golden_sig_a', None)
        self.register_buffer('golden_sig_b', None)
        self.register_buffer('mask_vector', None)
        # [Optimization] Removed explicit storage of perm_indices to save memory
        # self.register_buffer('perm_indices', None)

        # 5. 初始化 RADAR (计算签名)
        self._init_radar()

    def _quantize_to_int(self, w_float):
        """
        [Aligned] 将量化逻辑全权委托给 `real_quantize`。
        确保 RADAR 看到的整数值与量化推理时的整数值 100% 一致。
        """
        # 1. 调用统一量化器 (返回 int16)
        # w_float: [Out, In], self.scale: [Out, 1] -> 广播机制自动处理
        w_int16 = real_quantize(w_float, self.scale, self.wbits, w_float.device)

        # 2. 转为 int32
        # Checksum 计算需要累加 512 个数值，int16 可能会溢出，所以转为 int32
        return w_int16.to(torch.int32)

    def _get_interleaved_indices(self, total_len, device):
        stride = self.G
        pad_len = (stride - (total_len % stride)) % stride
        padded_len = total_len + pad_len
        indices = torch.arange(padded_len, device=device)
        num_rows = padded_len // stride
        indices = indices.view(num_rows, stride).t().contiguous().view(-1)
        if pad_len > 0:
            indices = indices[indices < total_len]
        return indices

    def _init_radar(self):
        """初始化：生成 Mask、重排索引、计算 Golden Signature"""
        with torch.no_grad():
            w_flat = self.weight.view(-1)
            numel = w_flat.numel()
            device = self.weight.device

            # 1. 生成交织索引 (Interleaving) - 动态生成，不注册
            if self.interleave:
                indices = self._get_interleaved_indices(numel, device)
            else:
                indices = torch.arange(numel, device=device)
            # self.register_buffer('perm_indices', indices) # Removed

            # 2. 生成随机 Mask (+1/-1)
            gen = torch.Generator(device='cpu')
            gen.manual_seed(self.seed)
            mask = torch.randint(0, 2, (numel,), generator=gen).to(device) * 2 - 1
            self.register_buffer('mask_vector', mask.to(torch.int8))

            # 3. 计算并保存 Golden Signature
            # 这里调用 _quantize_to_int，间接调用了 real_quantize
            w_int = self._quantize_to_int(self.weight)
            # 传入临时生成的 indices
            sig_a, sig_b = self._compute_signature_internal(w_int, indices)

            self.register_buffer('golden_sig_a', sig_a)
            self.register_buffer('golden_sig_b', sig_b)

    def _compute_signature_internal(self, w_int_tensor, indices):
        """计算签名的核心逻辑 (基于整数)"""
        w_flat = w_int_tensor.view(-1)
        # 使用传入的 indices
        w_permuted = w_flat[indices]
        mask_permuted = self.mask_vector[indices]

        numel = w_permuted.numel()
        pad_len = (self.G - (numel % self.G)) % self.G
        if pad_len > 0:
            w_permuted = F.pad(w_permuted, (0, pad_len), value=0)
            mask_permuted = F.pad(mask_permuted, (0, pad_len), value=0)

        w_groups = w_permuted.view(-1, self.G)
        mask_groups = mask_permuted.view(-1, self.G)

        # Checksum 计算
        M = (w_groups * mask_groups).sum(dim=1)

        # 提取第 8 位和第 7 位作为签名
        sig_a = torch.div(M, 256, rounding_mode='floor') % 2
        sig_b = torch.div(M, 128, rounding_mode='floor') % 2

        return sig_a.to(torch.int8), sig_b.to(torch.int8)

    def forward(self, x):
        """运行时前向传播：检测 + 恢复 + 计算"""
        # 1. Detection Phase (检测)
        # 实时量化当前权重 (可能已被攻击)
        w_curr_int = self._quantize_to_int(self.weight)

        # [Optimization] On-the-fly index generation
        numel = self.weight.numel()
        device = self.weight.device
        if self.interleave:
            current_indices = self._get_interleaved_indices(numel, device)
        else:
            current_indices = torch.arange(numel, device=device)

        curr_sig_a, curr_sig_b = self._compute_signature_internal(w_curr_int, current_indices)

        diff_a = (curr_sig_a != self.golden_sig_a)
        diff_b = (curr_sig_b != self.golden_sig_b)
        error_mask = (diff_a | diff_b)

        # 2. Recovery Phase (恢复)
        if error_mask.any():
            if self.debug:
                print(f"\n{'=' * 20} 🛡️ RADAR INTERCEPTION LOG 🛡️ {'=' * 20}")
                print(f"🚨 ALERT: Attack Detected in layer {self.weight.shape}!")

                bad_group_idx = torch.where(error_mask)[0][0].item()
                print(f"📍  Corrupted Group Index: {bad_group_idx}")

                g_a = self.golden_sig_a[bad_group_idx].item()
                g_b = self.golden_sig_b[bad_group_idx].item()
                c_a = curr_sig_a[bad_group_idx].item()
                c_b = curr_sig_b[bad_group_idx].item()
                print(f"🧾  Signature Mismatch:")
                print(f"    Golden (Safe): [A={g_a}, B={g_b}]")
                print(f"    Current (Bad): [A={c_a}, B={c_b}]")

                # 获取受损权重值用于 Debug
                w_flat = self.weight.view(-1)
                w_permuted = w_flat[current_indices] # 使用 current_indices
                w_bad_group = w_permuted.view(-1, self.G)[bad_group_idx]

                # [Fix] 转为 float 再转 numpy，避免 BFloat16 报错
                print(f"💣  Attacked Values (First 5): {w_bad_group[:5].detach().float().cpu().numpy()}")
                print(f"    -> Max Value: {w_bad_group.abs().max().item():.4f}")
                print(f"🚑  Action: Zero-out entire group...")

            # --- Zero-out Logic ---
            # 构造置零掩码
            group_mask = (~error_mask).float().unsqueeze(1).expand(-1, self.G).reshape(-1)
            if group_mask.numel() > numel:
                group_mask = group_mask[:numel]

            # 映射回物理地址
            # inv_indices = torch.argsort(self.perm_indices) # Old
            inv_indices = torch.argsort(current_indices) # New
            physical_mask = group_mask[inv_indices]

            if physical_mask.device != self.weight.device:
                physical_mask = physical_mask.to(self.weight.device)

            # 执行置零 (不修改 self.weight，只修改计算用的临时变量)
            w_recovered = self.weight.view(-1) * physical_mask
            w_final = w_recovered.view_as(self.weight)

            if self.debug:
                w_final_flat = w_final.view(-1)
                w_final_permuted = w_final_flat[current_indices] # 使用 current_indices
                w_fixed_group = w_final_permuted.view(-1, self.G)[bad_group_idx]
                print(f"✅  Recovered Values (First 5): {w_fixed_group[:5].detach().float().cpu().numpy()}")
                print(f"{'=' * 60}\n")

        else:
            w_final = self.weight

        # 3. Computation Phase (计算)
        # 确保类型匹配 (BFloat16/Float16 兼容)
        if w_final.dtype != x.dtype:
            w_final = w_final.to(dtype=x.dtype)

        return F.linear(x, w_final, self.bias)


def apply_radar_defense(model, quantizers, dev, G=512, interleave=True, debug=False):
    """
    [Integration] 将 Llama 模型中的 Linear 层替换为 RADARLinear。
    """
    debug = False
    logger.info(f"🛡️ [RADAR] Applying RADAR Defense... (G={G}, Interleave={interleave}, Debug={debug})")

    # 获取模型的 layers 列表以便索引
    if hasattr(model, 'model'):
        layers = model.model.layers if hasattr(model.model, 'layers') else model.model.decoder.layers
    elif hasattr(model, 'layers'):
        layers = model.layers  # OPT 有时直接是 model.layers
    elif hasattr(model, 'decoder'):
        layers = model.decoder.layers
    else:
        # 兜底：尝试从 named_modules 猜
        logger.warning("Could not find '.layers' attribute via standard paths. Iterating all modules.")
        layers = []

    replaced_count = 0
    for q_key, quantizer in quantizers.items():
        try:
            # 解析 Key: "0_self_attn.q_proj" -> layer_idx=0, sub_name="self_attn.q_proj"
            layer_idx_str, sub_name = q_key.split('_', 1)
            layer_idx = int(layer_idx_str)

            block = layers[layer_idx]
            parent = block
            sub_parts = sub_name.split('.')
            for p in sub_parts[:-1]:
                parent = getattr(parent, p)

            leaf_name = sub_parts[-1]
            old_linear = getattr(parent, leaf_name)

            if not isinstance(old_linear, nn.Linear):
                continue

            # 提取 Scale 并确保它是 detach 的
            scale = quantizer.scale.detach().to(dev)

            # 创建 RADAR 层
            radar_layer = RADARLinear(
                old_linear,
                scale=scale,
                G=G,
                interleave=interleave,
                secret_seed=42 + replaced_count,
                debug=debug
            ).to(dev)

            setattr(parent, leaf_name, radar_layer)
            replaced_count += 1

        except Exception as e:
            logger.warning(f"⚠️ [RADAR] Skipped {q_key}: {e}")
            continue

    logger.info(f"✅ [RADAR] Successfully protected {replaced_count} layers.")
    return model


def generate_radar_collision_attack(model, target_log, quantizers):
    """
    [White-box Attack - Fixed V4] 针对 RADAR 生成严格对齐的碰撞攻击对

    核心改进:
    1. 严格复刻 RADAR 的 Interleaving 逻辑，确保 Target 和 Partner 在同一个逻辑 Group。
    2. 使用 real_quantize 确保量化数值一致。
    3. 修复了 Scale 维度和分组计算的潜在 Bug。
    4. [Optimization] On-the-fly index generation logic mirrored from RADARLinear.
    """
    from radar_defense import RADARLinear

    layer_idx, layer_name, coords, target_bit, _ = target_log
    row, col = coords

    # 1. 定位目标层
    if hasattr(model, 'model'):
        layers = model.model.layers
    elif hasattr(model, 'decoder'):
        layers = model.decoder.layers
    else:
        layers = model.layers

    # 移除 layer_idx 前缀并定位模块
    block = layers[layer_idx]
    clean_sub_name = layer_name.split(f"{layer_idx}_")[1] if layer_name.startswith(f"{layer_idx}_") else layer_name

    target_module = block
    for p in clean_sub_name.split('.'):
        target_module = getattr(target_module, p)

    if not isinstance(target_module, RADARLinear):
        logger.error(f"❌ 目标层 {layer_name} 不是 RADARLinear，无法实施碰撞攻击！")
        return [target_log]

    # 2. 获取内部机密 & 重建映射
    radar = target_module
    device = radar.weight.device
    G = radar.G
    num_cols = radar.weight.shape[1]
    numel = radar.weight.numel()
    flat_idx_target = row * num_cols + col

    # [关键步骤] 严格复刻索引生成逻辑，确保分组 100% 对齐
    def get_perm_indices(total_len, stride, device):
        pad_len = (stride - (total_len % stride)) % stride
        padded_len = total_len + pad_len
        indices = torch.arange(padded_len, device=device)
        num_rows = padded_len // stride
        # RADAR 的核心逻辑: view -> transpose -> view
        indices = indices.view(num_rows, stride).t().contiguous().view(-1)
        if pad_len > 0:
            indices = indices[indices < total_len]
        return indices

    # 重新计算 perm_indices (由于去掉了 buffer，必须即时计算)
    perm_indices = get_perm_indices(numel, G, device)

    # 3. 寻找 Target 的逻辑位置 (Logical Index)
    # 我们需要找到 logical_idx，使得 perm_indices[logical_idx] == flat_idx_target
    # 这步比较耗时，但对于白盒攻击是必要的
    logical_locs = (perm_indices == flat_idx_target).nonzero(as_tuple=True)[0]
    if len(logical_locs) == 0:
        logger.error("❌ 无法在 perm_indices 中找到 Target，索引逻辑可能有误！")
        return [target_log]
    logical_idx_target = logical_locs.item()

    # 4. 确定真正的逻辑组 (Real Group ID)
    real_group_id = logical_idx_target // G

    # 获取 Target 的 Mask 和 量化值
    mask_target = radar.mask_vector[flat_idx_target].item()

    # 提取 Target 量化值
    w_tensor = radar.weight.data[row, col].view(1, 1)
    scale_tensor = radar.scale[row].view(1, 1)
    w_int = real_quantize(w_tensor, scale_tensor, radar.wbits, device).item()

    # 计算 Target 的 Checksum 贡献
    w_int_flipped = w_int ^ (1 << target_bit)
    delta_int = w_int_flipped - w_int
    delta_checksum = delta_int * mask_target

    logger.info(f"🕵️ [Radar-Hacker] Target: {coords} (Bit {target_bit})")
    logger.info(f"    PhysIdx: {flat_idx_target} -> LogicalIdx: {logical_idx_target} -> Group: {real_group_id}")
    logger.info(f"    Val: {w_int}->{w_int_flipped} (Δ={delta_int}) | Mask: {mask_target} | ΔSum: {delta_checksum}")

    # 5. 在同一个逻辑组内搜索共犯 (Partner)
    logger.info(f"    Searching in Logical Range [{real_group_id * G}, {(real_group_id + 1) * G})...")

    partner_log = None

    # 遍历该组内的所有逻辑索引
    start_logical = real_group_id * G
    end_logical = min(start_logical + G, numel)

    for log_idx in range(start_logical, end_logical):
        # 获取物理索引
        p_idx = perm_indices[log_idx].item()

        if p_idx == flat_idx_target: continue

        p_mask = radar.mask_vector[p_idx].item()

        # 计算所需的 Partner Delta
        # 方程: delta_checksum + (delta_partner * p_mask) = 0
        required_delta_int = -delta_checksum * p_mask

        # 简化攻击：只寻找能通过翻转同一 Bit (Target Bit) 满足要求的 Partner
        if abs(required_delta_int) != (1 << target_bit):
            continue

        p_row = p_idx // num_cols
        p_col = p_idx % num_cols

        # 获取 Partner 量化值
        p_tensor = radar.weight.data[p_row, p_col].view(1, 1)
        p_scale_tensor = radar.scale[p_row].view(1, 1)
        p_val_int = real_quantize(p_tensor, p_scale_tensor, radar.wbits, device).item()

        # 检查翻转可行性 (0->1 或 1->0)
        p_bit_val = (p_val_int >> target_bit) & 1

        can_flip = False
        if required_delta_int > 0 and p_bit_val == 0: can_flip = True
        if required_delta_int < 0 and p_bit_val == 1: can_flip = True

        if can_flip:
            logger.info(f"✅ Found Partner: [{p_row}, {p_col}] (LogIdx: {log_idx})")
            logger.info(f"    Mask: {p_mask} | Val: {p_val_int} | ReqDelta: {required_delta_int}")

            partner_log = [layer_idx, layer_name, [p_row, p_col], target_bit, 0]
            break

    if partner_log:
        return [target_log, partner_log]
    else:
        logger.warning("⚠️ Failed to find a suitable partner in this group (Try another target).")
        return [target_log]


def test_radar_functionality():
    print("=" * 60)
    print("🚀 启动 RADAR 算法功能性测试 (Aligned with real_quantize)")
    print("=" * 60)

    torch.manual_seed(1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 设置测试环境 ---
    # 定义一个小一点的层，方便观察
    in_features = 64
    out_features = 64
    G = 16  # Group Size
    interleave = True
    wbits = 8

    print(f"[Setup] Layer: {in_features}x{out_features}, GroupSize G={G}, Interleave={interleave}, Device={device}")

    # 创建原始 Linear
    original = nn.Linear(in_features, out_features).to(device)
    # 模拟量化参数 (Per-Channel Scale: [Out, 1])
    scale = (torch.rand(out_features, 1, device=device) * 0.005 + 0.001)

    # 实例化 RADAR (开启 Debug 模式)
    radar = RADARLinear(original, scale, G=G, interleave=interleave, wbits=wbits, debug=True).to(device)

    # 创建输入
    x = torch.randn(1, in_features, device=device)

    # --- 2. 测试: Clean Inference (一致性) ---
    print("\n>>> Test 1: Clean Inference (无攻击)")
    with torch.no_grad():
        # 注意: RADAR 前向传播内部会有量化误差(因为使用了int8计算签名过程可能涉及舍入? 不, RADAR只涉及检测)
        # RADAR forward 的计算逻辑是 F.linear(x, w, b)，用的还是 float 权重
        # 除非触发了恢复，否则输出应该完全一致
        y_orig = original(x)
        y_radar = radar(x)

    # 检查输出是否一致
    diff = (y_orig - y_radar).abs().max().item()
    if diff < 1e-6:
        print(f"✅ [PASS] Clean Output Matches (Diff: {diff:.9f})")
    else:
        print(f"❌ [FAIL] Clean Output Mismatch! (Diff: {diff:.9f})")
        return

    # --- 3. 测试: Attack & Detection (攻击检测) ---
    print("\n>>> Test 2: Injecting Bit-Flip Attack (注入攻击)")

    # 选一个倒霉的权重进行攻击 (比如第 100 个权重)
    # 物理索引 100
    victim_idx = 100

    # 获取原始值和 Scale
    row = victim_idx // in_features
    col = victim_idx % in_features

    # 使用 real_quantize 获取整数值
    w_tensor = radar.weight.data[row, col].view(1, 1)
    s_tensor = radar.scale[row].view(1, 1)
    w_int_before = real_quantize(w_tensor, s_tensor, wbits, device).item()

    print(f"    Target Index: {victim_idx} (Row={row}, Col={col})")
    print(f"    Int Value Before: {w_int_before}")

    # 模拟 MSB 翻转 (Bit 7)
    # 翻转逻辑: 异或 128
    w_int_after = w_int_before ^ 128

    # 计算 Float 变化量
    # 注意: real_quantize 是 round 过的，直接用 (w_int_after - w_int_before) * scale 近似
    delta_float = (w_int_after - w_int_before) * s_tensor.item()

    print(f"    Int Value After : {w_int_after} (Bit 7 flipped)")
    print(f"    Float Delta     : {delta_float:.6f}")

    # 实施攻击 (直接修改 .data)
    radar.weight.data.view(-1)[victim_idx] += delta_float

    # 执行一次 Forward (触发 RADAR)
    print("    Running Forward Pass...")
    with torch.no_grad():
        _ = radar(x)
        # 此时你应该在控制台看到 "🚨 ALERT: Attack Detected..."

    # --- 4. 测试: Recovery Verification (恢复验证) ---
    print("\n>>> Test 3: Verifying Recovery (验证恢复)")

    # 我们无法直接检查 radar.weight，因为那是 Parameter
    # 但我们可以再次运行 forward，并检查输出是否异常
    # 如果恢复成功，输出应该接近原始输出 (虽然少了一组权重)
    # 如果恢复失败，输出会受到 +delta_float 的巨大影响

    with torch.no_grad():
        y_recovered = radar(x)

    # 计算误差
    # Error vs Original (Clean)
    err_recovery = (y_recovered - y_orig).abs().mean().item()

    # Error vs Attacked (手动算一个不带防御的)
    # 创建一个临时的被攻击层
    bad_layer = nn.Linear(in_features, out_features).to(device)
    bad_layer.weight.data = radar.weight.data.clone()  # 此时已包含攻击
    bad_layer.bias.data = radar.bias.data.clone()
    y_no_defense = bad_layer(x)

    err_no_defense = (y_no_defense - y_orig).abs().mean().item()

    print(f"    MSE (No Defense): {err_no_defense:.4f} (Should be Large)")
    print(f"    MSE (RADAR)     : {err_recovery:.4f} (Should be Small)")

    if err_recovery < err_no_defense:
        print(f"✅ [PASS] RADAR reduced the error significantly!")
    else:
        print(f"❌ [FAIL] RADAR did not improve the error.")

    print("\n🎉 All Tests Completed!")


if __name__ == "__main__":
    test_radar_functionality()