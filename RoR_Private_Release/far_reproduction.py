import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from collections import defaultdict
import time
import sys
import copy

# ------------------------------------------------------------------------------
# 日志配置 (兼顾独立运行和被调用)
# ------------------------------------------------------------------------------
try:
    from custom_logger import logger
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)


# ==============================================================================
# 1. FaRLinear 类 (核心组件)
#    - 包含 [Sparse Optimized] (生产用) 和 [Original Logic] (验证用)
#    - 已集成 FP32 强制转换，解决 BF16 PPL 崩溃问题
# ==============================================================================
class FaRLinear(nn.Linear):
    def __init__(self, original_linear: nn.Linear):
        super().__init__(
            original_linear.in_features,
            original_linear.out_features,
            bias=original_linear.bias is not None
        )
        # 共享权重内存 (In-place modification)
        self.weight.data = original_linear.weight.data
        if original_linear.bias is not None:
            self.bias.data = original_linear.bias.data

        # 注册 Buffer 以便保存/加载 (不参与梯度更新)
        self.register_buffer('far_rows', torch.tensor([], dtype=torch.long))
        self.register_buffer('far_targets', torch.tensor([], dtype=torch.long))
        self.register_buffer('far_deads', torch.tensor([], dtype=torch.long))
        self.register_buffer('sparse_indices', torch.tensor([], dtype=torch.long))

        self.div_factor = 1
        self.is_config_ready = False
        self.mode = 'sparse'  # 默认为加速模式 ('sparse' 或 'origin')

    def load_config_and_modify_weights(self, rows, targets, deads):
        """
        加载 FaR 规则，执行物理权重修改 (W_dead = W_target)，并预计算稀疏索引。
        """
        if len(rows) == 0: return
        device = self.weight.device

        # 1. 存入 Buffer (兼容 List/Tensor)
        if not isinstance(rows, torch.Tensor): rows = torch.tensor(rows, dtype=torch.long, device=device)
        if not isinstance(targets, torch.Tensor): targets = torch.tensor(targets, dtype=torch.long, device=device)
        if not isinstance(deads, torch.Tensor): deads = torch.tensor(deads, dtype=torch.long, device=device)

        self.far_rows = rows.to(device)
        self.far_targets = targets.to(device)
        self.far_deads = deads.to(device)
        self.div_factor = self.far_deads.shape[1] + 1
        self.is_config_ready = True

        # 2. [物理修改] W_dead = W_target (关键步骤)
        with torch.no_grad():
            target_vals = self.weight.data[self.far_rows, self.far_targets]
            for i in range(self.far_deads.shape[1]):
                col_indices = self.far_deads[:, i]
                self.weight.data[self.far_rows, col_indices] = target_vals

        # 3. [加速准备] 预构建稀疏矩阵索引
        idx_rows_t = self.far_rows
        idx_cols_t = self.far_targets
        indices_t = torch.stack([idx_rows_t, idx_cols_t], dim=0)

        num_deads = self.far_deads.shape[1]
        idx_rows_d = self.far_rows.unsqueeze(1).expand(-1, num_deads).reshape(-1)
        idx_cols_d = self.far_deads.reshape(-1)
        indices_d = torch.stack([idx_rows_d, idx_cols_d], dim=0)

        # 合并 Target 和 Dead 的索引
        self.sparse_indices = torch.cat([indices_t, indices_d], dim=1).to(device)

    def forward(self, input):
        # 如果没有配置规则，回退到普通 Linear
        if not self.is_config_ready or self.far_rows.numel() == 0:
            return F.linear(input, self.weight, self.bias)

        # 根据模式选择路径
        if self.mode == 'sparse':
            return self.forward_sparse(input)
        elif self.mode == 'origin':
            return self.forward_origin(input)
        else:
            return F.linear(input, self.weight, self.bias)

    def forward_sparse(self, x):
        """
        [Fix PPL Explosion] 强制全程使用 FP32 进行系数计算和矩阵乘法
        """
        # A. 标准 GEMM (保持原精度，通常是 BF16，这是基准)
        y = F.linear(x, self.weight, self.bias)

        # ==========================================================
        # 核心修复 1: 提取权重并立即转为 FP32 (关键!)
        # ==========================================================
        # 如果这里不转，(Sum/Div - W) 会在 BF16 下产生非零噪声
        w_t = self.weight[self.far_rows, self.far_targets].to(torch.float32)
        w_d = self.weight[self.far_rows.unsqueeze(1), self.far_deads].squeeze(1).to(torch.float32)

        # 2. 计算 FaR 系数 (在高精度 FP32 下)
        # 此时 (Div*W / Div) - W 应该精确等于 0.0000000
        sum_w = w_t + w_d.sum(dim=1)
        val_t = (sum_w / self.div_factor) - w_t
        val_d = -w_d.reshape(-1)

        # 合并 Values (FP32)
        values = torch.cat([val_t, val_d])

        # 3. 构造稀疏矩阵
        out_dim, in_dim = self.weight.shape
        S_sparse = torch.sparse_coo_tensor(
            self.sparse_indices,
            values,
            size=(out_dim, in_dim)
        ).coalesce()

        # ==========================================================
        # 核心修复 2: 输入转 FP32 (关键!)
        # ==========================================================
        # 必须用 FP32 输入与 FP32 稀疏矩阵相乘
        x_flat = x.view(-1, x.shape[-1]).to(torch.float32)

        # Delta = S @ x.T
        delta_T = torch.sparse.mm(S_sparse, x_flat.t())

        # 5. 结果还原
        # 先转置，再转回 BF16 (此时 Delta 应该极其接近 0，转回 BF16 也是安全的)
        delta = delta_T.t().to(x.dtype).view_as(y)

        # E. Apply Correction
        y = y + delta

        return y

    def forward_sparse_64(self, x):
        # 1. 既然 BF16 不行，FP32 也不稳，我们直接用 FP64 (Double) 验明正身
        #    虽然慢一点，但如果是精度问题，这一定能修好。

        # A. 基准输出
        y = F.linear(x, self.weight, self.bias)

        # B. 强制转 Double 进行精密计算
        w_t = self.weight[self.far_rows, self.far_targets].double()
        w_d = self.weight[self.far_rows.unsqueeze(1), self.far_deads].squeeze(1).double()

        # 精密抵消
        sum_w = w_t + w_d.sum(dim=1)
        val_t = (sum_w / self.div_factor) - w_t
        val_d = -w_d.reshape(-1)

        values = torch.cat([val_t, val_d])

        # 构造 Double 稀疏矩阵
        out_dim, in_dim = self.weight.shape
        S_sparse = torch.sparse_coo_tensor(
            self.sparse_indices,
            values,
            size=(out_dim, in_dim)
        ).coalesce()

        # 输入也转 Double
        x_flat = x.view(-1, x.shape[-1]).double()

        # 执行高精乘法
        delta_T = torch.sparse.mm(S_sparse, x_flat.t())

        # 还原
        delta = delta_T.t().to(x.dtype).view_as(y)
        y = y + delta

        return y

    def forward_origin(self, x):
        """
        [Reference Path] 原始论文逻辑复刻 (Slow Loop)
        用于验证 forward_sparse 的数学正确性。
        """
        y = F.linear(x, self.weight, self.bias)
        # 强制转 FP32 以便与 sparse 模式进行公平数值对比
        x_flat = x.view(-1, x.shape[-1]).float()
        y_flat = y.view(-1, y.shape[-1]).float()

        rows = self.far_rows
        tgts = self.far_targets
        deads = self.far_deads
        div = self.div_factor

        w_t_all = self.weight[rows, tgts].float()
        w_d_all = self.weight[rows.unsqueeze(1), deads].float()

        # 逐规则修正
        for i in range(len(rows)):
            r, t, ds = rows[i], tgts[i], deads[i]
            w_t, w_ds = w_t_all[i], w_d_all[i]
            x_t = x_flat[:, t]
            x_ds = x_flat[:, ds]

            # Old Contribution
            term_old = x_t * w_t + (x_ds * w_ds).sum(dim=1)
            # New Contribution (Input Averaging)
            term_new = (x_t / div) * (w_t + w_ds.sum())

            y_flat[:, r] += (term_new - term_old)

        return y_flat.view_as(y).to(x.dtype)


# ==============================================================================
# 2. 配置生成函数 (Stage 1) - 全局梯度策略
# ==============================================================================
def compute_and_save_far_config(model, dataloader, dev, save_path,
                                far_ratio=0.01, div_factor=6, fraction_size=4,
                                nsamples=16, chunk_size=4):
    logger.info(f"[FaR-Gen] 开始生成配置 (Global Strategy | Ratio={far_ratio}, Div={div_factor})...")
    far_config_dict = {}

    # 适配模型结构
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        layers = model.model.decoder.layers
    else:
        raise ValueError("无法识别模型结构")

    num_layers = len(layers)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 分块处理层 (节省显存)
    for start_idx in range(0, num_layers, chunk_size):
        end_idx = min(start_idx + chunk_size, num_layers)
        logger.info(f"[FaR-Gen] Processing Chunk {start_idx}-{end_idx}...")

        model.eval()
        model.zero_grad()
        model.requires_grad_(False)

        active_modules = {}
        for i in range(start_idx, end_idx):
            layer = layers[i]
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.requires_grad_(True)
                    full_key = f"{i}.{name}"
                    active_modules[full_key] = module

        if not active_modules: continue

        # 1. 计算累计梯度
        grads_accum = {k: torch.zeros_like(m.weight, device=dev) for k, m in active_modules.items()}
        count = 0
        for batch in dataloader:
            if count >= nsamples: break
            inp = batch[0] if isinstance(batch, (tuple, list)) else batch
            inp = inp.to(dev)
            if inp.dim() == 2: inp = inp.unsqueeze(0)
            try:
                out = model(inp, labels=inp)
                out.loss.backward()
            except RuntimeError:
                model.zero_grad();
                torch.cuda.empty_cache();
                continue
            for k, m in active_modules.items():
                if m.weight.grad is not None:
                    grads_accum[k] += m.weight.grad.detach().abs()
            model.zero_grad()
            count += 1

        # 2. 全局筛选与匹配逻辑
        for key, module in active_modules.items():
            grad = grads_accum[key]
            rows, cols = grad.shape

            total_params = rows * cols
            target_limit = int(total_params * far_ratio)
            max_targets_per_row = cols // (div_factor + 1)  # 行容量限制

            if max_targets_per_row == 0:
                continue

            # A. 全局排序
            flat_grad = grad.view(-1)
            _, sorted_indices = torch.sort(flat_grad, descending=True)

            # B. 贪心分配
            sorted_indices_cpu = sorted_indices.cpu().numpy()
            row_counts = np.zeros(rows, dtype=np.int32)
            selected_targets_map = defaultdict(list)
            selected_count = 0

            for idx in sorted_indices_cpu:
                if selected_count >= target_limit: break
                r = idx // cols
                c = idx % cols
                if row_counts[r] < max_targets_per_row:
                    row_counts[r] += 1
                    selected_targets_map[r].append(c)
                    selected_count += 1

            # C. 局部配对 (Find Deads)
            l_rows, l_targets, l_deads = [], [], []
            for r, targets in selected_targets_map.items():
                if not targets: continue
                num_deads_needed = len(targets) * div_factor
                row_grad = grad[r, :]

                # 升序排序找最小梯度作为 Dead
                _, row_indices_asc = torch.sort(row_grad, descending=False)
                row_indices_asc = row_indices_asc.tolist()

                chosen_deads = []
                targets_set = set(targets)
                for idx in row_indices_asc:
                    if idx not in targets_set: chosen_deads.append(idx)
                    if len(chosen_deads) >= num_deads_needed: break

                # 配对
                if len(chosen_deads) >= num_deads_needed:
                    for i, t in enumerate(targets):
                        d_indices = chosen_deads[i * div_factor: (i + 1) * div_factor]
                        l_rows.append(r);
                        l_targets.append(t);
                        l_deads.append(d_indices)

            # 保存为 Tensor 以压缩体积
            if l_rows:
                far_config_dict[key] = {
                    'rows': torch.tensor(l_rows, dtype=torch.long),
                    'targets': torch.tensor(l_targets, dtype=torch.long),
                    'deads': torch.tensor(l_deads, dtype=torch.long)
                }

            del grads_accum[key]
            module.weight.grad = None

        del grads_accum
        torch.cuda.empty_cache()

    torch.save(far_config_dict, save_path)
    logger.info(f"[FaR-Gen] 配置已保存至: {save_path}")
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return far_config_dict


# ==============================================================================
# 3. 配置加载函数 (Stage 2) - 真实模型替换
# ==============================================================================
def load_and_apply_far_config(model, config_path, dev):
    logger.info(f"[FaR-Load] 正在加载配置: {config_path}")
    if not os.path.exists(config_path): return model

    # [Fix] 兼容 PyTorch 2.6+
    try:
        far_config = torch.load(config_path, map_location="cpu", weights_only=False)
    except TypeError:
        far_config = torch.load(config_path, map_location="cpu")

    # 适配模型结构
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers  # Llama
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        layers = model.model.decoder.layers  # OPT
    else:
        logger.warning("无法识别模型结构，未应用 FaR 配置")
        return model

    applied_count = 0
    for i, layer in enumerate(layers):
        targets_in_layer = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                key = f"{i}.{name}"
                if key in far_config: targets_in_layer[name] = (module, key)

        for name, (module, key) in targets_in_layer.items():
            cfg = far_config[key]
            parts = name.split('.')
            parent = layer
            for part in parts[:-1]: parent = getattr(parent, part)
            child_name = parts[-1]

            if not isinstance(module, FaRLinear):
                new_layer = FaRLinear(module).to(module.weight.device)
                setattr(parent, child_name, new_layer)
                module = new_layer

            module.load_config_and_modify_weights(cfg['rows'], cfg['targets'], cfg['deads'])

            applied_count += 1
            if applied_count <= 3 or applied_count % 20 == 0:
                count = cfg['rows'].numel() if isinstance(cfg['rows'], torch.Tensor) else len(cfg['rows'])
                logger.info(f"✅ [FaR-Apply] {key}: 应用 {count} 条规则")

    return model


# ==============================================================================
# [ViT 适配版] 核心流程
# 专门针对 Vision Transformer (HuggingFace格式) 的结构和输入进行适配
# ==============================================================================

def compute_and_save_far_config_vit(model, dataloader, dev, save_path,
                                    far_ratio=0.01, div_factor=6, fraction_size=4,
                                    nsamples=16, chunk_size=4):
    """
    [ViT版] 计算梯度 -> 生成重连规则 -> 保存配置
    区别：
    1. 识别 model.vit.encoder.layer 结构
    2. 处理 pixel_values 输入
    """
    logger.info(f"[FaR-Gen-ViT] 开始生成配置 (Ratio={far_ratio}, Div={div_factor})...")

    far_config_dict = {}

    # 1. 识别 ViT 层结构 (HuggingFace 标准结构)
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        layers = model.vit.encoder.layer
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):  # 某些变体
        layers = model.encoder.layer
    else:
        # 尝试直接查找 ModuleList (通用兜底)
        logger.warning("[FaR-Gen-ViT] 未识别标准 ViT 结构，尝试搜索 ModuleList...")
        found = False
        for name, m in model.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) > 0:
                layers = m
                found = True
                break
        if not found:
            raise ValueError("无法识别 ViT 模型层结构")

    num_layers = len(layers)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # --- 分块处理层 ---
    for start_idx in range(0, num_layers, chunk_size):
        end_idx = min(start_idx + chunk_size, num_layers)
        logger.info(f"[FaR-Gen-ViT] Processing Chunk {start_idx}-{end_idx}...")

        # A. 激活当前块
        model.eval()
        model.zero_grad()
        model.requires_grad_(False)

        active_modules = {}
        for i in range(start_idx, end_idx):
            layer = layers[i]
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.requires_grad_(True)
                    # Key 格式: "0.intermediate.dense"
                    full_key = f"{i}.{name}"
                    active_modules[full_key] = module

        if not active_modules: continue

        # B. 计算梯度
        grads_accum = {k: torch.zeros_like(m.weight, device=dev) for k, m in active_modules.items()}

        count = 0
        for batch in dataloader:
            if count >= nsamples: break

            # [ViT适配] 解包 ImageNet/CIFAR 数据
            # batch 通常是 (pixel_values, labels)
            if isinstance(batch, (tuple, list)):
                pixel_values = batch[0]
                labels = batch[1]
            else:
                # 可能是 dict (HF Dataset格式)
                pixel_values = batch['pixel_values'] if 'pixel_values' in batch else batch
                labels = batch['labels'] if 'labels' in batch else None

            pixel_values = pixel_values.to(dev)
            if labels is not None:
                labels = labels.to(dev)

            try:
                # [ViT适配] 传入 pixel_values 和 labels
                out = model(pixel_values, labels=labels)
                loss = out.loss
                if loss is None:
                    # 如果模型没返回 loss (例如 classifier 未初始化)，尝试手动计算
                    if hasattr(out, 'logits') and labels is not None:
                        loss = F.cross_entropy(out.logits, labels)
                    else:
                        raise ValueError("Model output loss is None")

                loss.backward()
            except RuntimeError as e:
                model.zero_grad()
                torch.cuda.empty_cache()
                continue

            for k, m in active_modules.items():
                if m.weight.grad is not None:
                    grads_accum[k] += m.weight.grad.detach().abs()

            model.zero_grad()
            count += 1

        # C. 生成规则 (同 LLM)
        for key, module in active_modules.items():
            grad = grads_accum[key]
            rows, cols = grad.shape

            row_sums = grad.sum(dim=1)
            top_rows = torch.argsort(row_sums, descending=True)

            limit = int(rows * cols * far_ratio)
            rewired = 0

            l_rows, l_targets, l_deads = [], [], []

            for r_idx in top_rows:
                if rewired >= limit: break
                row = r_idx.item()
                row_grad = grad[row, :]
                if row_grad.max() < 1e-9: continue

                r_vals, r_indices = torch.sort(row_grad)
                target = r_indices[-1].item()
                deads = [r_indices[k].item() for k in range(div_factor)]

                l_rows.append(row)
                l_targets.append(target)
                l_deads.append(deads)
                rewired += 1

            if l_rows:
                far_config_dict[key] = {
                    'rows': l_rows,
                    'targets': l_targets,
                    'deads': l_deads
                }

            del grads_accum[key]
            module.weight.grad = None

        del grads_accum
        torch.cuda.empty_cache()

    # 保存
    torch.save(far_config_dict, save_path)
    logger.info(f"[FaR-Gen-ViT] 配置已保存至: {save_path}")

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    return far_config_dict


def load_and_apply_far_config_vit(model, config_path, dev):
    """
    [ViT版] 加载配置 -> 替换 Linear -> 应用
    """
    logger.info(f"[FaR-Load-ViT] 正在加载配置: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"FaR 配置文件未找到: {config_path}")

    far_config = torch.load(config_path, map_location="cpu")

    # 打印 Debug 信息
    all_keys = list(far_config.keys())
    if all_keys:
        logger.info(f"   >>> [Debug] Config Keys Sample: {all_keys[:3]}")

    # 1. 识别 ViT 层结构
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        layers = model.vit.encoder.layer
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layers = model.encoder.layer
    else:
        # Fallback
        for name, m in model.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) > 0:
                layers = m
                break

    applied_count = 0

    # 2. 遍历并替换
    for i, layer in enumerate(layers):
        targets_in_layer = {}

        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                # Key: "0.intermediate.dense" (与 Gen 阶段保持一致)
                key = f"{i}.{name}"
                if key in far_config:
                    targets_in_layer[name] = (module, key)

        for name, (module, key) in targets_in_layer.items():
            cfg = far_config[key]
            # 复用原有的 _replace_module_by_name (它是通用的，只要 root_layer 和 name 对就行)
            # 注意：_replace_module_by_name 在 far_reproduction.py 里面定义了
            # 如果它不在作用域内，你需要确保它被导入或者在这里可见
            success, new_module = _replace_module_by_name(layer, name, module, cfg)

            if success:
                applied_count += 1
                if applied_count <= 5 or applied_count % 10 == 0:
                    logger.info(f"✅ [FaR-Apply-ViT] 成功应用: {key}")

    logger.info(f"[FaR-Report-ViT] 共应用了 {applied_count} 个层。")
    return model


def _replace_module_by_name(root_layer, relative_name, old_module, config_item):
    """
    [通用辅助函数] 根据 relative_name (如 'intermediate.dense')
    在 root_layer 中找到父对象并替换子对象为 FaRLinear
    """
    try:
        # 分割路径，例如 "attention.attention.query" -> ["attention", "attention", "query"]
        parts = relative_name.split('.')
        parent = root_layer

        # 逐级深入到倒数第二级（找到父模块）
        for part in parts[:-1]:
            parent = getattr(parent, part)

        child_name = parts[-1]

        # 1. 检查并替换结构
        if not isinstance(old_module, FaRLinear):
            # 创建新层
            new_layer = FaRLinear(old_module).to(old_module.weight.device)
            # 物理替换指针 (这是最关键的一步)
            setattr(parent, child_name, new_layer)
            target_module = new_layer
        else:
            # 已经是 FaRLinear 了 (可能之前处理过)
            target_module = old_module

        # 2. 注入配置 & 物理修改权重
        # 这一步会执行 w[dead] = w[target]
        target_module.load_config_and_modify_weights(
            config_item['rows'],
            config_item['targets'],
            config_item['deads']
        )

        return True, target_module

    except Exception as e:
        # 确保 logger 可用，如果报错就 print
        if 'logger' in globals():
            logger.error(f"替换层 {relative_name} 异常: {e}")
        else:
            print(f"Error replacing layer {relative_name}: {e}")
        return False, None


def generate_far_whitebox_attack(model, target_log, quantizers=None):
    """
    [White-box Attack for FaR] 针对 FaR 生成全量覆盖攻击 (Revert Dilution)

    原理:
    FaR 通过将误差除以 N (N=Div+1) 来稀释单点攻击。
    白盒攻击者读取 FaR 配置，同时翻转 Target 和所有的 Deads (共 N 个点)。
    Result: 误差累加为 N * (Delta/N) = Delta，稀释完全失效，伤害 100% 还原。

    Args:
        model: 已加载 FaR 层的模型
        target_log: 原始攻击目标 [LayerIdx, LayerName, [Row, Col], Bit, Score]
        quantizers: (占位符，保持接口一致)

    Returns:
        list: 包含 [Target, Dead1, Dead2, ... DeadK] 的攻击列表。
              如果 Target 不在 FaR 保护名单中，返回原列表 [Target]。
    """
    layer_idx, layer_name, coords, target_bit, _ = target_log
    row, col = coords

    # 1. 定位目标层 (FaRLinear)
    # 适配 Llama / OPT 结构
    if hasattr(model, 'model'):
        layers = model.model.layers if hasattr(model.model, 'layers') else model.model.decoder.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    elif hasattr(model, 'decoder'):
        layers = model.decoder.layers
    else:
        # 兜底
        layers = []
        for m in model.modules():
            if isinstance(m, nn.ModuleList):
                layers = m
                break

    if layer_idx >= len(layers):
        logger.warning(f"Layer Index {layer_idx} out of range.")
        return [target_log]

    block = layers[layer_idx]

    # 解析层名称 (去除前缀)
    # e.g. "1_self_attn.v_proj" -> "self_attn.v_proj"
    clean_sub_name = layer_name
    if layer_name.startswith(f"{layer_idx}_"):
        clean_sub_name = layer_name.split(f"{layer_idx}_")[1]

    curr = block
    try:
        for p in clean_sub_name.split('.'):
            curr = getattr(curr, p)
    except AttributeError:
        logger.warning(f"无法定位层 {layer_name}")
        return [target_log]

    target_module = curr

    # 2. 检查是否为 FaRLinear
    if not isinstance(target_module, FaRLinear):
        # 可能没被替换，或者不是 FaR 层
        # logger.info(f"Layer {layer_name} is not FaRLinear. Skipping white-box expansion.")
        return [target_log]

    if not target_module.is_config_ready:
        return [target_log]

    # 3. 在 FaR 配置中查找 Target
    # far_rows: [M], far_targets: [M], far_deads: [M, K]
    # 我们需要找到 idx，使得 far_rows[idx] == row 且 far_targets[idx] == col
    device = target_module.far_rows.device

    # 转换为 Tensor 进行搜索
    t_row = torch.tensor(row, device=device)
    t_col = torch.tensor(col, device=device)

    # 查找匹配的规则索引
    # mask = (rows == row) & (targets == col)
    mask = (target_module.far_rows == t_row) & (target_module.far_targets == t_col)
    indices = torch.nonzero(mask, as_tuple=True)[0]

    if len(indices) == 0:
        logger.info(f"Target {coords} NOT found in FaR rules. (Standard attack applied)")
        return [target_log]

    # 找到规则了！
    rule_idx = indices[0].item()

    # 4. 获取所有的 Deads
    deads = target_module.far_deads[rule_idx]  # [K]

    logger.info(f"⚔️ [FaR-Whitebox] Target {coords} hit Rule #{rule_idx}!")
    logger.info(f"    -> Found {len(deads)} Deads to attack: {deads.tolist()}")

    # 5. 构造全量攻击列表
    # Start with original target
    expanded_attack_logs = [target_log]

    # Add all deads
    # 格式: [LayerIdx, LayerName, [Row, DeadCol], Bit, Score]
    # 注意: Bit 必须相同，才能同向叠加误差
    for dead_col in deads.tolist():
        dead_log = [layer_idx, layer_name, [row, dead_col], target_bit, 0]
        expanded_attack_logs.append(dead_log)

    return expanded_attack_logs



# ==============================================================================
# 4. 独立运行时的自测模块 (Test Suite)
#    执行命令: python far_reproduction.py
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 80}\nFaRLinear 独立验证与性能基准测试\n{'=' * 80}")

    # 1. 准备测试环境
    Dim = 4096  # 真实尺寸
    Div = 6
    Ratio = 0.05

    # 使用 FP32 基准 (排除 BF16 精度干扰验证逻辑)
    real_layer = nn.Linear(Dim, Dim).to(dev).float()
    inp = torch.randn(1, 128, Dim, device=dev).float()
    far_layer = FaRLinear(real_layer).to(dev)

    # 2. 生成规则
    print(f"[Init] 生成规则 (Dim={Dim}, Ratio={Ratio}, Div={Div})...")
    num_rules = int(Dim * Ratio)
    rows, targets, deads = [], [], []
    for r in range(Dim):  # 简单填充
        if len(rows) >= num_rules: break
        perm = torch.randperm(Dim)[:Div + 1].tolist()
        rows.append(r)
        targets.append(perm[0])
        deads.append(perm[1:])

    far_layer.load_config_and_modify_weights(rows, targets, deads)

    # --- Case 1: 严格一致性 ---
    print("\n[Case 1] 严格一致性 (Sparse vs Original)")
    far_layer.mode = 'origin'
    y_origin = far_layer(inp)
    far_layer.mode = 'sparse'
    y_sparse = far_layer(inp)

    diff = (y_origin - y_sparse).abs().max().item()
    print(f"Max Diff: {diff:.9e}")
    if diff < 1e-5:
        print("✅ PASS: 结果严格一致")
    else:
        print("❌ FAIL: 结果不一致")

    # --- Case 2: 防御逻辑 ---
    print("\n[Case 2] 物理防御逻辑验证")
    # 模拟攻击：修改 Target
    inject_err = 100.0
    r0, t0 = rows[0], targets[0]

    # 对照组：Standard Linear
    w_dirty = far_layer.weight.data.clone()
    w_dirty[r0, t0] += inject_err
    y_std_dirty = F.linear(inp, w_dirty, far_layer.bias)
    y_std_clean = F.linear(inp, far_layer.weight.data, far_layer.bias)  # 注意：此时weight已被修改(copy dead)，但对于std来说这就是基准
    err_std = (y_std_dirty - y_std_clean).abs().max().item()

    # 实验组：FaR Linear
    far_layer.weight.data[r0, t0] += inject_err  # 物理攻击
    y_far_dirty = far_layer(inp)
    far_layer.weight.data[r0, t0] -= inject_err  # 恢复
    y_far_clean = far_layer(inp)
    err_far = (y_far_dirty - y_far_clean).abs().max().item()

    ratio = err_std / (err_far + 1e-9)
    print(f"原始误差: {err_std:.4f}, FaR误差: {err_far:.4f}")
    print(f"稀释倍数: {ratio:.4f} (理论值: {Div+1:.4f})")
    if abs(ratio - Div) < 1.0:
        print("✅ PASS: 防御生效")
    else:
        print("⚠️ WARNING: 防御偏差较大")

    print("\n测试完成。此脚本可作为库文件直接被 opt.py 引用。")