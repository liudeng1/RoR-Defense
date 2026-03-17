import torch
import torch.nn as nn
from custom_logger import logger

def real_quantize(x, scale, wbits, dev):
    """
    [核心函数] 执行对称真量化 (Float -> Int)

    针对 LLM Linear 层权重:
    - x shape: [out_features, in_features]
    - scale shape:
        - Per-Channel: [out_features, 1] (必须是2D且第二维是1)
        - Per-Tensor:  [1, 1] 或 标量

    严禁传入 1D scale，防止广播错误。
    """
    # 1. 参数检查
    maxq = 2 ** (wbits - 1) - 1
    if maxq < 0:
        raise ValueError(f"wbits {wbits} is too small.")

    # 2. 严格的维度检查 (Safety Check)
    if scale.dim() == 1:
        logger.error(f"Scale dimension error! Received 1D scale with shape {scale.shape}.")
        logger.error("For per-channel quantization on weights [Out, In], scale must be [Out, 1].")
        raise ValueError("Scale cannot be 1D. Please reshape it to (N, 1) before calling real_quantize.")

    # 确保 Scale 和 Weight 维度匹配 (针对 2D 权重)
    if x.dim() == 2:
        if scale.dim() == 2:
            # 检查广播兼容性: scale.shape[0] 必须等于 x.shape[0] (Per-Channel) 或 1 (Per-Tensor)
            # scale.shape[1] 必须为 1
            if scale.shape[1] != 1:
                raise ValueError(
                    f"Scale shape {scale.shape} invalid. Second dim must be 1 for broadcasting over [Out, In].")
            if scale.shape[0] != 1 and scale.shape[0] != x.shape[0]:
                raise ValueError(f"Scale shape {scale.shape} incompatible with weight {x.shape}.")

    # 3. 数据准备
    # clone 防止原地修改破坏原始模型权重 (由调用方决定是否需要，安全起见这里 clone)
    x = x.to(dev).clone()
    scale = scale.to(dev, dtype=x.dtype)

    # 4. 原地操作执行量化 (Symmetric: x / scale -> round -> clamp)
    # 广播机制: [Out, In] / [Out, 1] -> 每一行除以对应的 scale
    x.div_(scale)
    x.round_()
    x.clamp_(-maxq, maxq)

    # 5. 转整数
    W_q = x.to(torch.int16)

    del x
    return W_q


def fake_quantize(x, scale, wbits, dev):
    """
    [辅助] 伪量化流程: Float -> Int -> Float
    """
    # 1. 真量化 (Float -> Int)
    w_int = real_quantize(x, scale, wbits, dev)

    # 2. 反量化 (Int -> Float)
    # Scale 已经是 [N, 1] 或 [1, 1]，直接乘即可广播
    w_dequant = w_int.to(x.dtype) * scale.to(dev, dtype=x.dtype)

    return w_dequant


class Quantizer(nn.Module):
    """
    统一量化器: 负责计算并维护 Scale
    """

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        # 初始化 buffer，注意 shape 只是占位，后面 configure 会重置
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, wbits, perchannel=True, sym=True, mse=False):
        self.wbits = wbits
        self.perchannel = perchannel
        self.sym = True  # 本模块强制对称量化
        self.mse = mse
        self.maxq = 2 ** (self.wbits - 1) - 1

    def find_params(self, x, weight=True):
        """
        计算 Scale。
        针对 Linear Weight [Out, In]：
        - Per-Channel: Scale shape 应为 [Out, 1]
        - Per-Tensor:  Scale shape 应为 [1, 1]
        """
        if not weight:
            # 如果不是权重(比如激活)，这里暂不处理，或者按照 Per-Tensor 处理
            pass

        # x shape: [Out, In]
        if self.perchannel:
            # 1. 扁平化 dim 1 (In_features)，保留 dim 0 (Out_features)
            # x_flat: [Out, In]
            # 2. 对称量化：取绝对值的最大值
            # xmax: [Out] (1D Tensor)
            xmax = x.abs().max(dim=1).values

            # [关键] 必须在这里 reshape 为 [Out, 1]，满足 real_quantize 的严格检查
            xmax = xmax.unsqueeze(1)
        else:
            # Per-Tensor
            xmax = x.abs().max()
            # 变为 [1, 1] 以保持一致性
            xmax = xmax.reshape(1, 1)

        # 防止除以 0
        xmax = torch.clamp(xmax, min=1e-5)

        # 计算 Scale
        self.scale = xmax / self.maxq

        # Zero 恒为 0 (对称量化)
        self.zero = torch.zeros_like(self.scale)

    def free(self):
        self.scale = None
        self.zero = None
        torch.cuda.empty_cache()

    def quantize(self, x):
        return fake_quantize(x, self.scale, self.wbits, x.device)

    def forward(self, x):
        if self.wbits >= 16:
            return x
        return self.quantize(x)