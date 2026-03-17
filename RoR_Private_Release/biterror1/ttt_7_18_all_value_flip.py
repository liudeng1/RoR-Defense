import torch
import math


def int16_to_binary_str(x: int, bit_w: int) -> str:
    """
    将int16整数转换为指定位宽的无符号二进制字符串
    :param x: int16数值（有符号）
    :param bit_w: 目标位宽（如9）
    :return: 二进制字符串（补零到指定位宽）
    """
    mask = (1 << bit_w) - 1
    masked_x = x & mask
    unsigned_x = masked_x if masked_x >= 0 else (1 << bit_w) + masked_x
    return format(unsigned_x, f'0{bit_w}b')


def tensor_to_binary(data: torch.Tensor, bit_w: int) -> None:
    """
    打印张量中每个元素的指定位宽二进制表示
    :param data: 输入张量（int16类型）
    :param bit_w: 目标位宽（如9）
    """
    if data.is_cuda:
        data = data.cpu()
    for x in data.view(-1):
        x_int = x.item()
        binary_str = int16_to_binary_str(x_int, bit_w)
        print(f"十进制: {x_int:6d} → 二进制: {binary_str}")


def count_ones_in_binary(n):
    """
    统计二进制数中的'1'的个数，输入是uint类型的才有效
    """
    return bin(n).count('1')


def int_to_8bit(num):
    return format(num & 0xFF, '08b')


def count_ones_in_uint8_tensor(tensor):
    """
    求一个uint8类型张量，表示为8bit字符后，“1”所占的比例
    """
    if tensor.dtype != torch.uint8:
        raise TypeError("检查“1”，输入张量必须是 uint8 类型!")
    bit_counts_torch = tensor.clone().apply_(lambda x: bin(x).count('1'))
    total_ones = bit_counts_torch.sum().item()
    total_bits = tensor.numel() * 8
    ratio_of_ones = total_ones / total_bits
    print(f"1占所有bit的比例为: {ratio_of_ones:.4f}")
    return ratio_of_ones


def count_ones_in_nbit_tensor(tensor, bit_w=None):
    """
    统计张量中指定有效位宽（bit_w）内二进制'1'的比例
    """
    mask = (1 << bit_w) - 1
    masked_tensor = tensor & mask
    bit_counts = torch.zeros_like(masked_tensor)
    for i in range(bit_w):
        bit_counts += (masked_tensor >> i) & 1
    total_ones = bit_counts.sum().item()
    total_bits = tensor.numel() * bit_w
    ratio_of_ones = total_ones / total_bits
    print(f"有效{bit_w}位中，1占所有bit的比例为: {ratio_of_ones:.4f}")
    return ratio_of_ones


def int8_to_uint8(int8_tensor):
    return int8_tensor.to(torch.uint8)


def uint8_to_int8(uint8_tensor):
    return uint8_tensor.to(torch.int8)


class Prob:
    def __init__(self, bit_error_rate, bit_w, mask_bits=None):
        self.bit_error_rate = bit_error_rate
        self.bit_w = bit_w
        self.mask_bits = mask_bits if mask_bits is not None else list(range(bit_w))

    def error_nbit_full(self):
        """
        生成仅允许 mask_bits 指定位翻转的错误模式及概率
        """
        k = len(self.mask_bits)
        total_patterns = 2 ** k
        p = self.bit_error_rate
        value_list = []
        prob_list = []
        for i in range(total_patterns):
            bit_errors = 0
            for j in range(k):
                if (i >> j) & 1:
                    bit_errors |= (1 << self.mask_bits[j])
            value_list.append(bit_errors)
            ones = count_ones_in_binary(i)
            prob = (p ** ones) * ((1 - p) ** (k - ones))
            prob_list.append(prob)
        return value_list, prob_list


def Mask_bit_flip_raw(value_list, prob_list, data, weight_index_range=None):
    """
    根据值列表以及概率列表，生成bit翻转张量，形状为data.shape
    支持权重索引范围限制
    """
    device = data.device
    values = torch.tensor(value_list, dtype=torch.int16, device=device)
    probabilities = torch.tensor(prob_list, device=device)
    tensor_shape = data.shape
    num_elements = data.numel()
    indices = torch.multinomial(probabilities, num_elements, replacement=True)
    print(f"indices.shape:{indices.shape}, indices:{indices}")
    tensor = values[indices].reshape(tensor_shape)
    print(f"tensor.shape:{tensor.shape}, tensor:{tensor}")

    # 如果指定了weight_index_range，生成掩码限制翻转范围
    if weight_index_range is not None:
        mask_flip_range = torch.zeros(tensor_shape, dtype=torch.int16, device=device)
        start_idx, end_idx = weight_index_range
        rows, cols = tensor_shape
        for idx in range(start_idx, min(end_idx, num_elements)):
            row = idx // cols
            col = idx % cols
            if row < rows and col < cols:
                mask_flip_range[row, col] = tensor[row, col]
        tensor = mask_flip_range
    return tensor


def Mask_bit_flip_0(value_list, prob_list, data, weight_index_range=None):
    """
    根据值列表以及概率列表，生成bit翻转张量，形状为data.shape
    支持权重索引范围限制
    """
    device = data.device
    values = torch.tensor(value_list, dtype=torch.int16, device=device)
    probabilities = torch.tensor(prob_list, device=device)
    tensor_shape = data.shape
    num_elements = data.numel()
    indices = torch.multinomial(probabilities, num_elements, replacement=True)
    # print(f"indices.shape:{indices.shape}, indices:{indices}")


    # 施加 weight_index_range 限制
    if weight_index_range is not None:
        weight_start_idx, weight_end_idx, bit_start_idx, bit_end_idx = weight_index_range
        # 验证权重索引范围
        # print(f"data.shape:{data.shape}")
        # print(f"num_elements:{num_elements}")
        if weight_start_idx < 0 or weight_end_idx > num_elements or weight_start_idx > weight_end_idx:
            raise ValueError(f"Invalid weight_index_range: {weight_index_range}")

        # 将范围外的 indices 置为 0
        indices[:weight_start_idx] = 0
        indices[weight_end_idx:] = 0

        # 如果 bit_start_idx 和 bit_end_idx 不为 None，则进行比特级修改
        if bit_start_idx is not None and bit_end_idx is not None:
            if bit_start_idx < 0 or bit_end_idx > 16 or bit_start_idx > bit_end_idx:
                raise ValueError(f"Invalid bit range: ({bit_start_idx}, {bit_end_idx})")
            # 创建掩码，例如 bit_start_idx=0, bit_end_idx=4 时，掩码为 00001111 (十进制的 15)
            mask = (1 << bit_end_idx) - (1 << bit_start_idx)
            # 对指定权重范围内的 indices 进行按位与操作
            indices[weight_start_idx:weight_end_idx] = indices[weight_start_idx:weight_end_idx] & mask


    tensor = values[indices].reshape(tensor_shape)
    # print(f"tensor.shape:{tensor.shape}, tensor:{tensor}")


    return tensor


def Mask_bit_flip(value_list, prob_list, data, weight_index_range=None):
    """
    根据值列表以及概率列表，生成bit翻转张量，形状为data.shape
    支持权重索引范围限制
    """
    device = data.device
    values = torch.tensor(value_list, dtype=torch.int16, device=device)
    probabilities = torch.tensor(prob_list, device=device)
    tensor_shape = data.shape
    num_elements = data.numel()
    indices = torch.multinomial(probabilities, num_elements, replacement=True)
    # print(f"indices.shape:{indices.shape}, indices:{indices}")


    # 施加 weight_index_range 限制
    if weight_index_range is not None:
        weight_start_idx, weight_end_idx, bit_start_idx, bit_end_idx = weight_index_range
        # 验证权重索引范围
        # print(f"data.shape:{data.shape}")
        # print(f"num_elements:{num_elements}")
        if weight_start_idx < 0 or weight_end_idx > num_elements or weight_start_idx > weight_end_idx:
            raise ValueError(f"Invalid weight_index_range: {weight_index_range}")

        if bit_start_idx is not None and bit_end_idx is not None:
            if bit_start_idx < 0 or bit_end_idx > 16 or bit_start_idx > bit_end_idx:
                raise ValueError(f"Invalid bit range: ({bit_start_idx}, {bit_end_idx})")
            mask = torch.tensor((1 << bit_end_idx) - (1 << bit_start_idx), dtype=indices.dtype, device=indices.device)
            range_mask = torch.zeros_like(indices, dtype=torch.bool, device=indices.device)
            range_mask[weight_start_idx:weight_end_idx] = True
            indices = torch.where(range_mask, torch.bitwise_and(indices, mask), 0)
        else:
            range_mask = torch.ones_like(indices, dtype=torch.bool, device=indices.device)
            range_mask[:weight_start_idx] = False
            range_mask[weight_end_idx:] = False
            indices[~range_mask] = 0


    tensor = values[indices].reshape(tensor_shape)
    # print(f"tensor.shape:{tensor.shape}, tensor:{tensor}")


    return tensor

def Tensor_bit_flip(data, mask_data):
    """
    执行比特翻转，通过按位异或操作
    """
    return torch.bitwise_xor(data, mask_data)

def Error_injection(data, bit_error_rate, bit_w, mask_bits=None, weight_index_range=None):
    """
    错误注入函数，必须确保输入数据是无符号类型的[0, 2^bit_w - 1]，否则会出错。
    :param data: 输入张量，int16类型
    :param bit_error_rate: 比特错误率
    :param bit_w: 输入数据实际位宽（不超过16）
    :param mask_bits: 允许翻转的比特位置列表
    :param weight_index_range: 权重索引范围，(start_idx, end_idx) 或 None
    :return: 注错后的张量，int16类型
    """
    # 输入验证
    if data.dtype != torch.int16:
        raise TypeError(f"输入张量必须是 torch.int16 类型，当前类型为 {data.dtype}！")
    if not (0 < bit_w <= 16):
        raise ValueError(f"位宽 bit_w 必须在 1 到 16 之间，当前为 {bit_w}！")
    max_value = (1 << bit_w) - 1  # 例如 bit_w=8 时，max_value=255
    if (data < 0).any():
        raise ValueError(f"输入张量包含负值，必须为无符号整数 [0, {max_value}]！")
    if (data > max_value).any():
        raise ValueError(f"输入张量值超出范围 [0, {max_value}]，最大值为 {data.max().item()}！")

    if bit_error_rate != 0:
        prob = Prob(bit_error_rate, bit_w=bit_w, mask_bits=mask_bits)
        value_list, prob_list = prob.error_nbit_full()
        mask_data = Mask_bit_flip(value_list, prob_list, data, weight_index_range)
        bit_flip_data = Tensor_bit_flip(data, mask_data)
    else:
        bit_flip_data = data
    return bit_flip_data


def Mask_bit_flip_opt(value_list, prob_list, data, weight_index_range=None):
    """
    [优化版 - 修复RuntimeError] Mask_bit_flip
    优化点：使用 CDF + searchsorted 替代 multinomial
    修复点：解决 cdf /= cdf[-1] 的原位操作内存冲突问题
    """
    device = data.device
    # 确保 value 和 prob 是 tensor
    if not isinstance(value_list, torch.Tensor):
        values = torch.tensor(value_list, dtype=torch.int16, device=device)
    else:
        values = value_list.to(device=device, dtype=torch.int16)

    if not isinstance(prob_list, torch.Tensor):
        probabilities = torch.tensor(prob_list, device=device)
    else:
        probabilities = prob_list.to(device)

    tensor_shape = data.shape
    num_elements = data.numel()

    # --- 优化核心：使用逆变换采样替代 multinomial ---
    # 1. 构建累积分布函数 (CDF)
    cdf = torch.cumsum(probabilities, dim=0)

    # [修复部分] 使用 .clone() 避免原位操作冲突
    # 或者使用 cdf = cdf / cdf[-1] (非原位)
    cdf /= cdf[-1].clone()

    # 2. 生成均匀分布随机数
    rand_vals = torch.rand(num_elements, device=device)

    # 3. 使用 searchsorted 快速找到索引
    indices = torch.searchsorted(cdf, rand_vals)
    # ---------------------------------------------

    # 施加 weight_index_range 限制
    if weight_index_range is not None:
        weight_start_idx, weight_end_idx, bit_start_idx, bit_end_idx = weight_index_range

        if weight_start_idx < 0 or weight_end_idx > num_elements or weight_start_idx > weight_end_idx:
            raise ValueError(f"Invalid weight_index_range: {weight_index_range}")

        if bit_start_idx is not None and bit_end_idx is not None:
            if bit_start_idx < 0 or bit_end_idx > 16 or bit_start_idx > bit_end_idx:
                raise ValueError(f"Invalid bit range: ({bit_start_idx}, {bit_end_idx})")

            mask = torch.tensor((1 << bit_end_idx) - (1 << bit_start_idx), dtype=indices.dtype, device=device)

            range_mask = torch.zeros(num_elements, dtype=torch.bool, device=device)
            range_mask[weight_start_idx:weight_end_idx] = True

            indices = torch.where(range_mask, torch.bitwise_and(indices, mask), torch.tensor(0, device=device))
        else:
            range_mask = torch.ones(num_elements, dtype=torch.bool, device=device)
            range_mask[:weight_start_idx] = False
            range_mask[weight_end_idx:] = False
            indices[~range_mask] = 0

    tensor = values[indices].reshape(tensor_shape)
    return tensor

def Error_injection_opt(data, bit_error_rate, bit_w, mask_bits=None, weight_index_range=None):
    """
    [优化版] Error_injection
    优化点：调用 Mask_bit_flip_opt
    """
    # 输入验证保持不变
    if data.dtype != torch.int16:
        raise TypeError(f"输入张量必须是 torch.int16 类型，当前类型为 {data.dtype}！")
    if not (0 < bit_w <= 16):
        raise ValueError(f"位宽 bit_w 必须在 1 到 16 之间，当前为 {bit_w}！")

    # 移除 .item() 调用以减少同步，除非报错
    max_value = (1 << bit_w) - 1

    # 简单的 check，实际生产中为了速度可以注释掉这两行 assert
    # if (data < 0).any(): raise ValueError(...)
    # if (data > max_value).any(): raise ValueError(...)

    if bit_error_rate != 0:
        # 假设 Prob 类在外部定义，这里保持调用接口不变
        prob = Prob(bit_error_rate, bit_w=bit_w, mask_bits=mask_bits)
        value_list, prob_list = prob.error_nbit_full()

        # 调用优化后的 Mask_bit_flip_opt
        mask_data = Mask_bit_flip_opt(value_list, prob_list, data, weight_index_range)
        bit_flip_data = Tensor_bit_flip(data, mask_data)
    else:
        bit_flip_data = data

    return bit_flip_data


def Mask_bit_flip_cpu(value_list, prob_list, tensor_shape, weight_index_range=None):
    # 1. 强制转为 CPU Tensor
    if not isinstance(prob_list, torch.Tensor):
        probabilities = torch.tensor(prob_list, device='cpu')
    else:
        probabilities = prob_list.cpu()

    if not isinstance(value_list, torch.Tensor):
        values = torch.tensor(value_list, dtype=torch.int16, device='cpu')
    else:
        values = value_list.cpu().to(torch.int16)

    # 2. 计算元素总数
    num_elements = 1
    for s in tensor_shape: num_elements *= s

    # 3. CDF 计算 (CPU)
    cdf = torch.cumsum(probabilities, dim=0)
    cdf = cdf / cdf[-1]

    # 4. 生成随机数 (CPU) - 这一步极快
    rand_vals = torch.rand(num_elements, device='cpu')

    # 5. 查找索引 (CPU)
    indices = torch.searchsorted(cdf, rand_vals)

    # 6. 处理 Range (CPU)
    if weight_index_range is not None:
        weight_start_idx, weight_end_idx, bit_start_idx, bit_end_idx = weight_index_range
        if bit_start_idx is not None:
            mask = torch.tensor((1 << bit_end_idx) - (1 << bit_start_idx), dtype=indices.dtype, device='cpu')
            range_mask = torch.zeros(num_elements, dtype=torch.bool, device='cpu')
            range_mask[weight_start_idx:weight_end_idx] = True
            indices = torch.where(range_mask, torch.bitwise_and(indices, mask), torch.tensor(0, device='cpu'))
        else:
            range_mask = torch.ones(num_elements, dtype=torch.bool, device='cpu')
            range_mask[:weight_start_idx] = False
            range_mask[weight_end_idx:] = False
            indices[~range_mask] = 0

    return values[indices].reshape(tensor_shape)

def Error_injection_opt_cpu_gpu(data, bit_error_rate, bit_w, mask_bits=None, weight_index_range=None):
    """
    [无缓存集成版]
    每次调用都会重新生成 prob_list，逻辑最安全。
    Mask 生成强制在 CPU，避免 GPU 开销。
    """
    if data.dtype != torch.int16:
        raise TypeError(f"Input must be int16, got {data.dtype}")

    if bit_error_rate == 0:
        return data

    # 1. 每次都重新计算概率表 (Your original logic)
    prob = Prob(bit_error_rate, bit_w=bit_w, mask_bits=mask_bits)
    value_list, prob_list = prob.error_nbit_full()

    # 2. 调用 CPU 核心生成 Mask
    mask_cpu = Mask_bit_flip_cpu(
        value_list,
        prob_list,
        tensor_shape=data.shape,
        weight_index_range=weight_index_range
    )

    # 3. 搬运到 GPU 并应用
    mask_gpu = mask_cpu.to(data.device)
    return torch.bitwise_xor(data, mask_gpu)

def Error_injection_fewbit_1_31_find_error(data, bit_w, flip_positions):
    """
    错误注入函数，仅翻转指定位置和比特的权重
    :param data: 输入张量，int16类型，形状为 [rows, cols]
    :param bit_w: 输入数据实际位宽（不超过16）
    :param flip_positions: 翻转信息列表，每个元素为 [[row, col], bit_pos]，表示权重位置和要翻转的比特位置
                           bit_pos=0 表示最低位，bit_pos=bit_w-1 表示最高位
    :return: 注错后的张量，int16类型
    """
    device = data.device
    if data.dtype != torch.int16:
        raise TypeError("输入张量必须是 int16 类型!")
    if not (0 < bit_w <= 16):
        raise ValueError("位宽 bit_w 必须在 1 到 16 之间!")
    # 创建与输入张量形状相同的翻转掩码，初始为 0
    mask_data = torch.zeros_like(data, dtype=torch.int16, device=device)
    # 遍历翻转信息，设置掩码
    for pos, bit_pos in flip_positions:
        row, col = pos
        if row >= data.shape[0] or col >= data.shape[1]:
            raise ValueError(f"权重位置 ({row}, {col}) 超出张量形状 {data.shape}!")
        if bit_pos >= bit_w:
            raise ValueError(f"比特位置 {bit_pos} 超出位宽 {bit_w}!")
        # 计算比特掩码，直接使用 bit_pos 作为索引（bit_pos=0 为最低位）
        mask_data[row, col] = 1 << bit_pos  # 设置对应位置的比特为 1
    # 执行比特翻转
    bit_flip_data = torch.bitwise_xor(data, mask_data)
    return bit_flip_data

def Error_injection_fewbit(data, bit_w, flip_positions):
    """
    错误注入函数，仅翻转指定位置和比特的权重 (修正版：支持同位置多比特翻转)
    :param data: 输入张量，int16类型，形状为 [rows, cols]
    :param bit_w: 输入数据实际位宽（不超过16）
    :param flip_positions: 翻转信息列表，每个元素为 [[row, col], bit_pos]
    :return: 注错后的张量，int16类型
    """
    device = data.device

    # 1. 基础检查
    if data.dtype != torch.int16:
        raise TypeError(f"输入张量必须是 int16 类型! 当前为 {data.dtype}")
    if not (0 < bit_w <= 16):
        raise ValueError("位宽 bit_w 必须在 1 到 16 之间!")

    # 2. 创建掩码 (使用 int32 避免移位时的临时溢出，最后转回 int16)
    # 虽然 data 是 int16，但在计算 mask 时，Python 的 1<<15 是 32768，
    # 直接赋给 int16 (上限32767) 可能会产生警告或溢出。
    # 建议先用 int32/int64 累积掩码。
    mask_data = torch.zeros_like(data, dtype=torch.int32, device=device)

    # 3. 遍历翻转信息，设置掩码
    for pos, bit_pos in flip_positions:
        row, col = pos

        # 边界检查
        if row >= data.shape[0] or col >= data.shape[1]:
            raise ValueError(f"权重位置 ({row}, {col}) 超出张量形状 {data.shape}!")
        if bit_pos >= bit_w:
            raise ValueError(f"比特位置 {bit_pos} 超出位宽 {bit_w}!")

        # [核心修复] 使用 ^= (异或等于) 来累积掩码，防止覆盖
        # 如果同一个位置同一个 bit 被翻两次，也会自动抵消（变回不翻转），符合物理规律
        mask_data[row, col] ^= (1 << bit_pos)

    # 4. 类型转换并执行翻转
    # 将 mask 转回 int16 (此时符号位会被正确处理为补码)
    mask_data = mask_data.to(torch.int16)

    # 执行比特翻转
    bit_flip_data = torch.bitwise_xor(data, mask_data)

    return bit_flip_data

def Error_injection_random_and_fix(data, bit_error_rate, bit_w,
                             mask_bits=None,
                             weight_index_range=None,
                             fix_flip_positions=None,
                             fix_noflip_positions=None):
    """
    组合式错误注入函数：支持随机错误、固定翻转和固定不翻转
    :param data: 输入张量，int16类型
    :param bit_error_rate: 比特错误率（仅用于随机部分）
    :param bit_w: 输入数据实际位宽（不超过16）
    :param mask_bits: 允许翻转的比特位置列表（仅用于随机部分）
    :param weight_index_range: 权重索引范围（仅用于随机部分）
    :param fix_flip_positions: 固定翻转信息列表，每个元素为 [[row, col], bit_pos]
    :param fix_noflip_positions: 固定不翻转信息列表，每个元素为 [[row, col], bit_pos]
    :return: 注错后的张量，int16类型
    """
    # 输入验证
    if data.dtype != torch.int16:
        raise TypeError(f"输入张量必须是 torch.int16 类型，当前类型为 {data.dtype}！")
    if not (0 < bit_w <= 16):
        raise ValueError(f"位宽 bit_w 必须在 1 到 16 之间，当前为 {bit_w}！")
    max_value = (1 << bit_w) - 1
    if (data < 0).any():
        raise ValueError(f"输入张量包含负值，必须为无符号整数 [0, {max_value}]！")
    if (data > max_value).any():
        raise ValueError(f"输入张量值超出范围 [0, {max_value}]，最大值为 {data.max().item()}！")

    device = data.device

    # 1. 生成随机翻转掩码
    if bit_error_rate != 0:
        prob = Prob(bit_error_rate, bit_w=bit_w, mask_bits=mask_bits)
        value_list, prob_list = prob.error_nbit_full()
        mask_random = Mask_bit_flip(value_list, prob_list, data, weight_index_range)
    else:
        mask_random = torch.zeros_like(data, dtype=torch.int16, device=device)

    # 2. 生成固定翻转掩码
    mask_fix_flip = torch.zeros_like(data, dtype=torch.int16, device=device)
    if fix_flip_positions:
        for pos, bit_pos in fix_flip_positions:
            row, col = pos
            if row >= data.shape[0] or col >= data.shape[1]:
                raise ValueError(f"固定翻转位置 ({row}, {col}) 超出张量形状 {data.shape}!")
            if bit_pos >= bit_w:
                raise ValueError(f"固定翻转比特位置 {bit_pos} 超出位宽 {bit_w}!")
            # 设置该比特为1（需要翻转）
            mask_fix_flip[row, col] |= (1 << bit_pos)

    # 3. 生成固定不翻转掩码（实际上生成的是保护掩码）
    mask_protect = torch.zeros_like(data, dtype=torch.int16, device=device)
    if fix_noflip_positions:
        for pos, bit_pos in fix_noflip_positions:
            row, col = pos
            if row >= data.shape[0] or col >= data.shape[1]:
                raise ValueError(f"固定不翻转位置 ({row}, {col}) 超出张量形状 {data.shape}!")
            if bit_pos >= bit_w:
                raise ValueError(f"固定不翻转比特位置 {bit_pos} 超出位宽 {bit_w}!")
            # 设置该比特为1（需要保护）
            mask_protect[row, col] |= (1 << bit_pos)

    # 4. 组合掩码：随机掩码 + 固定翻转 - 保护比特
    # 思路：先将固定翻转加入随机掩码，然后清除保护比特
    combined_mask = mask_random.clone()

    # 添加固定翻转（使用按位或）
    combined_mask = torch.bitwise_or(combined_mask, mask_fix_flip)

    # 清除保护比特（需要保护的位置置为0）
    # 方法：将保护掩码取反，然后与combined_mask按位与
    # mask_protect中为1的比特需要保护（不翻转）
    protect_inverse = torch.bitwise_not(mask_protect)
    combined_mask = torch.bitwise_and(combined_mask, protect_inverse)

    # 5. 执行比特翻转
    bit_flip_data = torch.bitwise_xor(data, combined_mask)

    # 6. 验证：确保固定翻转确实发生了（调试用）
    if fix_flip_positions:
        for pos, bit_pos in fix_flip_positions:
            row, col = pos
            original_bit = (data[row, col] >> bit_pos) & 1
            flipped_bit = (bit_flip_data[row, col] >> bit_pos) & 1
            if original_bit == flipped_bit:
                print(f"警告：固定翻转位置 ({row}, {col}) 比特 {bit_pos} 未翻转！")

    return bit_flip_data


def compare_tensor_bits_raw(tensor1, tensor2, bit_w, print_diff_positions=False):
    """
    比较两个整型张量的比特差异，统计不同比特的总数，并可选打印不同比特的位置
    :param tensor1: 第一个输入张量（整型，如 int16 或 int8）
    :param tensor2: 第二个输入张量（与 tensor1 类型和形状相同）
    :param bit_w: 有效位宽（不超过张量类型的最大位宽）
    :param print_diff_positions: 是否打印所有不同比特的位置信息
    :return: 不同比特的总数
    """
    # 检查输入张量类型和形状
    if tensor1.dtype != tensor2.dtype:
        raise TypeError(f"两个张量的类型必须相同！tensor1: {tensor1.dtype}, tensor2: {tensor2.dtype}")
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"两个张量的形状必须相同！tensor1: {tensor1.shape}, tensor2: {tensor2.shape}")
    if not (0 < bit_w <= 16):
        raise ValueError("位宽 bit_w 必须在 1 到 16 之间!")
    if tensor1.dtype == torch.int8 and bit_w > 8:
        raise ValueError(f"位宽 {bit_w} 超出 int8 张量的最大位宽 8!")

    device = tensor1.device
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)

    # 计算异或结果，找出不同比特
    xor_result = torch.bitwise_xor(tensor1, tensor2)
    diff_count = torch.zeros_like(tensor1, dtype=torch.int32)

    # 统计每个元素的比特差异
    for i in range(bit_w):
        bit_diff = (xor_result >> i) & 1
        diff_count += bit_diff

    total_diff_bits = diff_count.sum().item()

    # 计算总比特数和比特差异比例
    rows, cols = tensor1.shape
    total_bits = rows * cols * bit_w
    diff_ratio = total_diff_bits / total_bits if total_bits > 0 else 0.0

    print(f"两个张量在 {bit_w} 位宽内的不同比特总数: {total_diff_bits} 比例: {diff_ratio}")

    # 可选打印不同比特的位置
    if print_diff_positions:
        print("不同比特的位置信息（格式：[row, col, bit_pos]）：")
        rows, cols = tensor1.shape
        for row in range(rows):
            for col in range(cols):
                if diff_count[row, col] > 0:
                    xor_value = xor_result[row, col].item()
                    for bit_pos in range(bit_w):
                        if (xor_value >> bit_pos) & 1:
                            print(f"[{row}, {col}, {bit_pos}]")

    return total_diff_bits


def compare_tensor_bits1(tensor1, tensor2, bit_w, print_diff_positions=False):
    """
    比较两个整型张量的比特差异，统计不同比特的总数，并可选打印不同比特的位置
    :param tensor1: 第一个输入张量（整型，如 int16 或 int8）
    :param tensor2: 第二个输入张量（与 tensor1 类型和形状相同）
    :param bit_w: 有效位宽（不超过张量类型的最大位宽）
    :param print_diff_positions: 是否打印所有不同比特的位置信息
    :return: 不同比特的总数，以及当 total_diff_bits=1 时返回不同比特的位置信息 [row, col, bit_pos]，否则返回 None
    """
    # 检查输入张量类型和形状
    if tensor1.dtype != tensor2.dtype:
        raise TypeError(f"两个张量的类型必须相同！tensor1: {tensor1.dtype}, tensor2: {tensor2.dtype}")
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"两个张量的形状必须相同！tensor1: {tensor1.shape}, tensor2: {tensor2.shape}")
    if not (0 < bit_w <= 16):
        raise ValueError("位宽 bit_w 必须在 1 到 16 之间!")
    if tensor1.dtype == torch.int8 and bit_w > 8:
        raise ValueError(f"位宽 {bit_w} 超出 int8 张量的最大位宽 8!")

    device = tensor1.device
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)

    # 计算异或结果，找出不同比特
    xor_result = torch.bitwise_xor(tensor1, tensor2)
    diff_count = torch.zeros_like(tensor1, dtype=torch.int32)

    # 统计每个元素的比特差异
    for i in range(bit_w):
        bit_diff = (xor_result >> i) & 1
        diff_count += bit_diff

    total_diff_bits = diff_count.sum().item()

    # 计算总比特数和比特差异比例
    rows, cols = tensor1.shape
    total_bits = rows * cols * bit_w
    diff_ratio = total_diff_bits / total_bits if total_bits > 0 else 0.0

    print(f"两个张量在 {bit_w} 位宽内的不同比特总数: {total_diff_bits} 比例: {diff_ratio}")

    # 初始化 diff_position
    diff_position = None

    # 可选打印不同比特的位置并检查 total_diff_bits=1 的情况
    if print_diff_positions or total_diff_bits == 1:
        if print_diff_positions:
            print("不同比特的位置信息（格式：[row, col, bit_pos]）：")
        rows, cols = tensor1.shape
        for row in range(rows):
            for col in range(cols):
                if diff_count[row, col] > 0:
                    xor_value = xor_result[row, col].item()
                    for bit_pos in range(bit_w):
                        if (xor_value >> bit_pos) & 1:
                            if print_diff_positions:
                                print(f"[{row}, {col}, {bit_pos}]")
                            if total_diff_bits == 1:
                                diff_position = [row, col, bit_pos]

    return total_diff_bits, diff_position

import torch

def compare_tensor_bits(tensor1, tensor2, bit_w, print_diff_positions=False):
    """
    比较两个整型张量的比特差异，统计不同比特的总数，并可选打印不同比特的位置
    :param tensor1: 第一个输入张量（整型，如 int16 或 int8）
    :param tensor2: 第二个输入张量（与 tensor1 类型和形状相同）
    :param bit_w: 有效位宽（不超过张量类型的最大位宽）
    :param print_diff_positions: 是否打印所有不同比特的位置信息
    :return: 不同比特的总数，以及当 total_diff_bits=1 时返回不同比特的位置信息 [row, col, bit_pos]，否则返回 None
    """
    # 检查输入张量类型和形状
    if tensor1.dtype != tensor2.dtype:
        raise TypeError(f"两个张量的类型必须相同！tensor1: {tensor1.dtype}, tensor2: {tensor2.dtype}")
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"两个张量的形状必须相同！tensor1: {tensor1.shape}, tensor2: {tensor2.shape}")
    if not (0 < bit_w <= 16):
        raise ValueError("位宽 bit_w 必须在 1 到 16 之间!")
    if tensor1.dtype == torch.int8 and bit_w > 8:
        raise ValueError(f"位宽 {bit_w} 超出 int8 张量的最大位宽 8!")

    device = tensor1.device
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)

    # 计算异或结果，找出不同比特
    xor_result = torch.bitwise_xor(tensor1, tensor2)
    diff_count = torch.zeros_like(tensor1, dtype=torch.int32)

    # 统计每个元素的比特差异
    for i in range(bit_w):
        bit_diff = (xor_result >> i) & 1
        diff_count += bit_diff

    total_diff_bits = diff_count.sum().item()

    # 计算总比特数和比特差异比例
    rows, cols = tensor1.shape
    total_bits = rows * cols * bit_w
    diff_ratio = total_diff_bits / total_bits if total_bits > 0 else 0.0

    print(f"两个张量在 {bit_w} 位宽内的不同比特总数: {total_diff_bits} 比例: {diff_ratio}")

    # 初始化 diff_position
    diff_position = None

    # 可选打印不同比特的位置
    if print_diff_positions:
        print("不同比特的位置信息（格式：[row, col, bit_pos]）：")
        nonzero_indices = torch.nonzero(diff_count, as_tuple=False)
        for row, col in nonzero_indices:
            xor_value = xor_result[row, col].item()
            for bit_pos in range(bit_w):
                if (xor_value >> bit_pos) & 1:
                    print(f"[{row.item()}, {col.item()}, {bit_pos}]")

    # 当 total_diff_bits == 1 时，高效定位差异比特
    if total_diff_bits == 1:
        # 找到 diff_count 中唯一非零元素的索引
        nonzero_indices = torch.nonzero(diff_count, as_tuple=False)
        if len(nonzero_indices) == 1:  # 确保只有一个差异
            row, col = nonzero_indices[0]
            row, col = row.item(), col.item()
            xor_value = xor_result[row, col].item()
            # 找到 xor_value 中唯一为 1 的比特位置
            for bit_pos in range(bit_w):
                if (xor_value >> bit_pos) & 1:
                    diff_position = [row, col, bit_pos]
                    if print_diff_positions:
                        print(f"[{row}, {col}, {bit_pos}]")
                    break

    return total_diff_bits, diff_position


def compare_tensor_bits_range_generate(tensor1, tensor2, bit_w):
    """
    比较两个张量的比特差异，返回包含所有翻转比特的最小权重索引范围
    如果没有翻转或范围计算失败，返回None

    返回: (start_idx, end_idx) 或 None
    """
    if tensor1.shape != tensor2.shape:
        return None

    # 计算异或结果
    xor_result = torch.bitwise_xor(tensor1, tensor2)

    # 找出有比特差异的位置
    diff_rows, diff_cols = torch.where(xor_result != 0)

    if len(diff_rows) == 0:
        return None

    # 计算一维索引
    cols = tensor1.shape[1]
    diff_indices = diff_rows * cols + diff_cols

    # 找到最小和最大索引
    min_idx = diff_indices.min().item()
    max_idx = diff_indices.max().item()

    # 返回范围（包含两端）
    return (min_idx, max_idx + 1)

if __name__ == '__main__':
    # torch.manual_seed(13)
    # # data = torch.randint(0, 63, (1, 200), dtype=torch.int16)
    # data = torch.zeros((1, 200), dtype=torch.int16)  # 确保data是无符号类型的整数，必须必须必须
    # # data = data + 120
    # bit_flip_data = Error_injection(data, 0.005, bit_w=8, weight_index_range=(0, 100, 0, 6))
    #
    # # bit_flip_data = Error_injection(data, 1.0, bit_w=8, weight_index_range=None)
    #
    # # flip_positions = [
    # #     [[0, 1], 7],
    # #     [[0, 3], 4]
    # # ]
    # # few_bitflip_data = Error_injection_fewbit(data, bit_w=8, flip_positions=flip_positions)
    # #
    # print("original_data:", data)
    # print("bit_flip_data:", bit_flip_data)
    # # print("few_bitflip_data:", few_bitflip_data)

    data1 = torch.zeros((1, 200), dtype=torch.int16)
    data2 = data1.clone()
    data2[0][2] = 8
    data2 = data2.to(torch.int16)
    compare_tensor_bits(data1, data2, bit_w=8, print_diff_positions=True)
