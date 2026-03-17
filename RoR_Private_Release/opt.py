import time
import random
import torch
import torch.nn as nn
import os
import csv
from collections import defaultdict
from typing import Final
import copy
import psutil
import gc
import json
import pickle
import pandas as pd

# === 引入您的自定义模块 ===
from custom_logger import logger
import time_config
from bit_flip_utils import *
from hadamard_utils import *
from quant import Quantizer, real_quantize, fake_quantize
from far_reproduction import compute_and_save_far_config, load_and_apply_far_config
from wrecon_reproduction import run_wrecon_reproduction, WReconLinear
from radar_defense import apply_radar_defense
# === 核心：引入统一工具链 ===
from modelutils import get_model_unified, get_model_layers, precompute_rope
from datautils import get_loaders, run_utility_evaluation, run_utility_evaluation_ac4, run_utility_evaluation_hellaswag
logger.info(f"程序启动时间：{time_config.APP_START_TIME}")
DEV = torch.device("cuda")


def process_calibration_data(dataloader, nsamples, seqlen, dev):
    """
    [辅助函数] 将 calibration dataloader (List) 转换为 Tensor 格式，
    供 Stage 1 的扫描函数使用。
    """
    cache = []
    # datautils 返回的 trainloader 通常是 [(inp, tar), ...] 列表
    # 或者对于 wikitext 是 list of tensors

    count = 0
    for data in dataloader:
        if count >= nsamples:
            break

        # 兼容不同的 dataloader 返回格式
        if isinstance(data, (tuple, list)):
            inp = data[0]  # 取 input_ids
        else:
            inp = data

        if inp.dim() == 2:
            cache.append(inp)
        else:
            cache.append(inp.unsqueeze(0))
        count += 1

    # 堆叠成 [nsamples, seqlen] 或 [nsamples, 1, seqlen]
    # bit_flip_utils 里的函数通常期望 testenc 是 [1, total_len] 或者 [batch, seqlen]
    # 这里我们将校准数据拼接成类似 testenc 的长 Tensor，或者保持 Batch 形式
    # 为了兼容 scan_outliers_layerwise 的 batch = testenc[:, (i*seqlen)...] 逻辑：

    # 策略：将校准样本拼接成一个长序列 [1, nsamples * seqlen]
    # 这样 scan 函数里的切片逻辑就能直接复用
    calib_tensor = torch.cat(cache, dim=1).to(dev)
    return calib_tensor


def run_mmlu_evaluation_2_4(model, tokenizer, device, shotnum=0, batch_size=1):
    """
    使用 lm-evaluation-harness 评估 MMLU 5-shot
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    logger.info(f">>> 开始 MMLU ({shotnum}-shot) 评估...")

    # 1. 将内存中的模型封装为 HFLM 对象
    # 注意：这就避免了重新从磁盘加载模型，评估的是当前的量化/攻击后模型
    hflm_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=str(device)
    )
    # # 混合几个科目来估算平均分 (包含文科理科)
    # tasks = [
    #     # "mmlu_abstract_algebra",  # 难 (Math)
    #     "mmlu_elementary_mathematics"  # 易 (Math)
    # ]

    tasks = ["mmlu_marketing"]
    # 2. 运行评估
    # mmlu 包含很多子任务，这里使用 'mmlu' 会运行所有子任务并取平均
    results = evaluator.simple_evaluate(
        model=hflm_model,
        tasks=tasks,
        num_fewshot=shotnum,  # MMLU 标准是 5-shot
        batch_size=batch_size
    )

    # 3. 提取结果
    # 格式通常是 results['results']['mmlu']['acc,none']
    metric_key = 'acc,none'  # 或者 'acc'，视版本而定

    # 计算所有 MMLU 子任务的平均分
    total_acc = 0
    count = 0
    for task_name, metrics in results['results'].items():
        if 'mmlu' in task_name:
            total_acc += metrics.get(metric_key, 0)
            count += 1

    avg_mmlu = total_acc / count if count > 0 else 0
    logger.info(f">>> MMLU 评估完成: Average Accuracy = {avg_mmlu:.4f}")

    return avg_mmlu, results



# ==============================================================================
# Stage 1: 模型优化与参数校准 (使用 Calibration Data)
# ==============================================================================
def stage1_optimize_model(model, calib_loader, dev, wbits, seqlen,
                          outlier_compute=False,
                          use_hadamard=False,
                          use_householder=False, h_sigma=6.0, h_rel=2.0, h_abs=1.0,
                          use_far=False, far_div=6, far_ratio=0.01,
                          use_wrecon=False, wrecon_args=None,
                          use_radar=False, radar_args=None,
                          # ----------------
                          get_hessian_grad=False,
                          model_name='opt',
                          model_type='opt',
                          dataset_name='wikitext2',
                          file_suffix='',
                          tokenizer=None):
    """
    阶段一：仅做优化和参数搜索。
    严禁接触测试集数据！
    """
    # logger.info(">>> 进入 Stage 1: 模型优化与校准阶段 (Calibration Phase)")
    logger.info(f">>> 进入 Stage 1: 模型优化与校准阶段 (Calibration Phase) | SeqLen={seqlen}")

    # 引入 FaRLinear 以便后续检查 (如果需要)
    from far_reproduction import FaRLinear

    # seqlen = model.seqlen
    model = model.to(dev)
    layers = get_model_layers(model)




    # 1. 预计算 RoPE (通用)
    rotary_pos_emb = precompute_rope(model, dev, seqlen)

    # 2. 准备 Model Kwargs
    model_kwargs = {}
    if rotary_pos_emb is not None:
        model_kwargs['position_embeddings'] = rotary_pos_emb

    # 3. 准备校准数据 Tensor
    # -------------------------------------------------------------------------
    # [警告] process_calibration_data 会耗尽 calib_loader 迭代器！
    # 后续不能再直接使用 calib_loader。
    # -------------------------------------------------------------------------
    calib_nsamples = 128
    calib_data_tensor = process_calibration_data(calib_loader, calib_nsamples, seqlen, dev)

    # 更新实际可用的样本数
    total_seqlen = calib_data_tensor.numel()
    actual_nsamples = total_seqlen // seqlen
    logger.info(f"校准数据准备完毕: {actual_nsamples} 样本用于优化统计")

    # 4. 路径准备
    base_temp_dir = f"tmp_data/{model_name}/{model_name}_{dataset_name}"
    outlier_path = f"{base_temp_dir}/householder/outlier_config{file_suffix}.csv"
    hap_path = f"{base_temp_dir}/hap_config.csv"
    gradient_dir = os.path.join(base_temp_dir, "gradients")
    create_dir(outlier_path)
    create_dir(gradient_dir)

    # ================= A. Outlier & Gradient Scan =================
    if outlier_compute:
        logger.info(f"[Stage 1] 计算 Outliers (使用校准集)...")
        scan_outliers_layerwise(
            model=model,
            testenc=calib_data_tensor,
            dev=dev,
            seqlen=seqlen,
            nsamples=min(128, actual_nsamples),
            filename=outlier_path,
            model_kwargs=model_kwargs,
            sigma_thresh=h_sigma, rel_thresh=h_rel, abs_thresh=h_abs
        )

        # scan_outliers_layerwise_add_grad(
        #     model=model,
        #     testenc=calib_data_tensor,
        #     dev=dev,
        #     seqlen=seqlen,
        #     nsamples=min(128, actual_nsamples),
        #     filename=outlier_path,
        #     model_kwargs=model_kwargs,
        #     sigma_thresh=h_sigma, rel_thresh=h_rel, abs_thresh=h_abs
        # )


        # scan_outliers_layerwise_L1_abssum(
        #     model=model,
        #     testenc=calib_data_tensor,
        #     dev=dev,
        #     seqlen=seqlen,
        #     nsamples=min(128, actual_nsamples),
        #     filename=outlier_path,
        #     model_kwargs=model_kwargs,
        #     sigma_thresh=h_sigma, rel_thresh=h_rel, abs_thresh=h_abs
        # )

        # scan_outliers_layerwise_average(
        #     model=model,
        #     testenc=calib_data_tensor,
        #     dev=dev,
        #     seqlen=seqlen,
        #     nsamples=min(128, actual_nsamples),
        #     filename=outlier_path,
        #     model_kwargs=model_kwargs,
        #     threshold_ratio=h_sigma, rel_thresh=h_rel, abs_thresh=h_abs
        # )

        torch.cuda.empty_cache()

        # logger.info(f"[Stage 1] 计算全局梯度敏感度 (使用校准集)...")
        # scan_layer_importance_by_global_grad(
        #     model=model,
        #     testenc=calib_data_tensor,
        #     dev=dev,
        #     seqlen=seqlen,
        #     nsamples=min(32, actual_nsamples),
        #     chunk_size=4,
        #     model_type=model_type,
        #     filename=hap_path
        # )
        # torch.cuda.empty_cache()

        # ================= B. Hadamard & Householder Rotation =================
    if use_hadamard:
        outlier_path = None
        logger.info(f"[Stage 1] 应用 Hadamard 旋转 (ViT)...")
        rotation_config = {
            "hadamard_mode": "All",
            "hadamard_target_layers": [],
            "householder_csv_path": outlier_path,
            "householder_config": {},
            "use_stack_acceleration": True,
            "use_cuda_kernel": False
        }
        inject_rotation_into_model(model, seed=42, rotation_config=rotation_config)

    if use_householder:
        logger.info(f"[Stage 1] 应用 Householder 旋转 (ViT)...")
        rotation_config = {
            # "hadamard_mode": "None",
            "hadamard_mode": "Use",
            "hadamard_target_layers": [],
            "householder_csv_path": outlier_path,
            "householder_config": {},
            # "use_stack_acceleration": False,
            # "use_cuda_kernel": True
            "use_stack_acceleration": True,
            "use_cuda_kernel": False  # 现在这个模式还有PPL错误，得debug才能用这个功能
        }
        inject_rotation_into_model(model, seed=42, rotation_config=rotation_config)
    else:
        logger.info(f"[Stage 1] Rotation: DISABLED")

    # if use_householder:
    #     logger.info(f"[Stage 1] 应用 Householder 旋转 (ViT)...")
    #     outlier_path = None  # linshi 修改
    #     rotation_config = {
    #         # "hadamard_mode": "None",
    #         "hadamard_mode": "Use",
    #         "hadamard_target_layers": [],
    #         "householder_csv_path": outlier_path,
    #         "householder_config": {},
    #         "use_stack_acceleration": True
    #     }
    #
    #     # [DEBUG] 切换到强制空注入模式
    #     # 注意：这里不需要 csv 路径，因为它会忽略所有配置
    #     inject_rotation_into_model_debug(model, seed=42, rotation_config=rotation_config)

    # ================= [新增] Competitor: FaR =================
    if use_far:
        # 构建保存路径
        # 例如: tmp_data/Llama-3.2-1B/Llama-3.2-1B_wikitext2/far/config_far-[0.01,6].pth
        far_save_dir = f"tmp_data/{model_name}/{model_name}_{dataset_name}/far"
        if not os.path.exists(far_save_dir):
            os.makedirs(far_save_dir, exist_ok=True)

        config_filename = f"config_far-[{far_ratio},{far_div}].pth"
        far_config_path = os.path.join(far_save_dir, config_filename)

        if os.path.exists(far_config_path):
            logger.info(f"检测到已存在的 FaR 配置: {far_config_path}，跳过生成。")
        else:
            # 计算梯度并保存配置 (这一步不返回修改后的模型，只存文件)
            compute_and_save_far_config(
                model, calib_loader, dev, far_config_path,
                far_ratio=far_ratio, div_factor=far_div, nsamples=min(32, actual_nsamples), chunk_size=32
            )

    # ================= C. Quantization Parameter Search =================
    logger.info(f"[Stage 1] 计算量化参数 (RTN/GPTQ)...")
    quantizers = {}

    chunk_size = 4
    num_layers = len(layers)

    if get_hessian_grad:
        model.train()
        pass
    else:
        model.eval()

    for start_idx in range(0, num_layers, chunk_size):
        end_idx = min(start_idx + chunk_size, num_layers)
        target_layers_range = range(start_idx, end_idx)

        for i in target_layers_range:
            layer = layers[i]
            # [重要] 确保能找到 FaRLinear
            # 因为 FaRLinear 继承自 nn.Linear，这里用 [nn.Linear] 是可以找到的
            # 但为了保险，可以显式检查
            subset = find_layers(layer, layers=[nn.Linear])

            for name, module in subset.items():
                key = f"{i}_{name}"
                quantizer = Quantizer()
                quantizer.configure(wbits, perchannel=True, sym=True, mse=False)
                quantizer.find_params(module.weight.data, weight=True)
                quantizers[key] = quantizer.cpu()



        torch.cuda.empty_cache()

    if use_far:
        if os.path.exists(far_config_path):
            logger.info(f">>> [Stage 2] 检测到 FaR 模式，正在加载配置: {far_config_path}")

            # 引入应用函数 (确保 far_reproduction.py 在路径下)
            model = load_and_apply_far_config(model, far_config_path, dev)

            # --- [DEBUG] 验证 FaR 是否真的生效 ---
            logger.info("-" * 40)
            logger.info(">>> [DEBUG] 正在验证 FaR 层是否已替换...")
            far_layer_count = 0
            verified_config = False

            for name, module in model.named_modules():
                # 检查类名是否包含 FaR (兼容 FaRLinear 和 FaRLinearOptimized)
                if "FaRLinear" in module.__class__.__name__:
                    far_layer_count += 1
                    # 抽查第一个层，看里面有没有规则
                    if not verified_config:
                        # 检查 buffer 是否非空
                        if hasattr(module, 'far_rows') and module.far_rows.numel() > 0:
                            logger.info(f"    ✅ 抽查层 '{name}': 类型正确，且包含 {module.far_rows.numel()} 条重连规则。")
                            verified_config = True
                        else:
                            logger.error(f"    ❌ 抽查层 '{name}': 类型正确，但配置为空！(防御将失效)")

            if far_layer_count > 0:
                logger.info(f"    ✅ 验证通过: 模型中共有 {far_layer_count} 个 Linear 层已替换为 FaR 层。")
            else:
                logger.error("    ❌ 验证失败: 模型中没有找到任何 FaRLinear 层！代码并未执行替换。")
                # 强制报错，防止跑无效实验浪费时间
                raise RuntimeError("FaR 模式已开启，但模型层未被替换。请检查 load_and_apply_far_config 逻辑。")
            logger.info("-" * 40)
            # -------------------------------------

        else:
            logger.warning(f"")
            logger.warning(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.warning(f"!!! 严重警告: 未找到 FaR 配置文件: {far_config_path}")
            logger.warning(f"!!! 模型将以 [无防御] 状态运行！请先运行 Stage 1 生成配置。")
            logger.warning(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # -------------------------------------------------------
    # 4. WRecon (防御层封装 - 第四位)
    # -------------------------------------------------------
    if use_wrecon:
        if wrecon_args:
            if len(wrecon_args) >= 3:
                G, P, K = wrecon_args
            elif len(wrecon_args) == 2:
                G, K = wrecon_args

        logger.info(f"*** [Step 4] Applying WRecon Defense ***")

        # 4.1 替换层 (传入 quantizers)
        model = run_wrecon_reproduction(
            model, quantizers, dev, G=G, P=P, K=K, wbits=wbits
        )
        # run_wrecon_reproduction

    # -------------------------------------------------------
    # 5. RADAR (运行时检测防御) <--- [新增]
    # -------------------------------------------------------
    if use_radar:
        # 默认参数
        G = 512
        interleave = True

        # 解析参数 radar_args = (G, interleave)
        if radar_args:
            if len(radar_args) >= 1: G = int(radar_args[0])
            if len(radar_args) >= 2: interleave = bool(radar_args[1])

        logger.info(f"*** [Step 5] Applying RADAR Defense (G={G}, Interleave={interleave}) ***")

        # 调用 radar_reproduction.py 中的函数
        model = apply_radar_defense(model, quantizers, dev, G=G, interleave=interleave)


    logger.info("[Stage 1] 优化完成。返回量化器和模型。")

    return model, quantizers, model_kwargs

# ==============================================================================
# Stage 2: 评估与攻击 (Evaluation & Attack) (使用 Test Data)
# ==============================================================================
def stage2_evaluate_attack(model, quantizers, testenc, model_kwargs, dev, wbits, seqlen,
                           flip_select=0, model_name='opt', dataset_name='wikitext2',
                           output_dir=None, file_suffix="", tokenizer=None):  # <--- [新增参数]
    """
    阶段二：验证模型鲁棒性 (Evaluation & Attack)。
    """
    # logger.info(">>> 进入 Stage 2: 评估与攻击阶段 (Evaluation Phase)")
    logger.info(f">>> 进入 Stage 2: 评估与攻击阶段 (Evaluation Phase) | SeqLen={seqlen}")

    # 1. 路径处理 [修改]
    # 如果未传入 output_dir，使用默认路径；如果传入了 (如 .../far)，则使用传入的
    if output_dir is None:
        output_dir = f"tmp_data/{model_name}/{model_name}_{dataset_name}"

    # 确保目录存在 (无论是默认的还是传入的 /far 目录)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Stage 2 输出目录: {output_dir}")

    # seqlen = model.seqlen
    model = model.to(dev)
    layers = get_model_layers(model)

    # 1. 准备测试数据
    if not isinstance(testenc, torch.Tensor):
        if hasattr(testenc, 'input_ids'):
            testenc = testenc.input_ids
        else:
            raise ValueError("testenc must be a Tensor or BatchEncoding")

    total_seqlen = testenc.numel()
    nsamples = total_seqlen // seqlen

    # 2. 准备输入缓存 (Catcher) - 针对测试集
    logger.info(f"正在对测试集 ({nsamples} 样本) 进行预处理...")

    nsamples = min(8, nsamples)
    logger.info(f"先只测试{nsamples}样本>>")

    # 一行代码搞定输入捕获
    inps, test_kwargs = prepare_model_inputs(model, testenc, dev, seqlen, nsamples)

    # 这里的 test_kwargs 已经包含了 mask 和 position_ids，合并原有的 kwargs (如 use_cache 等)
    # 注意：update 会覆盖重复键，确保 test_kwargs 是最新的
    final_kwargs = model_kwargs.copy()
    final_kwargs.update(test_kwargs)

    # ================= A. Baseline Evaluation =================
    logger.info(">>> 正在计算基准 PPL (Before Attack)...")
    original_loss, original_ppl = evaluate_full_model(
        model, layers, inps, testenc, dev, seqlen, nsamples,
        model_kwargs=final_kwargs
    )
    logger.info(f"基准全精度测试结果: Loss={original_loss:.4f}, PPL={original_ppl:.4f}")

    original_loss, original_ppl = evaluate_quantized_full_model(
            model, layers, quantizers, wbits, inps, testenc, dev, seqlen, nsamples,
            model_kwargs=final_kwargs
        )
    logger.info(f"基准量化后测试结果: Loss={original_loss:.4f}, PPL={original_ppl:.4f}")

    # ================= B. Attacks (Flip Select) =================


    # Mode 0: Random Flip (随机翻转)
    if flip_select == 0:
        logger.info(">>> Running Attack Mode 0: Random Flip")
        random_bits2flip_rate = 0.0003
        output_file = os.path.join(output_dir, f"flip_select_0{file_suffix}.txt")

        with open(output_file, "w") as f:
            f.write("Random_Seed\tFlipped_Loss\tFlipped_PPL\n")
        # seed_list = [1092, 1986, 1622, 949]
        # seed_list = [5735, 4227, 5134, 5581, 5466]
        for random_seed in range(4000, 6000):
        # for random_seed in seed_list:
            flipped_loss, flipped_ppl = flip_and_evaluate_randomflip_opt(
                model, layers, quantizers, inps,
                testenc, dev, wbits, seqlen, nsamples=min(2, nsamples),
                random_bits2flip_rate=random_bits2flip_rate,
                random_seed=random_seed,
                model_kwargs=final_kwargs
            )
            logger.info(f"Seed {random_seed}: PPL={flipped_ppl:.4f}")
            with open(output_file, "a") as f:
                f.write(f"{random_seed}\t{flipped_loss:.6f}\t{flipped_ppl:.6f}\n")

    # Mode 1: Locate Bad Bits (定位坏比特)
    elif flip_select == 1:
        logger.info(">>> Running Attack Mode 1: Locate Bad Bits")
        random_bits2flip_rate = 0.0003
        # random_seeds = [5735, 4277, 5134, 5581, 5466]  # Llama3-8B
        # random_seeds = [4254 ,5526 ,5379 ,4346 ,5922 ,5969 ,5318 ,5681 ,5859 ,5881 ,5734 ,
        #                 5765 ,4598 ,5134 ,4507 ,5782 ,4051 ,4528 ,5721 ,5498 ,4411 ,4470 ,
        #                 5976 ,5055 ,5864 ,5900 ,4145 ,5199 ,4900]  # Qwen2.5-7B
        random_seeds = [68,65,63,11,92]  # opt-125m

        # random_seeds = [1078 ,100 ,1621 ,114,1500,360,1974,643,1964,371,435,1671,1687,1134,1240,1965,222,294,1040,1761,272,1013,1062,1299,1209,1397,559,725,68]  # Qwen2.5-7B

        output_file = os.path.join(output_dir, f"flip_select_1{file_suffix}.txt")
        # ... (后续代码不变)

        with open(output_file, "w") as f:
            f.write(f"Seed\t层\t子模块\t权重索引\t比特\tppl\n")

        for seed in random_seeds:
            worst_layer, worst_name, worst_weight, worst_bit, max_ppl = find_bad_bit_optimized(
                model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
                random_bits2flip_rate=random_bits2flip_rate,
                random_seed=seed,
                model_kwargs=final_kwargs
            )
            logger.info(f"Found Bad Bit: Layer {worst_layer}, Bit {worst_bit}, PPL {max_ppl:.2f}")
            with open(output_file, "a") as f:
                f.write(f"{seed}\t{worst_layer}\t{worst_name}\t{worst_weight}\t{worst_bit}\t{max_ppl:.6f}\n")

    # Mode 2: 定向翻转 1 bit (Targeted 1-bit Flip)
    elif flip_select == 2:
        logger.info(">>> Running Attack Mode 2: Targeted 1-bit Flip")
        output_file = os.path.join(output_dir, f"flip_select_2{file_suffix}.txt")
        with open(output_file, "w") as f:
            f.write("layer_idx\tname_idx\tweight_idx\tbit_position\tloss\tppl\n")

        # 示例目标列表 (Channel, Row)
        channel_rows = [[17094, 1923], [11245, 1923], [10208, 1923]]

        for channel_row in channel_rows:
            channel, row = channel_row
            # 构造攻击指令: [layer, name, [row, col], bit, dummy_score]
            # 注意：权重通常是 (Out, In)，所以 index 取决于 shape。假设 channel是Out, row是In
            bad_bits = [1, "mlp.gate_proj", [channel, row], 7, 0]

            if bad_bits is not None:
                logger.info(f">>> Testing bad_bits: {bad_bits}")
                attack_logs = bad_bits
                layer_idx, layer_name, weight_idx, bit_pos, _ = attack_logs

                # 设置 Hook (可选)
                hook_layer_list = None
                # hook_layer_list = ['1.self_attn.v_proj']

                bad_flipped_loss, bad_flipped_ppl, _ = flip_and_evaluate_onebit_hook(
                    model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
                    attack_logs,  # 注意：函数内部叫 attack_log
                    hook_layer=hook_layer_list,
                    save_dir=None,
                    max_samples=2,
                    model_kwargs=final_kwargs
                )

                logger.info(f">>> Result: Loss={bad_flipped_loss:.6f}, PPL={bad_flipped_ppl:.6f}")
                with open(output_file, "a") as f:
                    f.write(
                        f"{layer_idx}\t{layer_name}\t{weight_idx}\t{bit_pos}\t{bad_flipped_loss:.6f}\t{bad_flipped_ppl:.6f}\n")

    # Mode 3: 定向施加 Δw (Direct Delta W - Quantized)
    elif flip_select == 3:
        logger.info(">>> Running Attack Mode 3: Direct Delta W (Quantized)")
        output_file = os.path.join(output_dir, f"flip_select_3{file_suffix}.txt")
        with open(output_file, "w") as f:
            f.write("layer_idx\tname_idx\tweight_idx\tdirection\tΔw\tloss\tppl\tΔloss/Δw\n")

        # 加载攻击列表 (需确保文件存在，或使用参数传入)
        attack_log_path = f"tmp_data/{model_name}/{model_name}_{dataset_name}/attack_logs.csv"
        if os.path.exists(attack_log_path):
            target_list = load_attack_logs(attack_log_path)
        else:
            logger.warning(f"攻击日志 {attack_log_path} 不存在，使用示例列表")
            target_list = [(1, 'self_attn.v_proj', [1100, 1512])]

        for layer, name, pos in target_list:
            # for direction in [0.1, -0.1]:  # 可以改为 ["+", "-"] 测试极限
            for direction in ["+", "-"]:  # 可以改为 ["+", "-"] 测试极限
                attack_logs = (layer, name, pos, direction)

                bad_flipped_loss, bad_flipped_ppl, real_delta_w = flip_and_evaluate_delta_w_hook(
                    model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
                    attack_logs,
                    hook_layer=None,
                    save_dir=None,
                    max_samples=2,
                    model_kwargs=final_kwargs
                )
                logger.info(f"{layer}\t{name}\t{pos}\t{direction}\t{real_delta_w:.6f}\t{bad_flipped_loss:.6f}\t{bad_flipped_ppl:.6f}\t")
                sensitivity = (bad_flipped_loss - original_loss) / (abs(real_delta_w) + 1e-12)

                with open(output_file, "a") as f:
                    f.write(
                        f"{layer}\t{name}\t{pos}\t{direction}\t{real_delta_w:.6f}\t{bad_flipped_loss:.6f}\t{bad_flipped_ppl:.6f}\t{sensitivity:.6f}\n")

    # Mode 4: 定向施加 Δw (Floating Point)
    elif flip_select == 4:
        logger.info(">>> Running Attack Mode 4: Direct Delta W (Float)")
        output_file = os.path.join(output_dir, f"flip_select_4{file_suffix}.txt")
        with open(output_file, "w") as f:
            f.write("layer_idx\tname_idx\tweight_idx\tdirection\tΔw\tloss\tppl\tΔloss/Δw\n")

        attack_log_path = f"tmp_data/{model_name}/{model_name}_{dataset_name}/attack_logs.csv"
        if os.path.exists(attack_log_path):
            target_list = load_attack_logs(attack_log_path)
        else:
            target_list = [(1, 'self_attn.v_proj', [1100, 1512])]

        for layer, name, pos in target_list:
            for direction in [0.1, -0.1]:
                attack_logs = (layer, name, pos, direction)

                bad_flipped_loss, bad_flipped_ppl, real_delta_w = flip_and_evaluate_float_delta_w_hook(
                    model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
                    attack_logs,
                    hook_layer=None,
                    save_dir=None,
                    max_samples=2,
                    model_kwargs=final_kwargs
                )

                sensitivity = (bad_flipped_loss - original_loss) / (abs(real_delta_w) + 1e-12)
                logger.info(f"{layer}\t{name}\t{pos}\t{direction}\t{real_delta_w:.6f}\t{bad_flipped_loss:.6f}\t{bad_flipped_ppl:.6f}\t")
                with open(output_file, "a") as f:
                    f.write(
                        f"{layer}\t{name}\t{pos}\t{direction}\t{real_delta_w:.6f}\t{bad_flipped_loss:.6f}\t{bad_flipped_ppl:.6f}\t{sensitivity:.6f}\n")

    # Mode 5: 单/多 bit翻转
    elif flip_select == 5:
        logger.info(">>> Running Attack Mode 5: few bit flip")
        output_file = os.path.join(output_dir, f"flip_select_5{file_suffix}.txt")
        with open(output_file, "w") as f:
            f.write("layer_idx\tname_idx\tweight_idx\tdirection\tΔw\tloss\tppl\tΔloss/Δw\n")

        attack_log_path = f"tmp_data/{model_name}/{model_name}_{dataset_name}/attack_logs.csv"
        if os.path.exists(attack_log_path):
            target_list = load_attack_logs(attack_log_path)
        else:
            target_list = [(1, 'self_attn.v_proj', [1100, 1512])]

        # current_attack_logs = [[1, '1_self_attn.v_proj', [1100, 1512], 7, 0]]
        # current_attack_logs = [[1, '1_self_attn.v_proj', [1444, 1512], 6, 0]]
        # current_attack_logs = [[1, '1_self_attn.v_proj', [2398, 1512], 5, 0]]

        # 2. [关键] 调用自动黑客函数，生成 Bypass 列表
        # 注意：这需要 model 已经应用了 RADAR


        # raw_target = [1, '1_self_attn.v_proj', [1100, 1512], 7, -0.09810066]  # llama2-7b
        raw_target = [1, '1_mlp.gate_proj', [17094, 1923], 7, -0.15864563]  # qwen2.5-7b
        # raw_target = [1, '1_mlp.gate_proj', [1865, 1923], 7, 0]  # qwen2.5-7b


        # raw_target = [1, '1_self_attn.v_proj', [200, 200], 7, 0]

        logger.info(f"raw_target={raw_target}")
        final_attack_logs = [raw_target]  # 默认回退




        # 动态选择攻击模式
        # 假设 args.opt_method 是 "radar" 或 "far"
        method_name = args.opt_method.lower() if args.opt_method else ""

        try:
            if "radar" in method_name:
                from radar_defense import generate_radar_collision_attack
                final_attack_logs = generate_radar_collision_attack(model, raw_target, quantizers)
                logger.info(f"🚀 [RADAR] Constructed Collision Attack with {len(final_attack_logs)} bits.")

            elif "householder" in method_name:
                # from hadamard_utils import generate_ror_whitebox_attack
                # final_attack_logs = generate_ror_whitebox_attack(model, raw_target, quantizers)
                # logger.info(f"🚀 [RADAR] Constructed Collision Attack with {len(final_attack_logs)} bits.")
                from hadamard_utils import generate_and_evaluate_ror_exact_whitebox, generate_and_evaluate_ror_exact_whitebox_fp

                # Step 1：验证 RoR 对原始单 bit 的防御效果
                flipped_loss_naive, flipped_ppl_naive = flip_and_evaluate_fewbit(
                    model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
                    attack_logs=[raw_target],
                    model_kwargs=final_kwargs,
                    debug_flip=True
                )
                logger.info(f">>> [RoR 防御效果] 原始单 bit 攻击后 PPL={flipped_ppl_naive:.6f}")

                # # Step 2：精确白盒攻击，复原单 bit 效果
                flipped_loss, flipped_ppl = generate_and_evaluate_ror_exact_whitebox(  # 量化版
                    model=model,
                    layers=layers,
                    quantizers=quantizers,
                    initial_input=inps,
                    testenc=testenc,
                    dev=dev,
                    wbits=wbits,
                    seqlen=seqlen,
                    nsamples=nsamples,
                    raw_target=raw_target,
                    model_kwargs=final_kwargs,
                )
                # flipped_loss, flipped_ppl = generate_and_evaluate_ror_exact_whitebox_fp(  # 全精度版
                #     model=model,
                #     layers=layers,
                #     initial_input=inps,
                #     testenc=testenc,
                #     dev=dev,
                #     seqlen=seqlen,
                #     nsamples=nsamples,
                #     raw_target=raw_target,
                #     model_kwargs=final_kwargs,
                # )
                logger.info(f">>> [RoR 精确白盒攻击 FP] PPL={flipped_ppl:.6f}")
                logger.info(f">>> 对比: 防御后={flipped_ppl_naive:.4f} vs 精确白盒={flipped_ppl:.4f}")




            elif "far" in method_name:
                from far_reproduction import generate_far_whitebox_attack
                # 调用我们刚写的函数
                final_attack_logs = generate_far_whitebox_attack(model, raw_target, quantizers)

                if len(final_attack_logs) > 1:
                    logger.info(
                        f"🚀 [FaR] Constructed White-box Attack with {len(final_attack_logs)} bits (1 Target + {len(final_attack_logs) - 1} Deads).")
                else:
                    logger.info(f"ℹ️ [FaR] Target not in protected rules. Using single bit attack.")

        except Exception as e:
            logger.warning(f"无法生成白盒攻击: {e}")
            # Fallback to raw target
            final_attack_logs = [raw_target]


        # final_attack_logs = [[1, '1_self_attn.v_proj', [1000, 805], 6, 0],  # Llama3-8B-householder
        #               [2, '1_self_attn.v_proj', [1000, 805], 5, 0],
        #               [3, '1_self_attn.v_proj', [1000, 2570], 7, 0]]

        logger.info(f"Final Attack List Size: {len(final_attack_logs)}")
        # logger.info(f"Final Attack List: {final_attack_logs}")

        ## PPL评估 ##
        # ======================================= #
        flipped_loss, flipped_ppl = flip_and_evaluate_fewbit(
            model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
            attack_logs=final_attack_logs,
            model_kwargs=final_kwargs,
            debug_flip=True
        )
        logger.info(f">>> Result: Loss={flipped_loss:.6f}, PPL={flipped_ppl:.6f}")
        # ======================================= #

        ## 测试模型权重恢复  ##
        # ======================================= #\
        original_loss, original_ppl = evaluate_quantized_full_model(
            model, layers, quantizers, wbits, inps, testenc, dev, seqlen, nsamples,
            model_kwargs=final_kwargs
        )
        logger.info(f"坏bit翻转后模型精度恢复结果: Loss={original_loss:.4f}, PPL={original_ppl:.4f}")


        # ======================================= #




        # with open(output_file, "a") as f:
        #     f.write(
        #         f"{layer_idx}\t{layer_name}\t{weight_idx}\t{bit_pos}\t{bad_flipped_loss:.6f}\t{bad_flipped_ppl:.6f}\n")

    # Mode 6: 随机翻转 + 固定保护策略 (Random + Fix)
    elif flip_select == 6:
        logger.info(">>> Running Attack Mode 6: Random + Fix Strategy")
        random_bits2flip_rate = 0.0003
        output_file = os.path.join(output_dir, f"flip_select_6{file_suffix}.txt")

        with open(output_file, "w") as f:
            f.write("Random_Seed\tFlipped_Loss\tFlipped_PPL\n")

        # 定义配置项: [layer, module, pos, flip_bits(empty=protect all)]
        error_config_items = [
            [1, "self_attn.v_proj", [1100, 1512], [7]]  # 保护该位置
        ]

        # random_seed_select = 1360
        for random_seed in range(3000, 4000):
            # # 对比：普通随机
            # loss_base, ppl_base, _ = flip_and_evaluate_random_and_fix_hook(
            #     model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
            #     random_bits2flip_rate=random_bits2flip_rate,
            #     random_seed=random_seed,
            #     error_config_items=None,  # 无保护
            #     model_kwargs=final_kwargs
            # )

            # 实验：带保护
            loss_fix, ppl_fix, _ = flip_and_evaluate_random_and_fix_hook(
                model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
                random_bits2flip_rate=random_bits2flip_rate,
                random_seed=random_seed,
                error_config_items=error_config_items,  # 带保护
                model_kwargs=final_kwargs
            )

            # logger.info(f"Seed {random_seed}: Base PPL={ppl_base:.4f} -> Fix PPL={ppl_fix:.4f}")
            with open(output_file, "a") as f:
                f.write(f"{random_seed}\t{loss_fix:.6f}\t{ppl_fix:.6f}\n")

    # Mode 8: PBS 攻击 (Progressive Bit Search)
    elif flip_select == 8:
        # eval_steps = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100]
        # eval_steps = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        eval_steps = list(range(1, 20))  # 取连续整数，细致分析是哪个bit造成的突变
        # eval_steps = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150, 170, 200, 250, 300, 350, 400]
        # eval_steps = [1, 2, 3, 5]

        max_attack_iters = max(eval_steps)
        pbs_attack_path = os.path.join(output_dir, f"pbs_attack{file_suffix}.csv")
        curve_path = os.path.join(output_dir, f"pbs_damage_curve{file_suffix}.csv")
        pbs_logs = []

        # 1. 尝试读取现有日志
        if os.path.exists(pbs_attack_path):
            logger.info(f">>> [PBS] 检测到现有的攻击日志: {pbs_attack_path}")
            try:
                import csv
                with open(pbs_attack_path, mode='r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader: pbs_logs.append(row)
                logger.info(f">>> [PBS] 读取成功，共 {len(pbs_logs)} 行。")
            except Exception as e:
                logger.error(f">>> [PBS] 读取 CSV 失败: {e}")
                pbs_logs = []

        # 2. 如果没有日志，执行搜索
        # 注意：PBS 搜索是一个耗时过程，通常使用校准集或小规模测试集
        # 为了演示，这里直接调用 progressive_bit_search_opt
        if not pbs_logs:
            logger.info(f">>> [PBS] 启动新的 PBS 搜索...")
            pbs_logs = progressive_bit_search_opt(
                model=model,
                layers=layers,
                testenc=testenc,  # 使用测试集进行搜索(白盒)或校准集(黑盒)
                dev=dev,
                wbits=wbits,
                quantizers=quantizers,
                nsamples=3,
                seqlen=seqlen,
                attack_iters=max_attack_iters,
                chunk_size=4,
                output_csv=pbs_attack_path
            )

        print(f"pbs_logs: {pbs_logs}")

        # 3. 生成攻击曲线
        logger.info(f">>> [PBS] 生成攻击曲线: {curve_path}")
        damage_results = []

        original_loss, original_ppl = evaluate_full_model(
            model, layers, inps, testenc, dev, seqlen, nsamples,
            model_kwargs=final_kwargs
        )
        logger.info(f"PBS准备攻击前的全精度测试结果: Loss={original_loss:.4f}, PPL={original_ppl:.4f}")

        original_loss, original_ppl = evaluate_quantized_full_model(
            model, layers, quantizers, wbits, inps, testenc, dev, seqlen, nsamples,
            model_kwargs=final_kwargs
        )
        logger.info(f"PBS准备攻击前的量化精度测试结果: Loss={original_loss:.4f}, PPL={original_ppl:.4f}")

        logger.info(f"Top-{0}: PPL={original_ppl:.4f}")
        damage_results.append({"Top-K": 0, "Loss": original_loss, "PPL": original_ppl})  # 将不攻击作为基准(衡量优化算法的基础损失)
        for k in eval_steps:
            if k > len(pbs_logs): break
            current_attack_logs = pbs_logs[:k]

            flipped_loss, flipped_ppl = flip_and_evaluate_fewbit(
                model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
                attack_logs=current_attack_logs,
                model_kwargs=final_kwargs,
                debug_flip=True
            )

            logger.info(f"Top-{k}: PPL={flipped_ppl:.4f}")
            damage_results.append({"Top-K": k, "Loss": flipped_loss, "PPL": flipped_ppl})

        pd.DataFrame(damage_results).to_csv(curve_path, index=False)
        original_loss, original_ppl = evaluate_full_model(
            model, layers, inps, testenc, dev, seqlen, nsamples,
            model_kwargs=final_kwargs
        )
        logger.info(f"攻击后恢复原始模型测试结果: Loss={original_loss:.4f}, PPL={original_ppl:.4f}")

    elif flip_select == 81:
        # -------------------------------------------------------------------------
        # PBS 攻击评估 (PPL + Utility)
        # -------------------------------------------------------------------------

        # 评估步长 (Top-K)
        eval_steps = [0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # eval_steps = [0, 1, 2, 3, 5]
        # eval_steps = [0, 10, 50, 100] # 调试用短列表

        max_attack_iters = max(eval_steps)
        pbs_attack_path = os.path.join(output_dir, f"pbs_attack{file_suffix}.csv")
        curve_path = os.path.join(output_dir, f"pbs_damage_curve_utility{file_suffix}.csv")
        pbs_logs = []

        # 1. 读取日志 (PBS Log)
        if os.path.exists(pbs_attack_path):
            logger.info(f">>> [PBS] 加载攻击日志: {pbs_attack_path}")
            try:
                import csv
                with open(pbs_attack_path, mode='r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader: pbs_logs.append(row)
                logger.info(f">>> [PBS] 加载成功，共 {len(pbs_logs)} 条翻转记录。")
            except Exception as e:
                logger.error(f">>> [PBS] 读取失败: {e}")
                exit(1)  # 没有日志就无法复现攻击，直接退出
        else:
            logger.error(f">>> [PBS] 未找到日志文件: {pbs_attack_path}。请先运行搜索步骤生成日志。")
            exit(1)

        # 2. 循环评估
        results = []
        logger.info(f">>> [PBS] 开始多维评估 (PPL + MMLU + PIQA)...")
        for k in eval_steps:
            logger.info(f"\n=== Evaluating Top-{k} Flips ===")

            current_logs = pbs_logs[:k]  # 取前 k 个翻转

            # --- [Step A] 测量 PPL (使用原有的临时翻转函数) ---
            # 这个函数逻辑独立，测完会自动复原，适合测 PPL
            if k == 0:
                # Baseline
                loss, ppl = evaluate_full_model(model, layers, inps, testenc, dev, seqlen, nsamples,
                                                model_kwargs=final_kwargs)
            else:
                # 你的 flip_and_evaluate_fewbit 函数已经包含了 "翻转->测PPL->恢复" 的逻辑
                loss, ppl = flip_and_evaluate_fewbit(
                    model, layers, quantizers, inps, testenc, dev, wbits, seqlen, nsamples,
                    attack_logs=current_logs,  # 传入 logs
                    model_kwargs=final_kwargs,
                    debug_flip=False
                )
            logger.info(f"   [Metric] PPL: {ppl:.4f}")

            # --- [Step B] 测量 Utility (MMLU/PIQA) ---
            # 需要手动应用攻击到模型权重，测完后再恢复

            mmlu_score = 0.0
            piqa_score = 0.0
            hs_acc = 0.0

            eval_batch_size = 64
            shotnum = 5  # 零/5 样本
            limit = 200  # 只取前100道题

            if k == 0:
                # Baseline Utility (无攻击)
                # limit=None 跑全量子任务 (或者 100 加速)
                # scores = run_utility_evaluation(model, tokenizer, dev, shotnum=shotnum, limit=limit)  #
                scores = run_utility_evaluation_hellaswag(model, tokenizer, dev, shotnum=shotnum, limit=limit)  #

                mmlu_score = scores.get('MMLU (Avg)', 0)
                piqa_score = scores.get('PIQA', 0)
                hs_acc = scores.get('HellaSwag', 0)
            else:
                # 1. 应用攻击 (修改 model.weight)
                # 使用新写的函数，严格复用 Error_injection_fewbit

                scores = apply_attack_and_evaluate_utility(model,layers,quantizers,current_logs,wbits,dev,tokenizer,
                    eval_batch_size=eval_batch_size,
                    shotnum=shotnum,  # 比如 5
                    limit=limit,  # 比如 100
                    debug_flip=True  # 开启校验
                )

                mmlu_score = scores.get('MMLU (Avg)', 0)
                piqa_score = scores.get('PIQA', 0)
                hs_acc = scores.get('HellaSwag', 0)

            logger.info(f" [Metric]   HellaSwag: {hs_acc:.4f} | MMLU: {mmlu_score:.4f} | PIQA: {piqa_score:.4f}")

            # --- [Step C] 保存数据 ---
            results.append({
                "Top-K": k,
                "wikitext": ppl,
                "HellaSwag": hs_acc,
                "MMLU": mmlu_score,
                "PIQA": piqa_score,
                "Loss": loss
            })

            # 实时写入 CSV
            pd.DataFrame(results).to_csv(curve_path, index=False)

        logger.info(f">>> [PBS] 评估完成! 结果已保存: {curve_path}")

    elif flip_select == 90:
        logger.info(">>> Running Attack Mode 9: Time Test (Latency Benchmark)")
        output_file = os.path.join(output_dir, f"latency_benchmark{file_suffix}.txt")

        # 1. 准备一个真实的 model_kwargs (从数据集中取一个样本来生成)
        # 这步是为了获取正确的 attention_mask 和 position_ids
        inps, test_kwargs = prepare_model_inputs(model, testenc, dev, seqlen, 1)
        final_kwargs = model_kwargs.copy()
        final_kwargs.update(test_kwargs)

        # 2. 运行基准测试
        # 建议测试不同的 Batch Size (例如 1 和 32)
        # Batch=1 模拟在线推理延迟
        # Batch=32 模拟离线处理吞吐量

        results = []

        # Test Case 1: Latency (BS=1)
        lat_ms, _ = benchmark_inference_latency(
            model, layers, dev, seqlen, wbits,
            model_kwargs=final_kwargs,
            batch_size=1, repeats=50
        )
        results.append(f"Batch=1, Seq={seqlen}: {lat_ms:.2f} ms")

        # Test Case 2: Throughput (BS=8 or 16 depending on memory)
        # 注意：如果显存不够，把 batch 改小
        try:
            lat_ms_batch, tps = benchmark_inference_latency(
                model, layers, dev, seqlen, wbits,
                model_kwargs=final_kwargs,
                batch_size=128, repeats=20
            )
            results.append(f"Batch=8, Seq={seqlen}: {tps:.2f} tokens/s")
        except RuntimeError:
            logger.warning("Batch=8 OOM, skipping.")

        # 写入文件
        with open(output_file, "w") as f:
            f.write("\n".join(results))

    elif flip_select == 91:  # 论文数据版本
        logger.info(">>> Running Attack Mode 9: Optimized Latency Test")  # RQ5
        output_file = os.path.join(output_dir, f"latency_optimized{file_suffix}.txt")

        # 准备 Kwargs
        inps, test_kwargs = prepare_model_inputs(model, testenc, dev, seqlen, 1)
        final_kwargs = model_kwargs.copy()
        final_kwargs.update(test_kwargs)

        results = []

        # 测试 1: 低延迟场景 (Batch=1)
        try:
            lat_1 = benchmark_inference_latency_compiled(
                model, layers, dev, seqlen, wbits,
                model_kwargs=final_kwargs,
                batch_size=1, repeats=100
            )
            results.append(f"Batch=1: {lat_1:.2f} ms")
        except Exception as e:
            logger.error(f"BS=1 Benchmark Failed: {e}")

        # 测试 2: 高吞吐场景 (Batch=16)
        try:
            lat_16 = benchmark_inference_latency_compiled(
                model, layers, dev, seqlen, wbits,
                model_kwargs=final_kwargs,
                batch_size=16, repeats=50
            )
            results.append(f"Batch=16: {lat_16:.2f} ms (Avg per batch)")
        except Exception as e:
            logger.warning(f"BS=16 Skipped (OOM or Error): {e}")

        with open(output_file, "w") as f:
            f.write("\n".join(results))

    elif flip_select == 9_2_26:
        logger.info(">>> Running Attack Mode 91: Native Eager Mode Latency Test")  # RQ5
        output_file = os.path.join(output_dir, f"latency_optimized{file_suffix}.txt")

        results = []

        # ==========================================
        # 测试 1: 低延迟场景 (Batch=1)
        # ==========================================
        try:
            # 专门为 Batch=1 准备 kwargs
            _, test_kwargs_bs1 = prepare_model_inputs(model, testenc, dev, seqlen, 1)
            final_kwargs_bs1 = model_kwargs.copy() if model_kwargs is not None else {}
            final_kwargs_bs1.update(test_kwargs_bs1)

            lat_1 = benchmark_inference_latency(
                model, layers, dev, seqlen, wbits,
                model_kwargs=final_kwargs_bs1,
                batch_size=1, repeats=100
            )
            results.append(f"Batch=1: {lat_1:.2f} ms")
        except Exception as e:
            logger.error(f"BS=1 Benchmark Failed: {e}")

        # ==========================================
        # 测试 2: 高吞吐场景 (Batch=16)
        # ==========================================
        try:
            # [关键修复]：专门为 Batch=16 重新生成 kwargs，防止 mask/position_ids 维度报错
            _, test_kwargs_bs16 = prepare_model_inputs(model, testenc, dev, seqlen, 16)
            final_kwargs_bs16 = model_kwargs.copy() if model_kwargs is not None else {}
            final_kwargs_bs16.update(test_kwargs_bs16)

            lat_16 = benchmark_inference_latency(
                model, layers, dev, seqlen, wbits,
                model_kwargs=final_kwargs_bs16,
                batch_size=16, repeats=50
            )
            results.append(f"Batch=16: {lat_16:.2f} ms")
        except Exception as e:
            logger.warning(f"BS=16 Skipped (OOM or Error): {e}")

        # 将结果写入文件
        with open(output_file, "w") as f:
            f.write("\n".join(results))

        logger.info(f">>> 延迟测试结果已保存至: {output_file}")

    # flip_select == 9 调用处
    elif flip_select == 9:
        logger.info(">>> Running Attack Mode 9: Latency Benchmark")
        output_file = os.path.join(output_dir, f"latency{file_suffix}.txt")

        results = []
        for ns in [1, 8]:
            try:
                inps, test_kwargs = prepare_model_inputs(model, testenc, dev, seqlen, ns)
                final_kwargs = {**(model_kwargs or {}), **test_kwargs}

                avg, std = benchmark_inference_latency(
                    model, layers, inps, dev, seqlen,
                    model_kwargs=final_kwargs,
                    repeats=50, warmup=10
                )
                results.append(f"nsamples={ns}: {avg:.2f} ± {std:.2f} ms")
                logger.info(f"nsamples={ns}: {avg:.2f} ± {std:.2f} ms")
            except Exception as e:
                logger.warning(f"nsamples={ns} failed: {e}")
                results.append(f"nsamples={ns}: Failed ({e})")

        with open(output_file, "w") as f:
            f.write("\n".join(results))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--model-type', type=str, default='opt', help='opt, llama, qwen')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='wikitext2, ptb, c4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seqlen', type=int, default=2048, help='Fixed sequence length for consistency (default: 2048)')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--wbits', type=int, default=16)

    # [Stage 2] 攻击模式
    parser.add_argument('--flip-select', type=int, default=0, help='0=Random, 1=Search, 2=Targeted...')

    # [统一接口] 优化方法 (支持参数解析)
    # 示例: "householder", "far-[0.01, 6]", 或不传(Baseline)
    parser.add_argument('--opt-method', type=str, default=None,
                        help='Optimization method, e.g., "householder" or "far-[0.01, 6]"')

    args = parser.parse_args()

    # ==========================================
    # 1. 解析 opt-method 参数 & 定义实验目录
    # ==========================================
    use_hadamard = False
    use_householder = False; h_sigma = 6.0; h_rel = 2.0; h_abs = 1.0
    use_far = False; far_ratio = 0.01; far_div = 6

    use_wrecon = False
    wrecon_args = None

    use_radar = False  # <--- [新增]
    radar_args = None  # <--- [新增]

    # 基础路径: tmp_data/model/model_dataset/
    model_name = os.path.basename(args.model)
    base_dir = f"tmp_data/{model_name}/{model_name}_{args.dataset}"

    # 默认设置 (Baseline)
    method_dir_name = "baseline"
    file_suffix = "_baseline"

    if args.opt_method is not None:
        method_str = args.opt_method.strip()  # 保留原始大小写，不去掉空格以免影响解析
        method_key = method_str.split('-')[0].lower()  # 提取 "far", "householder"

        if 'hadamard' in method_key:
            use_hadamard = True
            method_dir_name = "hadamard"
            file_suffix = "_hadamard"
            logger.info(">>> Option Selected: Hadamard Optimization")

        elif 'householder' in method_key:
            use_householder = True
            method_dir_name = "householder"
            safe_tag = method_str.replace(' ', '')
            file_suffix = f"_{safe_tag}"
            logger.info(">>> Option Selected: Householder Optimization")

            # 解析自定义参数: householder--[6.0, 2.0, 1.0]
            try:
                if '[' in method_str:
                    content = method_str.split('[')[1].split(']')[0]
                    parts = [x.strip() for x in content.split(',')]

                    if len(parts) >= 1: h_sigma = float(parts[0])
                    if len(parts) >= 2: h_rel = float(parts[1])
                    if len(parts) >= 3: h_abs = float(parts[2])

                    logger.info(f"    [Config] Sigma Thresh: {h_sigma} * std")
                    logger.info(f"    [Config] Relative Thresh: {h_rel} * mean")
                    logger.info(f"    [Config] Absolute Thresh: {h_abs}")

            except Exception as e:
                logger.warning(f"⚠️ 解析 Householder 参数失败: {e}，将使用默认值 [6.0, 2.0, 1.0]")

        # --- Case B: FaR (Competitor) ---
        elif 'far' in method_key:
            use_far = True
            method_dir_name = "far"
            # 直接使用用户输入的完整字符串作为后缀，如 "_far-[0.01, 6]"
            # 替换掉可能导致路径问题的字符 (虽然 [] 在 Linux 文件名中合法，但保持原样更直观)
            safe_tag = method_str.replace(' ', '')  # 去掉空格 far-[0.01,6]
            file_suffix = f"_{safe_tag}"

            try:
                # 解析参数: far-[0.01, 6] -> 0.01, 6
                if '[' in method_str and ']' in method_str:
                    content = method_str.split('[')[1].split(']')[0]
                    parts = [x.strip() for x in content.split(',')]
                    if len(parts) == 2:
                        far_ratio = float(parts[0])
                        far_div = int(parts[1])

                logger.info(f">>> Option Selected: FaR (Forget and Rewire)")
                logger.info(f"    - Ratio: {far_ratio * 100}%")
                logger.info(f"    - Div Factor: {far_div}")

            except Exception as e:
                logger.warning(f"FaR 参数解析异常，使用默认值 (0.01, 6)。错误: {e}")

        # --- [新增] Case C: WRecon (Competitor) ---
        elif 'wrecon' in method_key:
            use_wrecon = True
            method_dir_name = "wrecon"
            # 使用用户输入的字符串作为后缀，例如 "_wrecon-[4,0.8,4]"
            safe_tag = method_str.replace(' ', '')
            file_suffix = f"_{safe_tag}"

            # 解析参数: wrecon-[G, P, K]
            try:
                if '[' in method_str and ']' in method_str:
                    content = method_str.split('[')[1].split(']')[0]
                    parts = [x.strip() for x in content.split(',')]
                    if len(parts) == 3:
                        # G(int), P(float), K(int)
                        g_val = int(parts[0])
                        p_val = float(parts[1])
                        k_val = int(parts[2])
                        wrecon_args = (g_val, p_val, k_val)

                logger.info(f">>> Option Selected: WRecon (Weight Reconstruction)")
                logger.info(f"    - G={wrecon_args[0]}, P={wrecon_args[1]}, K={wrecon_args[2]}")

            except Exception as e:
                logger.warning(f"WRecon 参数解析异常，使用默认值 (16, 0.8, 8)。错误: {e}")

        # --- [新增] Case D: RADAR ---
        elif 'radar' in method_key:
            use_radar = True
            method_dir_name = "radar"
            safe_tag = method_str.replace(' ', '')
            file_suffix = f"_{safe_tag}"

            # 解析参数: radar-[G, Interleave] -> 例如 radar-[512, 1]
            try:
                if '[' in method_str and ']' in method_str:
                    content = method_str.split('[')[1].split(']')[0]
                    parts = [x.strip() for x in content.split(',')]

                    # radar_args = [G, interleave]
                    radar_args = []
                    if len(parts) >= 1: radar_args.append(int(parts[0]))
                    if len(parts) >= 2: radar_args.append(int(parts[1]))

                logger.info(f">>> Option Selected: RADAR (Run-time Detection)")
                if radar_args:
                    logger.info(f"    - Params: {radar_args}")

            except Exception as e:
                logger.warning(f"RADAR 参数解析异常，使用默认值 (512, True)。错误: {e}")

        else:
            # 其他情况或拼写错误
            logger.warning(f"未识别的方法 '{method_str}'，将作为 Baseline 运行。")

    else:
        logger.info(">>> Option Selected: Baseline (No Optimization)")

    # 最终构建实验日志目录
    # e.g. tmp_data/.../far/
    experiment_dir = os.path.join(base_dir, method_dir_name)
    os.makedirs(experiment_dir, exist_ok=True)

    logger.info(f"实验结果将保存至: {experiment_dir}")
    logger.info(f"文件后缀 (Suffix): {file_suffix}")

    # ==========================================
    # 2. 加载模型与数据
    # ==========================================
    model = get_model_unified(args.model)

    # [关键] 强制统一定义 seqlen，覆盖模型可能的默认值 (如 4096->2048)
    # 这样所有实验都基于相同的上下文长度，PPL 可比
    unified_seqlen = args.seqlen
    if hasattr(model, 'seqlen'):
        model.seqlen = unified_seqlen

    logger.info(f"正在加载数据集: {args.dataset} | SeqLen 设置为: {unified_seqlen}")

    # [关键] 将 seqlen 传入 get_loaders，确保数据切分一致
    calib_loader, test_loader = get_loaders(
        args.dataset, nsamples=args.nsamples, seqlen=unified_seqlen, seed=args.seed, model=args.model
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)




    # from datautils import simple_mmlu_eval_fast
    # scores = simple_mmlu_eval_fast(
    #     model, tokenizer, dev,
    #     n_shot=5,
    #     batch_size=64,
    #     limit=100
    # )

    # from datautils import simple_mmlu_eval_robust
    #
    # scores = simple_mmlu_eval_robust(
    #     model, tokenizer, dev,
    #     n_shot=5,
    #     batch_size=64,
    #     limit=100
    # )



    # 1. 快速验证 (调试用，跑得飞快)
    # limit=10, 可以在几分钟内跑完，验证代码没崩，分数大概正常
    # scores_debug = run_utility_evaluation(model, tokenizer, dev, limit=10)



    # ==========================================
    # 3. 运行 Stage 1: 优化
    # ==========================================
    model, quantizers, model_kwargs = stage1_optimize_model(
        model=model,
        calib_loader=calib_loader,
        dev=DEV,
        wbits=args.wbits,
        seqlen=unified_seqlen, # [修改] 传入统一的 seqlen
        outlier_compute=use_householder,
        use_hadamard=use_hadamard,
        use_householder=use_householder, h_sigma=h_sigma, h_rel=h_rel, h_abs=h_abs,
        use_far=use_far, far_div=far_div, far_ratio=far_ratio,
        use_wrecon=use_wrecon, wrecon_args=wrecon_args,
        use_radar=use_radar, radar_args=radar_args,
        # ---------------------
        model_name=model_name,
        model_type=args.model_type,
        dataset_name=args.dataset,
        file_suffix=file_suffix,
        tokenizer=tokenizer
    )

    # ==========================================
    # 4. 运行 Stage 2: 评估与攻击
    # ==========================================
    # [关键] 将计算好的 路径 和 后缀 传入 Stage 2
    stage2_evaluate_attack(
        model=model,
        quantizers=quantizers,
        testenc=test_loader,
        model_kwargs=model_kwargs,
        dev=DEV,
        wbits=args.wbits,
        seqlen=unified_seqlen,  # [修改] 传入统一的 seqlen
        flip_select=args.flip_select,
        model_name=model_name,
        dataset_name=args.dataset,
        # [新增参数] 传入自定义路径配置
        output_dir=experiment_dir,
        file_suffix=file_suffix,
        tokenizer=tokenizer
    )

    logger.info("所有实验完成。")