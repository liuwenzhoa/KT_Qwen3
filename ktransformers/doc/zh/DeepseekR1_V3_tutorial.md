<!-- omit in toc -->

# 在仅有 24GB VRAM 的桌面设备上运行 GPT-4/o1 级别的本地 VSCode Copilot

- [概述](#概述)
  - [展示环境](#展示环境)
  - [性能测试结果](#性能测试结果)
    - [V0.2.1](#v021)
      - [内存消耗：](#内存消耗)
      - [更新日志](#更新日志)
      - [基准测试结果](#基准测试结果)
    - [V0.2](#v02)
      - [设置](#设置)
      - [内存消耗：](#内存消耗-1)
      - [基准测试结果](#基准测试结果-1)
    - [V0.3-预览版](#v03-预览版)
      - [设置](#设置-1)
      - [内存消耗：](#内存消耗-2)
      - [基准测试结果](#基准测试结果-2)
  - [如何运行](#如何运行)
    - [v0.2.4](#v024)
    - [v0.2.2 & v0.2.3 更长上下文 & FP8 内核](#v022--v023-更长上下文--fp8-内核)
      - [更长上下文](#更长上下文)
      - [FP8 内核](#fp8-内核)
    - [V0.2 & V0.2.1 示例](#v02--v021-示例)
      - [单插槽版本（32核）](#单插槽版本32核)
      - [双插槽版本（64核）](#双插槽版本64核)
    - [V0.3 示例](#v03-示例)
      - [双插槽版本（64核）](#双插槽版本64核-1)
  - [一些说明](#一些说明)
  - [后续计划](#后续计划)
    - [更快](#更快)
    - [更简单](#更简单)
  - [常见问题](#常见问题)
    - [R1 没有思考](#r1-没有思考)
    - [更多常见问题](#更多常见问题)

# 概述

> **2025年2月10日**：支持在单 GPU（24GB VRAM）/多 GPU 以及 382G DRAM 上运行 DeepseekR1 和 V3，提速最高达 3~28 倍。<br>

大家好，我们是 KTransformers 团队（之前因我们在 DeepSeek-V2 上的本地 CPU/GPU 混合推理开源项目而广为人知）。

我们收到了很多关于支持 DeepSeek-R1/V3 的请求——现在我们终于能够交付了！
为了等待时间表示歉意，但我们一直在开发一些真正令人惊叹的东西！

今天，我们很高兴地宣布，我们不仅支持 DeepSeek-R1/V3，如下面的视频所示：

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

</p>

- **[全新!!!] 本地 671B DeepSeek-Coder-V3/R1：** 使用仅 14GB VRAM 和 382GB DRAM 运行其 Q4_K_M 版本。
  - 预填充速度（tokens/s）：
    - KTransformers: 54.21（32 核）→ 74.362（双插槽，2×32 核）→ 255.26（优化的基于 AMX 的 MoE 内核，仅限 V0.3）→ 286.55（选择性使用 6 个专家，仅限 V0.3）
    - 与 llama.cpp 的 2×32 核 10.31 tokens/s 相比，速度提升最高达到 **27.79 倍**。
  - 解码速度（tokens/s）：
    - KTransformers: 8.73（32 核）→ 11.26（双插槽，2×32 核）→ 13.69（选择性使用 6 个专家，仅限 V0.3）
    - 与 llama.cpp 的 2×32 核 4.51 tokens/s 相比，速度提升最高达到 **3.03 倍**。

我们还预览了即将推出的优化，包括 Intel AMX 加速内核和选择性专家激活方法，这将显著提高性能。使用 V0.3-预览版，我们在预填充阶段实现了高达 286 tokens/s 的速度，使其比 llama.cpp 的本地推理快至多 **28 倍**。
二进制分发版现已可用，源代码将尽快发布！可以在[这里](https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl)获取 wheel 包

> **2025年2月15日**：KTransformers V0.2.1：更长的上下文（24GB VRAM 从 4K 增加到 8K）和稍快的速度（+15%）（最高达 16 Tokens/s），更新了[此处](../en/DeepseekR1_V3_tutorial.md)的文档和[在线书籍](https://kvcache-ai.github.io/ktransformers/)。

我们稍微加快了解码和预填充速度。性能提升有限的主要原因在于推理过程仍受到 CPU 计算速度和内存带宽的限制。由 GPU 处理的 MLA 部分所占比例相对较小。

除了速度方面的改进，我们还对文档进行了重大更新以提高可用性，包括：<br>

- 添加了多 GPU 配置教程。
- 整合了安装指南。
- 添加了有关使用 ExpertMarlin 注册额外 GPU 内存的详细教程；

## 展示环境

我们在以下环境中运行最佳性能测试（V0.2）<br>
CPU：Intel (R) Xeon (R) Gold 6454S 1T DRAM（2 个 NUMA 节点）<br>
GPU：4090D 24G VRAM<br>
内存：标准 DDR5-4800 服务器 DRAM（1 TB），每个插槽 8×DDR5-4800

## 性能测试结果

### V0.2.1

- 模型：DeepseekV3-q4km (int4)<br>
- CPU：cpu_model_name: Intel (R) Xeon (R) Gold 6454S，每个插槽 32 核，2 个插槽，2 个 numa 节点
- GPU：4090 24G VRAM
- 我们在充分预热后进行测试

#### 内存消耗：

- 单插槽：382G DRAM，至少 14GB VRAM
- 双插槽：1T DRAM，至少 14GB VRAM

#### 更新日志

- 更长的上下文（24GB VRAM 从 4K 到 8K）和稍快的速度（+15%）：<br>
  集成了来自 sglang 项目的高效 Triton MLA 内核，实现了更长的上下文长度和稍快的预填充/解码速度
- 我们怀疑某些改进来自硬件平台的变化（4090D->4090）

#### 基准测试结果

"6 专家"示例是 V0.3 预览版的一部分


| 提示长度             | hi (2)   | 1K (969)  | 2K (1930) | 4K (3846)                | 8K (7678) |
| -------------------- | -------- | --------- | --------- | ------------------------ | --------- |
| 输出长度             | 10tokens | 300tokens | 300tokens | 300tokens                | 300tokens |
| **6 专家 V0.2.0**    |          |           |           |                          |           |
| 预填充 token/s       | 13       | 105       | 102       | 88                       | CUDA OOM  |
| 解码 token/s         | 16.8     | 15.4      | 14.2      | 13.0                     | CUDA OOM  |
| **6 专家 V0.2.1**    |          |           |           |                          |           |
| 预填充 token/s       | 13       | 111       | 112.5     | 102**(1.16x 加速)**      | 101       |
| 解码 token/s         | 16.8     | 15.9      | 15.4      | 14.9**(1.15x 加速)**     | 13.9      |
| **8 专家 V0.2.1**    |          |           |           |                          |           |
| 预填充 token/s       | 12.2     | 88.2      | 88.5      | 81.9                     | 80        |
| 解码 token/s         | 13.4     | 13.5      | 13.4      | 13.2                     | 12.4      |

### V0.2

#### 设置

- 模型：DeepseekV3-q4km (int4)<br>
- CPU：cpu_model_name: Intel (R) Xeon (R) Gold 6454S，每个插槽 32 核，2 个插槽，2 个 numa 节点
- GPU：4090D 24G VRAM
- 我们在充分预热后进行测试

#### 内存消耗：

- 单插槽：382G DRAM，至少 14GB VRAM
- 双插槽：1T DRAM，至少 14GB VRAM

#### 基准测试结果

"6 专家"示例是 V0.3 预览版的一部分


| 提示长度<br>(500 tokens) | 双插槽 Ktrans (6 专家) | 双插槽 Ktrans (8 专家) | 单插槽 Ktrans (6 专家) | 单插槽 Ktrans (8 专家) | llama.cpp (8 专家) |
| ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ------------------ |
| 预填充 token/s        | 97.32                  | 82.94                  | 65.14                  | 54.21                  | 10.31              |
| 解码 token/s          | 13.69                  | 12.208                 | 10.303                 | 8.73                   | 4.51               |

**最高加速达到解码阶段的 <u>3.03 倍</u>和预填充阶段的 <u>9.44 倍</u>。**

### V0.3-预览版

#### 设置

- 模型：DeepseekV3-BF16（在线量化为 CPU 的 int8 和 GPU 的 int4）
- CPU：cpu_model_name: Intel (R) Xeon (R) Gold 6454S，每个插槽 32 核，2 个插槽，2 个 numa 节点
- GPU：(1~4)x 4090D 24GVRAM（更长的提示需要更多 VRAM）

#### 内存消耗：

- 644GB DRAM，至少 14GB VRAM

#### 基准测试结果


| 提示长度                      | 1K     | 2K     | 4K     | 8K     |
| ----------------------------- | ------ | ------ | ------ | ------ |
| KTrans (8 专家) 预填充 token/s | 185.96 | 255.26 | 252.58 | 195.62 |
| KTrans (6 专家) 预填充 token/s | 203.70 | 286.55 | 271.08 | 207.20 |

**KTrans V0.3 的预填充速度比 KTrans V0.2 快至多 <u>3.45 倍</u>，比 llama.cpp 快至多 <u>27.79 倍</u>。**
**解码速度与 KTrans V0.2（6 专家版本）相同，因此省略**

主要加速来源于：

- Intel AMX 指令集和我们专门设计的缓存友好内存布局
- 基于领域外数据的离线配置文件结果选择更少专家的专家选择策略

*根据我们对 DeepSeekV2、DeepSeekV3 和 DeepSeekR1 的研究，
当我们在推理中略微减少激活专家数量时，
输出质量不会改变。但解码和预填充的速度
得到提升，这很鼓舞人心。所以我们的展示利用了这一发现*

## 如何运行

### v0.2.4
我们提供了一个服务器脚本，在 v0.2.4 版本中支持多并发功能。

```
python ktransformers/server/main.py --model_path /mnt/data/models/DeepSeek-V3 --gguf_path /mnt/data/models/DeepSeek-V3-GGUF/DeepSeek-V3-Q4_K_M/ --cpu_infer 62 --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml --port 10002 --chunk_size 256 --max_new_tokens 1024 --max_batch_size 4 --port 10002 --cache_lens 32768 --backend_type balance_serve
```

它具有以下参数：

- `--chunk_size`：引擎在单次运行中处理的最大标记数。
- `--cache_lens`：调度器分配的 kvcache 的总长度。所有请求共享对应于 32768 个标记的 kvcache 空间，请求完成后占用的空间将被释放。
- `--backend_type`：`balance_serve` 是 v0.2.4 版本中引入的多并发后端引擎。原始的单并发引擎是 `ktransformers`。
- `--max_batch_size`：引擎在单次运行中处理的最大请求数（预填充 + 解码）。（仅由 `balance_serve` 支持）

### v0.2.2 & v0.2.3 更长上下文 & FP8 内核

#### 更长上下文

要使用此功能，请先[安装 flashinfer](https://github.com/flashinfer-ai/flashinfer)。

注意：FlashInfer 中最新的 MLA 内核仍有一些小问题。他们正在主分支上不断修复它们。如果您使用 FlashInfer，请从主源代码安装。

如果您想在预填充阶段使用长上下文（超过 20K），请在预填充阶段启用矩阵吸收 MLA，这将显著减少 kv 缓存的大小。像这样修改 yaml 文件：

```
- match:
    name: "^model\\.layers\\..*\\.self_attn$"
  replace:
    class: ktransformers.operators.attention.KDeepseekV2Attention # 优化的 MLA 实现
    kwargs:
      generate_device: "cuda"
      prefill_device: "cuda"
      absorb_for_prefill: True # 改为 True 以启用长上下文（预填充可能会变慢）。
```

如果 VRAM 仍然不足，尝试减小 `chunk_size` 参数（默认为 8192）以进一步减少块预填充期间的中间结果。

#### FP8 内核

DeepSeek-AI 团队为 DeepSeek-R1/V3 模型提供了 FP8 safetensors。我们通过以下工作实现性能优化：

- **FP8 GPU 内核集成**：KTransformers 中集成了 FP8 线性层加速内核
- **混合量化架构**：
  - 注意力和共享专家模块使用 FP8 精度（提高计算精度）
  - 专家模块保留 GGML 量化（GGUF 格式，驻留在 CPU 中以节省 GPU 内存）

因此，那些追求最佳性能的人可以为 DeepSeek-V3/R1 使用 FP8 线性内核。

详细指南在[这里](../en/fp8_kernel.md)。

### V0.2 & V0.2.1 示例

#### 单插槽版本（32核）

我们的 local_chat 测试命令是：

```shell
numactl -N 1 -m 1 python ./ktransformers/local_chat.py --model_path <你的模型路径> --gguf_path <你的 gguf 路径>  --prompt_file <你的提示文本文件>  --cpu_infer 33 --max_new_tokens 1000
<当你看到聊天提示时，按回车键加载文本提示文件>
```

`<你的模型路径>` 可以是本地的，也可以设置为在线 huggingface 如 deepseek-ai/DeepSeek-V3。如果在线遇到连接问题，尝试使用镜像（hf-mirror.com）<br>
`<你的 gguf 路径>` 也可以是在线的，但由于它很大，我们建议您下载它并将模型量化为您想要的格式（注意它是目录路径）<br>
`--max_new_tokens 1000` 是最大输出标记长度。如果您发现答案被截断，您
可以增加数字以获得更长的答案（但要注意 OOM，增加它会降低生成速率）。

命令 `numactl -N 1 -m 1` 旨在避免 numa 节点之间的数据传输<br>
注意！如果您测试 R1 并且它可能跳过思考。所以您可以添加参数：`--force_think true`。这在[常见问题](#常见问题)部分中有解释

#### 双插槽版本（64核）

确保在安装之前（使用 install.sh 或 `make dev_install`），通过 `export USE_NUMA=1` 设置环境变量 `USE_NUMA=1`（如果已经安装，请使用此环境变量重新安装）。您可以查看[此处](./install.md)的文档获取安装详情。<br>

测试命令：

```shell
# ---对于尚未安装 ktransformers 的用户---
# git clone https://github.com/kvcache-ai/ktransformers.git
# cd ktransformers
# git submodule init
# git submodule update
# export USE_NUMA=1
# make dev_install # 或 sh ./install.sh
# ----------------------------------------------------
python ./ktransformers/local_chat.py --model_path <你的模型路径> --gguf_path <你的 gguf 路径>  --prompt_file <你的提示文本文件>  --cpu_infer 65 --max_new_tokens 1000
<当你看到聊天提示时，按回车键加载文本提示文件>
```

参数的含义相同。但由于我们使用双插槽，我们将 cpu_infer 设置为 65

### V0.3 示例

#### 双插槽版本（64核）

我们的 local_chat 测试命令是：

```shell
wget https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl
pip install ./ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl
python -m ktransformers.local_chat --model_path <你的模型路径> --gguf_path <你的 gguf 路径>  --prompt_file <你的提示文本文件>  --cpu_infer 65 --max_new_tokens 1000
<当你看到聊天提示时，按回车键加载文本提示文件>
```

参数的含义与 V0.2 相同。但由于我们使用双插槽，我们将 cpu_infer 设置为 65

## 一些说明

1. 我们还想进一步利用 Xeon Gold CPU 上的两个 NUMA 节点。
   为避免节点之间的数据传输成本，我们在
   两个节点上"复制"关键矩阵，这会消耗更多内存但加速预填充和解码过程。
   但这种方法在加载权重时需要大量内存并且速度较慢，所以在加载时请耐心等待
   并监控内存使用情况。我们将优化这个巨大的内存开销。敬请期待~ <br>
2. 命令参数 `--cpu_infer 65` 指定要使用多少核心（它可以超过物理数量，
   但并不是越多越好。将其略微调整为低于您的实际核心数量）<br>
3. 为什么使用 CPU/GPU 混合推理？
   DeepSeek 的 MLA 操作符计算密集度非常高。虽然在 CPU 上运行所有内容是可能的，但将繁重的计算卸载到 GPU 会带来巨大的性能提升。
4. 加速来自哪里？

   - 专家卸载：与传统的基于层或 KVCache 的卸载（如 llama.cpp 中所见）不同，我们将专家计算卸载到 CPU，将 MLA/KVCache 卸载到 GPU，这与 DeepSeek 的架构完美结合，实现最佳效率。
   - Intel AMX 优化 – 我们的 AMX 加速内核经过精心调整，运行速度比现有的 llama.cpp 实现快数倍。我们计划在清理后开源这个内核，并考虑向 llama.cpp 上游贡献。
5. 为什么选择 Intel CPU？
   Intel 目前是唯一支持类似 AMX 指令的 CPU 供应商，与仅支持 AVX 的替代品相比，它提供了显著更好的性能。

## 后续计划

### 更快

* FlashInfer (https://github.com/flashinfer-ai/flashinfer) 项目正在发布更高效的融合 MLA 算子，有望进一步提高速度
* vLLM 已经在 DeepSeek-V3 中探索了多令牌预测，我们的路线图上支持它以获得更好的性能
* 我们正在与 Intel 合作增强 AMX 内核（v0.3）并为 Xeon6/MRDIMM 优化

### 更简单

* 官方 Docker 镜像简化安装
* 修复服务器集成以实现 Web API 访问
* 修复本地聊天只接受单行提示的问题（目前 \n 开始生成提示）
* 支持更多量化类型，包括来自 unsloth 的高度要求的动态量化

敬请期待更多更新！

## 常见问题

### R1 没有思考

注意！如果您正在测试 R1 并且它可能跳过思考。所以您可以添加参数：`--force_think true`。详情请参阅[常见问题](../en/FAQ.md)部分 <br>

### 更多常见问题

[查看详情](../en/FAQ.md) 