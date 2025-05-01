# ktransformers的Balance Serve后端（多并发）

## KTransformers v0.2.4 发布说明

我们很高兴宣布期待已久的**KTransformers v0.2.4**正式发布！
在这个版本中，我们通过对整个架构的重大重构，更新了超过10,000行代码，为社区添加了备受期待的**多并发**支持。
借鉴sglang优秀架构的灵感，我们在C++中实现了高性能异步并发调度，包括连续批处理、分块预填充等功能。得益于并发场景下的GPU共享，整体吞吐量也得到了一定程度的提升。以下是演示：

https://github.com/user-attachments/assets/faa3bda2-928b-45a7-b44f-21e12ec84b8a

</p>

### 🚀 主要更新

1. 多并发支持
   - 增加了处理多个并发推理请求的能力。支持同时接收和执行多个任务。
   - 我们基于高性能和高度灵活的算子库[flashinfer](https://github.com/flashinfer-ai/flashinfer/)实现了[custom_flashinfer](https://github.com/kvcache-ai/custom_flashinfer/tree/fix-precision-mla-merge-main)，并实现了可变批量大小的CUDA Graph，在减少内存和填充开销的同时进一步增强了灵活性。
   - 在我们的基准测试中，4路并发下整体吞吐量提高了约130%。
   - 在英特尔的支持下，我们在最新的Xeon6 + MRDIMM-8800平台上测试了KTransformers v0.2.4。通过增加并发度，总输出吞吐量从17 tokens/s增加到40 tokens/s。我们观察到瓶颈现在已经转移到GPU。使用比4090D更高端的GPU可能会进一步提高性能。
2. 引擎架构优化
   ![image](https://github.com/user-attachments/assets/f5f001fa-dca7-4377-a01a-32192902aa47)
   受sglang调度框架的启发，我们通过更新11,000行代码，用更清晰的三层架构重构了KTransformers，现在支持完全多并发：
   - 服务器：处理用户请求并提供兼容OpenAI的API。
   - 推理引擎：执行模型推理并支持分块预填充。
   - 调度器：管理任务调度和请求编排。通过先来先服务（FCFS）方式将排队的请求组织成批次并发送到推理引擎，支持连续批处理。
3. 项目结构重组
   所有C/C++代码现在都集中在/csrc目录下。
4. 参数调整
   移除了一些遗留和废弃的启动参数，使配置体验更加简洁。
   我们计划在未来的版本中提供完整的参数列表和详细的文档，以便灵活配置和调试。

### 📚 升级说明

- 由于参数变化，建议已安装先前版本的用户删除~/.ktransformers目录并重新初始化。
- 要启用多并发，请参考最新文档中的配置示例。

### 更新内容

实现**custom_flashinfer** @Atream @ovowei @qiyuxinlin
基于**FlashInfer**实现**balance_serve**引擎 @qiyuxinlin @ovowei
在C++中实现**连续批处理**调度器 @ErvinXie
发布：提升版本到v0.2.4 由@Atream @Azure-Tang @ErvinXie @qiyuxinlin @ovowei @KMSorSMS @SkqLiao

## 下载Docker镜像测试v0.2.4
访问[链接](https://hub.docker.com/r/approachingai/ktransformers/tags)拉取镜像，以`v0.2.4-AVX512`为例。

```bash
docker pull approachingai/ktransformers:v0.2.4-AVX512
docker run -it --gpus all --privileged --shm-size 64g --name ktrans --network=host -v /mnt:/mnt approachingai/ktransformers:v0.2.4-AVX512 /bin/bash
# 打开新终端
docker exec -it ktrans bash
```

## 安装指南

⚠️ 请注意，安装此项目将替换您环境中的flashinfer。强烈建议创建新的conda环境！！！

⚠️ 请注意，安装此项目将替换您环境中的flashinfer。强烈建议创建新的conda环境！！！

⚠️ 请注意，安装此项目将替换您环境中的flashinfer。强烈建议创建新的conda环境！！！

### 1. 设置Conda环境

我们建议使用Miniconda3/Anaconda3进行环境管理：

```bash
# 下载Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 创建环境
conda create --name ktransformers python=3.11
conda activate ktransformers

# 安装所需库
conda install -c conda-forge libstdcxx-ng

# 验证GLIBCXX版本（应包含3.4.32）
strings ~/anaconda3/envs/ktransformers/lib/libstdc++.so.6 | grep GLIBCXX
```

> **注意：** 如果您的安装目录与`~/anaconda3`不同，请调整Anaconda路径

### 2. 安装依赖

```bash
sudo apt install libtbb-dev libssl-dev libcurl4-openssl-dev libaio1 libaio-dev libfmt-dev libgflags-dev zlib1g-dev patchelf
pip3 install packaging ninja cpufeature numpy openai
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

```

### 3. 构建ktransformers

```bash
# 克隆仓库
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive


# 安装单NUMA依赖
USE_BALANCE_SERVE=1  bash ./install.sh
# 对于拥有两个CPU和1T RAM（双NUMA）的用户：
USE_BALANCE_SERVE=1 USE_NUMA=1 bash ./install.sh
```

## 运行DeepSeek-R1-Q4KM模型

### 1. 在24GB VRAM GPU上运行

使用我们为受限VRAM优化的配置：

```bash
python ktransformers/server/main.py \
  --port 10002 \
  --model_path <safetensor配置路径> \
  --gguf_path <gguf文件路径> \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --max_batch_size 4 \
  --backend_type balance_serve \
  --force_think # 对R1有用
```

它具有以下参数：

- `--max_new_tokens`: 每个请求生成的最大令牌数。
- `--cache_lens`: 调度器分配的kvcache总长度。所有请求共享kvcache空间。
- `--max_batch_size`: 引擎在单次运行中处理的最大请求数（预填充+解码）。（仅被`balance_serve`支持）
- `--chunk_size`: 引擎在单次运行中处理的最大令牌数。
  对应32768个令牌，占用的空间将在请求完成后释放。
- `--backend_type`: `balance_serve`是v0.2.4版本中引入的多并发后端引擎。原始的单并发引擎是`ktransformers`。
- `--model_path`: safetensor配置路径（仅需要配置，不需要模型safetensors）。
  请注意，自`ver 0.2.4`起，`${model_path}`目录名称的最后一段**必须**是包含模型配置文件的本地目录。目前不支持Hugging Face链接（例如deepseek-ai/DeepSeek-R1）。
- `--force_think`: 强制响应`DeepSeek R1`的推理标签。

`max_batch_size`、`cache_lens`和`max_new_tokens`之间的关系应满足：
`cache_lens > max_batch_size * max_new_tokens`，否则并发度将降低。

### 2. 访问服务器

```
curl -X POST http://localhost:10002/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "model": "DeepSeek-R1",
    "temperature": 0.3,
    "top_p": 1.0,
    "stream": true
  }'
``` 