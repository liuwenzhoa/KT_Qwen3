# 目录
- [Qwen3-235B-A22B部署步骤](#qwen3-235b-a22b部署步骤)
- [下载模型](#下载模型)
- [安装基础依赖](#安装基础依赖)
- [ktransformers安装步骤](#ktransformers安装步骤)
- [构建web服务](#构建web服务)
- [启动KTransformers服务（普通版）](#启动ktransformers服务普通版)
- [测试API](#测试api)
- [安装 open-webui](#安装-open-webui)
- [启动KTransformers服务（AMX版，如果CPU支持且下载了BF16模型权重）](#启动ktransformers服务amx版如果cpu支持且下载了bf16模型权重)
- [性能监控和调优](#性能监控和调优)
- [配置调优建议](#配置调优建议)

# Qwen3-235B-A22B部署步骤
- 官网：https://qwenlm.github.io/zh/blog/qwen3/
- github：https://github.com/QwenLM/Qwen
- 魔搭千问官方的模型权重地址：https://modelscope.cn/models/Qwen/Qwen3-235B-A22B/
- Hugging Face千问官方的模型权重地址：https://huggingface.co/Qwen/Qwen3-235B-A22B
- unsloth提供的GGUF格式的权重地址：https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF
- GGUF-Q4_K_M格式的权重地址：https://huggingface.co/lmstudio-community/Qwen3-235B-A22B-GGUF
- ktransformers官方文档： https://kvcache-ai.github.io/ktransformers/en/injection_tutorial.html
- transformers针对Qwen3的文档地址：https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md
- flash-atte库地址:https://github.com/Dao-AILab/flash-attention/releases
- qwen3推荐微调设置：https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune
- Qwen3 GPU内存需求和吞吐量：https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html
## 系统环境：PyTorch 2.5.1，Python 3.11(ubuntu22.04)，CUDA 12.4
```
GPU：RTX 4090D(24GB) * 4
CPU：60 vCPU Intel(R) Xeon(R) Platinum 8474C
内存：320GB
硬盘系统盘：30 GB
数据盘：180GB SSD
```

# 下载模型
```bash
# 由于实际下载时间可能持续4-6个小时，因此最好使用screen开启持久化会话，避免因为关闭会话导致下载中断。
# 安装screen工具用于持久化会话
sudo apt install screen -y
# 开启持久会话
screen -S kt
# 如果会话断开，可以输入如下命令回到之前的会话：
screen -r kt

# 使用 huggingface_hub 下载模型
pip install huggingface_hub

# 默认情况下，Huggingface会将下载文件保存在/root/.cache文件夹中，在 /root/autodl-tmp 下创建名为 HF_download 文件夹作为huggingface下载文件保存文件夹，这样模型就保存在磁盘而非系统盘
cd /root/autodl-tmp
mkdir -p HF_download Qwen3-235B-A22B-GGUF Qwen3-235B-A22B

#  配置HuggingFace下载目录环境变量，找到root文件夹下的 .bashrc 文件
echo 'export HF_HOME="/root/autodl-tmp/HF_download"' >> ~/.bashrc
source ~/.bashrc

# 在AutoDL开启学术资源加速，（针对HuggingFace等境外资源加速）
# 文档：https://www.autodl.com/docs/network_turbo/
source /etc/network_turbo

# 方式一：使用jupyter下载模型权重
# 进入下载目录：
cd /root/autodl-tmp
# 启动Jupyter
jupyter lab --allow-root
# 开启AutoDL隧道工具可以让本地访问服务器的jupyter，并使用jupyter下载模型权重
# 文档：https://www.autodl.com/docs/ssh_proxy/
# 访问Jupyter Web：http://127.0.0.1:8889/lab?token=d05632341ad9255a31ed844b5d1737604d5d1983b1fa918a
# 这个是Q4_K_M的模型文件:
# lmstudio社区模型下载地址： https://huggingface.co/lmstudio-community/Qwen3-235B-A22B-GGUF
# Q4_K_M 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\install.md
# 页面创建一个ipynb文件运行以下代码：
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = 'lmstudio-community/Qwen3-235B-A22B-GGUF',
    local_dir = '/root/autodl-tmp/Qwen3-235B-A22B-GGUF',
    allow_patterns = ['*Q4_K_M*'],
)

# 方式二：使用python代码下载特定量化模型
# 使用Python代码下载Qwen3-235B-A22B-GGUF-Q4_K_M量化模型（133GB，推荐选择，平衡性能和质量）
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = 'unsloth/Qwen3-235B-A22B-GGUF',
    local_dir = '/root/autodl-tmp/Qwen3-235B-A22B-GGUF',
    allow_patterns = ['*Q4_K_M*'],
)

# 如果需要AMX优化(仅适用于支持AMX的CPU)，可以下载BF16版本
# 注意：BF16版本约470GB，需要确保有足够存储空间
# 参考文档 D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\AMX.md
# 下载地址：https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/tree/main/BF16
# python -c "
# from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id = 'unsloth/Qwen3-235B-A22B-GGUF',
#     local_dir = '/root/autodl-tmp/Qwen3-235B-A22B-GGUF',
#     allow_patterns = ['*BF16*'],
# )
# "

# 模型权重下载完成需要取消学术加速，避免对正常网络造成影响
unset http_proxy && unset https_proxy

# 下载 Qwen3-235B-A22B 原版模型的配置文件
# 根据KTransformer的要求，还需要下载 Qwen3-235B-A22B 原版模型的除了模型权重文件外的其他配置文件，方便进行灵活的专家加载。# 因此我们还需要使用 modelscope 下载 Qwen3-235B-A22B 模型除了模型权重（.safetensors）外的其他全部文件，可以按照如下方式进行下载，地址：https://modelscope.cn/models/Qwen/Qwen3-235B-A22B/
# Qwen3并没有提供模型架构代码，但是KTransformers自行实现了一套代码(Qwen3 MoE(混合专家)模型的核心实现)，用于读取Qwen3的模型配置文件，并进行专家加载。代码地址：D:\deepseek_ubuntu本地部署\ktransformers\ktransformers\models\modeling_qwen3_moe.py

# 安装魔搭
pip install modelscope
# 进入下载目录：
cd /root/autodl-tmp
# 下载Qwen3-235B-A22B模型配置文件（不包含模型权重）
modelscope download --model Qwen/Qwen3-235B-A22B --exclude '*.safetensors' --local_dir /root/autodl-tmp/Qwen3-235B-A22B

# 最终我们是下载了 Qwen3-235B-A22B-GGUF-Q4_K_M 模型权重和 Qwen3-235B-A22B 的模型配置文件，并分别保存在两个文件夹中：
Qwen3-235B-A22B-GGUF-Q4_K_M 模型权重地址：/root/autodl-tmp/Qwen3-235B-A22B-GGUF
Qwen3-235B-A22B 的模型配置文件地址：/root/autodl-tmp/Qwen3-235B-A22B

# 检查GGUF模型文件
ls -lh /root/autodl-tmp/Qwen3-235B-A22B-GGUF/
# 检查模型配置文件
ls -lh /root/autodl-tmp/Qwen3-235B-A22B/
```

# 安装基础依赖
```bash
# 1. 创建独立Python环境,防止对系统环境造成兼容性影响
# 创建Python 3.11环境（KTransformers在3.11下测试更充分）
conda init bash # 初始化conda环境
source ~/.bashrc # 使环境变量生效
# 创建 Python 3.11 环境，kt311 是环境名称
conda create -n kt311 python=3.11 -y
# 激活环境
conda activate kt311

# 2. 安装系统依赖
# 基础编译工具，参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\install.md
sudo apt-get update
sudo apt-get install -y gcc g++ build-essential cmake ninja-build patchelf
# 多并发支持依赖（balance_serve后端需要），Balance Serve是KTransformers框架中的一个组件，主要用于实现多并发支持。
# 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\install.md
sudo apt install -y libtbb-dev libssl-dev libcurl4-openssl-dev libaio1 libaio-dev libgflags-dev zlib1g-dev libfmt-dev libnuma-dev

# 3.升级libstdc++（针对 Ubuntu 22.04 需要添加）
# 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\FAQ.md
# 安装软件源工具
sudo apt-get install -y software-properties-common
# 添加 Ubuntu Toolchain PPA
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
# 升级libstdc++6
sudo apt-get install -y --only-upgrade libstdc++6
# 确保conda环境的libstdc++也是更新到最新版本C++标准库
# 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\install.md
conda install -c conda-forge libstdcxx-ng -y
# 验证libstdc++版本（应包含3.4.32）
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail

# 4. 安装 Python 依赖和基础环境依赖
# 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\balance-serve.md
# ktransformers对于0.3版本的PyTorch使用的是2.7.0版本进行测试的，所以需要安装2.7.0版本的PyTorch以及配套的torchvision和torchaudio
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install packaging ninja cpufeature numpy openai
# 验证 torch 版本
python -c "import torch; print(torch.__version__)"
# 验证 torchvision 版本
python -c "import torchvision; print(torchvision.__version__)"

# 5. 安装Flash-Attention（flash attention的实现比标准注意力机制更高效）
# 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\install.md
# 验证 CUDA 版本
nvcc --version
# 当前环境：CUDA 12.4 Python 3.11 PyTorch 2.7.0

# 进入数据盘目录
cd autodl-tmp/

# 先卸载可能已经安装的版本
pip uninstall -y flash-attn

# Flash-Attention 安装地址：https://github.com/Dao-AILab/flash-attention/releases
# 最合适的 flash_attn 地址：https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
# 这个版本有以下特点：
# cu12: 适用于CUDA 12.x
# torch2.6: 专门为PyTorch 2.7构建（匹配PyTorch 2.7.0）
# cp311: 针对Python 3.11构建
# cxx11abiTRUE: 与PyTorch 2.7.0版本的ABI兼容
# linux_x86_64: 适用于Linux x86_64平台
# 下载适合PyTorch 2.7.0的预编译wheel
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install ./flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
# 验证安装
python -c "import flash_attn; print(flash_attn.__version__)"
```

# 构建web服务
```bash
  # 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\api\server\website.md
  sudo apt-get update -y && sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/nodesource.gpg
  sudo chmod 644 /usr/share/keyrings/nodesource.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_23.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
  # 确保安装新版Node.js
  sudo apt-get update -y
  sudo apt-get install nodejs -y
  
  # 进入网站目录并安装Vue CLI
  cd /root/autodl-tmp/ktransformers/website
  npm install @vue/cli
  
  # 构建前端项目
  npm run build
```

# ktransformers安装步骤
```bash
# 确保conda环境激活
conda activate kt311
# 卸载现有ktransformers
pip uninstall -y ktransformers
# 删除旧仓库文件
rm -rf /root/autodl-tmp/ktransformers
# 开启学术加速
source /etc/network_turbo
# 进入数据盘目录
cd /root/autodl-tmp
# 克隆KTransformers仓库
git clone https://github.com/kvcache-ai/ktransformers.git
# 取消学术加速
unset http_proxy && unset https_proxy
# 进入KTransformers目录
cd ktransformers
# 初始化和更新子模块
git submodule update --init --recursive

# 检查AMX支持（对于使用AMX优化部署方案）
# 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\AMX.md
lscpu | grep -i amx
# 如果输出中包含amx-bf16、amx-int8和amx-tile，说明CPU支持AMX
# 例如：Flags: ... amx-bf16 amx-int8 amx-tile ...
# 如果没有找到AMX相关标志，您的CPU可能不支持AMX，或AMX在BIOS中被禁用

# 根据CPU配置选择安装方式，获取详细的GPU计算能力信息
nvidia-smi --query-gpu=name,compute_cap --format=csv
# 当前RTX 4090 D具有8.9的计算能力
# 8.9代表Hopper架构（如H100 GPU）的计算能力，9.0代表最新的Blackwell架构，+PTX表示包含PTX中间代码，确保向后兼容性
(kt311) root@autodl-container-b4bd4c9525-4d58915a:~# nvidia-smi --query-gpu=name,compute_cap --format=csv
name, compute_cap
NVIDIA GeForce RTX 4090 D, 8.9
NVIDIA GeForce RTX 4090 D, 8.9
NVIDIA GeForce RTX 4090 D, 8.9
NVIDIA GeForce RTX 4090 D, 8.9


# 设置环境变量，在编译KTransformers时针对特定的NVIDIA GPU架构进行优化
# 这个设置确保KTransformers编译出来的代码能够最大限度地发挥GPU硬件性能
# 参考：D:\deepseek_ubuntu本地部署\ktransformers\Dockerfile
export TORCH_CUDA_ARCH_LIST="8.9;9.0+PTX"

# 查看系统NUMA节点配置，确定CPU拓扑结构
lscpu | grep NUMA
# 输出结果：
# 系统有2个NUMA节点，这通常对应于2个物理CPU插槽
# 第一个CPU插槽控制着逻辑CPU核心0-47和96-143
# 第二个CPU插槽控制着逻辑CPU核心48-95和144-191
NUMA node(s):                       2
NUMA node0 CPU(s):                  0-47,96-143
NUMA node1 CPU(s):                  48-95,144-191

# 安装numactl工具，在启用USE_NUMA=1和USE_BALANCE_SERVE=1后，系统会根据NUMA信息优化任务调度
# KTransformers在启用USE_NUMA=1时需要numactl工具及其库来访问和管理NUMA节点信息
# 如果没有这个工具，系统无法获取正确的NUMA拓扑，导致NUMA优化无效
apt-get install -y numactl
# 检查和确认系统的NUMA(非统一内存访问)架构信息
numactl --hardware

# NUMA优化对于多插槽服务器至关重要
# NUMA（Non-Uniform Memory Access）是一种内存架构，在多插槽服务器中，每个CPU插槽都有自己的本地内存
# 当一个CPU核心访问另一个CPU插槽的内存时，会产生额外的延迟，这称为NUMA惩罚
# KTransformers的NUMA优化可以确保计算任务尽可能在本地内存上执行，减少跨NUMA节点访问
# NUMA优化配置选择指南：
# 1. 对于双插槽或多插槽服务器：启用NUMA支持（USE_NUMA=1）
#    - 这将使KTransformers根据CPU拓扑结构优化线程分配，使其合理分配计算任务
#    - 对于MoE（混合专家）模型如DeepSeek-V3 或 Qwen3-235B，NUMA优化可以显著提高推理性能
#    - 特别是在处理多并发请求时，减少跨NUMA节点访问，可以减少内存访问延迟
# 2. 对于单插槽服务器：不需要启用NUMA支持
#    - 单插槽服务器没有NUMA架构，所有CPU核心访问内存的延迟相同
#    - 在这种情况下，启用NUMA支持不会带来性能提升，反而可能增加不必要的开销

# 1. 对于双槽服务器(60+ vCPU)，启用NUMA支持和多并发功能
# 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\balance-serve.md
# 在命令行中设置环境变量：
export USE_NUMA=1 # 启用NUMA支持，优化多核CPU性能
export USE_BALANCE_SERVE=1 # 启用多并发支持，实现高效请求处理
export KTRANSFORMERS_FORCE_BUILD=TRUE # 强制重新编译，确保使用最新的代码和配置
# 打印环境变量确认设置成功
echo $USE_NUMA
echo $USE_BALANCE_SERVE
echo $KTRANSFORMERS_FORCE_BUILD
# 在同一命令行中设置环境变量并执行安装脚本：
USE_NUMA=1 USE_BALANCE_SERVE=1 KTRANSFORMERS_FORCE_BUILD=TRUE bash ./install.sh
# 安装完成后验证NUMA配置是否生效
# 如果返回True，说明NUMA支持已成功启用
# 如果返回False，可能是因为在容器环境中NUMA支持受限，或者libnuma-dev未正确安装
# python -c "import ktransformers; print('NUMA支持：', hasattr(ktransformers, 'numa_available') and ktransformers.numa_available())"

# 2. 对于不支持AMX的CPU或单NUMA节点系统,正确做法（在同一命令行中设置环境变量并执行安装脚本）：
export USE_BALANCE_SERVE=1 # 启用多并发支持，实现高效请求处理
USE_BALANCE_SERVE=1 KTRANSFORMERS_FORCE_BUILD=TRUE bash ./install.sh

# 查看ktransformers安装版本
pip show ktransformers
# 验证 ktransformers 安装是否成功
python -c "import ktransformers; print(ktransformers.__version__)"
# 验证 NUMA 支持（如果启用了USE_NUMA=1），查看进程的NUMA映射
test -f /proc/self/numa_maps && echo "NUMA支持已启用" || echo "NUMA未启用"
```

# 启动KTransformers服务（普通版）
```bash
# 确保处于ktransformers目录下
cd /root/autodl-tmp/ktransformers

# 确保激活Python环境
conda activate kt311

# 确保在命令行中设置环境变量：
export USE_NUMA=1 # 启用NUMA支持，优化多核CPU性能
export USE_BALANCE_SERVE=1 # 启用多并发支持，实现高效请求处理
export KTRANSFORMERS_FORCE_BUILD=TRUE # 强制重新编译，确保使用最新的代码和配置

# 验证NUMA支持（如果启用了USE_NUMA=1），查看进程的NUMA映射
test -f /proc/self/numa_maps && echo "NUMA支持已启用" || echo "NUMA未启用"

# 启动KTransformers服务，使用Q4_K_M模型权重和配置文件

# 简单启动：
# 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\AMX.md
python ktransformers/server/main.py --architectures Qwen3MoeForCausalLM --model_path /root/autodl-tmp/Qwen3-235B-A22B --gguf_path /root/autodl-tmp/Qwen3-235B-A22B-GGUF --optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve.yaml --backend_type balance_serve

# 解释：
1. 通过 --model_path 和 --architectures Qwen3MoeForCausalLM 参数，系统会知道需要加载 Qwen3 MoE 架构的模型
2. KTransformers框架内部会根据 architectures 参数查找并加载对应的模型类，这个过程中会自动导入这两个文件：
  - ktransformers/ktransformers/models/modeling_qwen3_moe.py
  - ktransformers/ktransformers/models/configuration_qwen3_moe.py
  - configuration_qwen3_moe.py：定义了Qwen3MoeConfig类，包含模型的所有超参数
  - modeling_qwen3_moe.py：实现了模型的核心架构，包括Qwen3MoeForCausalLM类
3. 框架在初始化时，会先从model_path加载模型原版配置文件（config.json），然后用 Qwen3MoeConfig 解析这些配置
4. 然后使用这些配置实例化Qwen3MoeForCausalLM类来创建模型
5. 我们使用--optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve.yaml 指定了优化配置
6. 这个YAML文件专门为Qwen3 MoE模型进行了优化配置。它通过一系列优化规则，定义了模型各组件的执行位置（CPU/GPU），它会引用 modeling_qwen3_moe.py 中定义的类和方法，将实现替换为优化版本。
7. 实际执行流程：
  → 解析命令行参数 
  → 加载模型配置(config.json)
  → 使用configuration_qwen3_moe.py解析配置
  → 使用modeling_qwen3_moe.py创建模型结构
  → 加载GGUF格式的权重
  → 应用Qwen3Moe-serve.yaml中的优化规则
  → 启动服务

# 完整启动并启用web服务：
# 参考文档：D:\deepseek_ubuntu本地部署\ktransformers\doc\zh\balance-serve.md
python ktransformers/server/main.py \
  --architectures Qwen3MoeForCausalLM \
  --model_path /root/autodl-tmp/Qwen3-235B-A22B \
  --gguf_path /root/autodl-tmp/Qwen3-235B-A22B-GGUF \
  --optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve.yaml \
  --port 10002 \
  --cpu_infer 54 \
  --chunk_size 256 \
  --max_new_tokens 4096 \
  --max_batch_size 4 \
  --cache_lens 32768 \
  --backend_type balance_serve \
  --web True

  --architectures：指定模型架构
  --model_path：指向您在/root/autodl-tmp/Qwen3-235B-A22B的模型配置文件路径
  --gguf_path：指向您在/root/autodl-tmp/Qwen3-235B-A22B-GGUF的GGUF权重文件
  --optimize_config_path：使用Qwen3Moe-serve.yaml配置文件
  --port：服务端口，默认10002
  --cpu_infer 54：设为54,确保内存不会溢出导致崩溃
  --chunk_size：引擎在单次运行中一次性处理的最大token数
  --max_new_tokens：每次请求生成的最大新token数量
  --max_batch_size：引擎在单次运行中能够处理的最大并发请求数
  --cache_lens: 调度器分配的KV缓存总长度，所有并发请求共享此空间，更大的缓存长度支持处理更长的上下文或更多的并发请求，确保cache_lens > max_batch_size * max_new_tokens才能支持满载并发
  --backend_type balance_serve：启用多并发支持，实现高效请求处理
  --web True：启用Web界面，可通过浏览器访问

# 访问web服务：http://127.0.0.1:10002/web/index.html#/chat
```

# 测试API
```bash
# 获取模型列表
curl http://localhost:10002/v1/models

# 简单测试
# 微调参数参考：https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune
curl -X POST http://localhost:10002/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，请介绍一下自己 /no_think"}
    ],
    "model": "Qwen3-235B-A22B",
    "Min_P": 0.0,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'

curl -X POST http://localhost:10002/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你是谁"}
    ],
    "model": "Qwen3-235B-A22B",
    "Min_P": 0.0,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
}'
```

# 安装 open-webui
```bash
# 确保激活Python环境
conda activate kt311

# 进入数据盘目录
cd /root/autodl-tmp

# 安装open-webui
pip install open-webui
# 设置离线模式
export HF_HUB_OFFLINE=1
# 启动open-webui
open-webui serve
# 访问open-webui：http://127.0.0.1:8080/

# 卸载open-webui
pip show open-webui
pip uninstall open-webui -y
# 如果上面的不管用，就根据依赖安装地址手动卸载 rm -rf **/open_webui
# 检查是否有残留数据库
find ~ -name "*.db" | grep -i webui
```

# 启动KTransformers服务（AMX版，如果CPU支持且下载了BF16模型权重）
```bash
# 确保处于ktransformers目录下
cd /root/autodl-tmp/ktransformers

# 激活Python环境
conda activate kt311

# 检查CPU是否支持AMX指令集
lscpu | grep -i amx
# 如果输出包含amx-bf16, amx-int8, amx-tile说明CPU支持AMX指令集

# 启动KTransformers服务
# 注意：目前AMX版本只能读取BF16 GGUF文件
python ktransformers/server/main.py \
  --architectures Qwen3MoeForCausalLM \
  --model_path /root/autodl-tmp/Qwen3-235B-A22B \
  --gguf_path /root/autodl-tmp/Qwen3-235B-A22B-GGUF-BF16 \
  --optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve-amx.yaml \
  --backend_type balance_serve \
  --port 10002 \
  --cpu_infer 56 \
  --chunk_size 256 \
  --max_new_tokens 4096 \
  --max_batch_size 4 \
  --cache_lens 16384 \
  --web True
```

# 性能监控和调优
```bash
# 1. 内存监控
# 监控系统内存使用情况
watch -n 2 free -h
# 如果内存不足，可以调整以下参数：
# --chunk_size 128  # 降低单次处理的标记数
# --max_batch_size 4  # 降低批处理大小
# --absorb_for_prefill True  # 启用矩阵吸收减少内存使用

# 2. GPU内存监控
watch -n 2 nvidia-smi
# 5个32GB vGPU足够运行DeepSeek-V3，如果GPU内存不足，考虑：
# 1. 使用更低精度量化模型(Q3_K_M或Q2_K)
# 2. 减小batch_size和cache_lens参数

# 3.CPU使用监控
# 监控CPU使用情况
htop
# 优化参数：
# --cpu_infer: 可在50-58之间调整，找到最佳性能点
# 如果观察到NUMA节点间不平衡，可以尝试numactl绑定

# 4.多并发优化
# --max_batch_size: 对于5个32GB GPU，可以设置为6-8
# --chunk_size: 256是平衡速度和内存的好选择
# --cache_lens: 确保 cache_lens > max_batch_size * max_new_tokens
# 确保USE_BALANCE_SERVE=1已启用，以支持高效多并发

# 5. libstdc++问题排查：
# 如果出现version 'GLIBCXX_X.X.XX' not found错误，需重新执行libstdc++升级步骤
# 确保conda环境和系统环境都有最新版本
```

# 配置调优建议

以下是一些可以尝试的配置组合，用于不同场景优化：

- **高吞吐量配置**（适合多用户并发）：
  ```bash
  --max_batch_size 8 --chunk_size 512 --cache_lens 32768 --max_new_tokens 2048
  ```

- **长上下文配置**（适合单用户，需要长文档处理）：
  ```bash
  --max_batch_size 2 --chunk_size 256 --cache_lens 32768 --max_new_tokens 8192
  ```

- **平衡配置**（通用场景）：
  ```bash
  --max_batch_size 4 --chunk_size 256 --cache_lens 32768 --max_new_tokens 4096
  ```