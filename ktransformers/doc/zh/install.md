<!-- omit in toc -->

# 如何运行 DeepSeek-R1

- [如何运行 DeepSeek-R1](#如何运行-deepseek-r1)
  - [准备工作](#准备工作)
  - [安装](#安装)
    - [注意事项](#注意事项)
    - [支持的模型列表](#支持的模型列表)
    - [支持的量化格式](#支持的量化格式)

本文档将向您展示如何在本地机器上安装和运行 KTransformers。目前有两个版本：

* V0.2 是当前的主分支。
* V0.3 是预览版本，目前仅提供二进制分发。
* 如需复现我们的 DeepSeek-R1/V3 结果，安装完成后请参考 [Deepseek-R1/V3 教程](../en/DeepseekR1_V3_tutorial.md) 获取更详细的设置。

## 准备工作

一些准备工作：

- CUDA 12.1 及以上版本，如果您尚未安装，可以从[这里](https://developer.nvidia.com/cuda-downloads)下载安装。

  ```sh
  # 将 CUDA 添加到 PATH
  if [ -d "/usr/local/cuda/bin" ]; then
      export PATH=$PATH:/usr/local/cuda/bin
  fi

  if [ -d "/usr/local/cuda/lib64" ]; then
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
      # 或者您可以将其添加到 /etc/ld.so.conf 并以 root 身份运行 ldconfig：
      # echo "/usr/local/cuda-12.x/lib64" | sudo tee -a /etc/ld.so.conf
      # sudo ldconfig
  fi

  if [ -d "/usr/local/cuda" ]; then
      export CUDA_PATH=$CUDA_PATH:/usr/local/cuda
  fi
  ```
- Linux-x86_64 环境，gcc, g++ >= 11 和 cmake >= 3.25（以 Ubuntu 为例）
- **注意**：Ubuntu 22.04 LTS 或更高版本中的默认 CMake 版本可能不支持较新的 CUDA 语言方言（例如 CUDA 20）。这可能导致如下错误：Target "cmTC_xxxxxx" requires the language dialect "CUDA20", but CMake does not know the compile flags to use to enable it. 要解决此问题，请安装较新的 CMake 版本，例如通过添加 Kitware APT 仓库。

  ```sh
  sudo apt-get update 
  sudo apt-get install build-essential cmake ninja-build patchelf
  ```
- 我们建议使用 [Miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) 或 [Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh) 创建 Python=3.11 的虚拟环境来运行我们的程序。假设您的 Anaconda 安装目录是 `~/anaconda3`，您应确保 Anaconda 使用的 GNU C++ 标准库版本标识符包含 `GLIBCXX-3.4.32`

  ```sh
  conda create --name ktransformers python=3.11
  conda activate ktransformers # 您可能需要先运行 'conda init' 并重新打开 shell

  conda install -c conda-forge libstdcxx-ng # Anaconda 提供了一个名为 `libstdcxx-ng` 的包，其中包含更新版本的 `libstdc++`，可以通过 `conda-forge` 安装。

  strings ~/anaconda3/envs/ktransformers/lib/libstdc++.so.6 | grep GLIBCXX
  ```
- 确保已安装 PyTorch、packaging 和 ninja。您也可以[安装以前版本的 PyTorch](https://pytorch.org/get-started/previous-versions/)

  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  pip3 install packaging ninja cpufeature numpy
  ```
- 同时，您应该从 https://github.com/Dao-AILab/flash-attention/releases 下载并安装相应版本的 flash-attention。

## 安装

### 注意事项

如果您想使用 numa 支持，不仅需要设置 USE_NUMA=1，还需要确保已安装了 libnuma-dev（`sudo apt-get install libnuma-dev` 可能对您有所帮助）。

[可选] 如果您想使用多并发版本，请安装以下依赖项。

```
sudo apt install libtbb-dev libssl-dev libcurl4-openssl-dev libaio1 libaio-dev libgflags-dev zlib1g-dev libfmt-dev
```

<!-- 1. ~~使用 Docker 镜像，请参阅 [Docker 文档](./doc/en/Docker.md)~~
   
   >我们正在开发最新的 docker 镜像，请稍等片刻。

2. ~~您可以使用 Pypi 进行安装（针对 linux）：~~
    > 我们正在开发最新的 pypi 包，请稍等片刻。
   
   ```
   pip install ktransformers --no-build-isolation
   ```
   
   对于 Windows，我们准备了预编译的 whl 包 [ktransformers-0.2.0+cu125torch24avx2-cp312-cp312-win_amd64.whl](https://github.com/kvcache-ai/ktransformers/releases/download/v0.2.0/ktransformers-0.2.0+cu125torch24avx2-cp312-cp312-win_amd64.whl)，该包需要 cuda-12.5、torch-2.4、python-3.11，更多预编译包正在制作中。  -->

下载源代码并编译：

- 初始化源代码

  ```sh
  git clone https://github.com/kvcache-ai/ktransformers.git
  cd ktransformers
  git submodule update --init --recursive
  ```
- [可选] 如果您想运行带有网站的版本，请在执行 ``bash install.sh`` 之前[编译网站](../en/api/server/website.md)
- 对于 Linux

  - 简单安装：

    ```shell
    bash install.sh
    ```
  - 对于拥有双 CPU 和 1T RAM 的系统：

    ```shell
    # 确保您的系统有双插槽 CPU 和比模型大小大一倍的 RAM（例如，对于 512G 模型需要 1T RAM）
     apt install libnuma-dev
     export USE_NUMA=1
     bash install.sh # 或 #make dev_install
    ```
  - 拥有 500G RAM 的多并发系统：

    ```shell
    USE_BALANCE_SERVE=1 bash ./install.sh
    ```
  - 拥有双 CPU 和 1T RAM 的多并发系统：

    ```shell
    USE_BALANCE_SERVE=1 USE_NUMA=1 bash ./install.sh
    ```
- 对于 Windows（Windows 原生版本暂时不可用，请尝试 WSL）

  ```shell
  install.bat
  ```

* 如果您是开发者，可以使用 makefile 来编译和格式化代码。<br> makefile 的详细用法在[这里](../en/makefile_usage.md)

<h3>本地聊天</h3>
我们提供了一个简单的命令行本地聊天 Python 脚本，您可以运行它进行测试。

> 注意：这是一个非常简单的测试工具，仅支持单轮聊天，没有任何关于上次输入的记忆。如果您想尝试模型的完整功能，请前往 [RESTful API 和 Web UI](#id_666)。

<h4>运行示例</h4>

```shell
# 从克隆的仓库根目录开始！
# 从克隆的仓库根目录开始！！
# 从克隆的仓库根目录开始！！！ 

# 从 huggingface 下载 mzwing/DeepSeek-V2-Lite-Chat-GGUF
mkdir DeepSeek-V2-Lite-Chat-GGUF
cd DeepSeek-V2-Lite-Chat-GGUF

wget https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/main/DeepSeek-V2-Lite.Q4_K_M.gguf -O DeepSeek-V2-Lite-Chat.Q4_K_M.gguf

cd .. # 回到仓库根目录

# 启动本地聊天
python -m ktransformers.local_chat --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF

# 如果看到 "OSError: We couldn't connect to 'https://huggingface.co' to load this file"，请尝试：
# GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
# python  ktransformers.local_chat --model_path ./DeepSeek-V2-Lite --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF
```

它具有以下参数：

- `--model_path`（必需）：模型名称（例如 "deepseek-ai/DeepSeek-V2-Lite-Chat"，它将自动从 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) 下载配置）。或者如果您已经有了本地文件，可以直接使用该路径初始化模型。

  > 注意：目录中不需要 <strong>.safetensors</strong> 文件。我们只需要配置文件来构建模型和分词器。
  >
- `--gguf_path`（必需）：包含 GGUF 文件的目录路径，可以从 [Hugging Face](https://huggingface.co/mzwing/DeepSeek-V2-Lite-Chat-GGUF/tree/main) 下载。请注意，该目录应该只包含当前模型的 GGUF 文件，这意味着您需要为每个模型使用单独的目录。
- `--optimize_config_path`（除 Qwen2Moe 和 DeepSeek-V2 外都必需）：包含优化规则的 YAML 文件路径。[ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) 目录中预先编写了两个规则文件，用于优化 DeepSeek-V2 和 Qwen2-57B-A14 这两个最先进的 MoE 模型。
- `--max_new_tokens`：整数（默认=1000）。生成的最大新标记数。
- `--cpu_infer`：整数（默认=10）。用于推理的 CPU 数量。理想情况下应设置为（总核心数 - 2）。

<h3>启动服务器</h3>
我们提供了一个服务器脚本，在 v0.2.4 版本中支持多并发功能。

```
python ktransformers/server/main.py --model_path /mnt/data/models/DeepSeek-V3 --gguf_path /mnt/data/models/DeepSeek-V3-GGUF/DeepSeek-V3-Q4_K_M/ --cpu_infer 62 --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml --port 10002 --chunk_size 256 --max_new_tokens 1024 --max_batch_size 4 --port 10002 --cache_lens 32768 --backend_type balance_serve
```

它具有以下参数：

- `--chunk_size`：引擎在单次运行中处理的最大标记数。
- `--cache_lens`：调度器分配的 kvcache 的总长度。所有请求共享对应于 32768 个标记的 kvcache 空间，请求完成后占用的空间将被释放。
- `--backend_type`：`balance_serve` 是 v0.2.4 版本中引入的多并发后端引擎。原始的单并发引擎是 `ktransformers`。
- `--max_batch_size`：引擎在单次运行中处理的最大请求数（预填充 + 解码）。（仅由 `balance_serve` 支持）

<details>
<summary>支持的模型/量化</summary>

### 支持的模型列表


| ✅**支持的模型** | ❌**已弃用的模型**    |
| ---------------- | ---------------------- |
| DeepSeek-R1      | ~~InternLM2.5-7B-Chat-1M~~ |
| DeepSeek-V3      |                        |
| DeepSeek-V2      |                        |
| DeepSeek-V2.5    |                        |
| Qwen2-57B        |                        |
| DeepSeek-V2-Lite |                        |
| Mixtral-8x7B     |                        |
| Mixtral-8x22B    |                        |

### 支持的量化格式


| ✅**支持的格式** | ❌**已弃用的格式** |
| --------------- | ------------------ |
| IQ1_S           | ~~IQ2_XXS~~        |
| IQ2_XXS         |                    |
| Q2_K_L          |                    |
| Q2_K_XS         |                    |
| Q3_K_M          |                    |
| Q4_K_M          |                    |
| Q5_K_M          |                    |
| Q6_K            |                    |
| Q8_0            |                    |

</details>

<details>
<summary>推荐的模型</summary>


| 模型名称                       | 模型大小 | VRAM  | 最小 DRAM      | 推荐 DRAM       |
| ------------------------------ | -------- | ----- | -------------- | --------------- |
| DeepSeek-R1-q4_k_m             | 377G     | 14G   | 382G           | 512G            |
| DeepSeek-V3-q4_k_m             | 377G     | 14G   | 382G           | 512G            |
| DeepSeek-V2-q4_k_m             | 133G     | 11G   | 136G           | 192G            |
| DeepSeek-V2.5-q4_k_m           | 133G     | 11G   | 136G           | 192G            |
| DeepSeek-V2.5-IQ4_XS           | 117G     | 10G   | 107G           | 128G            |
| Qwen2-57B-A14B-Instruct-q4_k_m | 33G      | 8G    | 34G            | 64G             |
| DeepSeek-V2-Lite-q4_k_m        | 9.7G     | 3G    | 13G            | 16G             |
| Mixtral-8x7B-q4_k_m            | 25G      | 1.6G  | 51G            | 64G             |
| Mixtral-8x22B-q4_k_m           | 80G      | 4G    | 86.1G          | 96G             |
| InternLM2.5-7B-Chat-1M         | 15.5G    | 15.5G | 8G(32K context) | 150G (1M context) |

更多模型将很快推出。请告诉我们您最感兴趣的模型。

请注意，使用 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/LICENSE) 和 [QWen](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/LICENSE) 时，您需要遵守它们各自的模型许可。

</details>

<details>
  <summary>点击查看如何运行其他示例</summary>

* Qwen2-57B

  ```sh
  pip install flash_attn # 为 Qwen2 安装

  mkdir Qwen2-57B-GGUF && cd Qwen2-57B-GGUF

  wget https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF/resolve/main/qwen2-57b-a14b-instruct-q4_k_m.gguf?download=true -O qwen2-57b-a14b-instruct-q4_k_m.gguf

  cd ..

  python -m ktransformers.local_chat --model_name Qwen/Qwen2-57B-A14B-Instruct --gguf_path ./Qwen2-57B-GGUF

  # 如果看到 "OSError: We couldn't connect to 'https://huggingface.co' to load this file"，请尝试：
  # GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct
  # python  ktransformers/local_chat.py --model_path ./Qwen2-57B-A14B-Instruct --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF
  ```
* Deepseek-V2

  ```sh
  mkdir DeepSeek-V2-Chat-0628-GGUF && cd DeepSeek-V2-Chat-0628-GGUF
  # 下载权重
  wget https://huggingface.co/bartowski/DeepSeek-V2-Chat-0628-GGUF/resolve/main/DeepSeek-V2-Chat-0628-Q4_K_M/DeepSeek-V2-Chat-0628-Q4_K_M-00001-of-00004.gguf -o DeepSeek-V2-Chat-0628-Q4_K_M-00001-of-00004.gguf
  wget https://huggingface.co/bartowski/DeepSeek-V2-Chat-0628-GGUF/resolve/main/DeepSeek-V2-Chat-0628-Q4_K_M/DeepSeek-V2-Chat-0628-Q4_K_M-00002-of-00004.gguf -o DeepSeek-V2-Chat-0628-Q4_K_M-00002-of-00004.gguf
  wget https://huggingface.co/bartowski/DeepSeek-V2-Chat-0628-GGUF/resolve/main/DeepSeek-V2-Chat-0628-Q4_K_M/DeepSeek-V2-Chat-0628-Q4_K_M-00003-of-00004.gguf -o DeepSeek-V2-Chat-0628-Q4_K_M-00003-of-00004.gguf
  wget https://huggingface.co/bartowski/DeepSeek-V2-Chat-0628-GGUF/resolve/main/DeepSeek-V2-Chat-0628-Q4_K_M/DeepSeek-V2-Chat-0628-Q4_K_M-00004-of-00004.gguf -o DeepSeek-V2-Chat-0628-Q4_K_M-00004-of-00004.gguf

  cd ..

  python -m ktransformers.local_chat --model_name deepseek-ai/DeepSeek-V2-Chat-0628 --gguf_path ./DeepSeek-V2-Chat-0628-GGUF

  # 如果看到 "OSError: We couldn't connect to 'https://huggingface.co' to load this file"，请尝试：

  # GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628

  # python -m ktransformers.local_chat --model_path ./DeepSeek-V2-Chat-0628 --gguf_path ./DeepSeek-V2-Chat-0628-GGUF
  ```


| 模型名称          | 权重下载链接                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| Qwen2-57B         | [Qwen2-57B-A14B-gguf-Q4K-M](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF/tree/main)                   |
| DeepseekV2-coder  | [DeepSeek-Coder-V2-Instruct-gguf-Q4K-M](https://huggingface.co/LoneStriker/DeepSeek-Coder-V2-Instruct-GGUF/tree/main) |
| DeepseekV2-chat   | [DeepSeek-V2-Chat-gguf-Q4K-M](https://huggingface.co/bullerwins/DeepSeek-V2-Chat-0628-GGUF/tree/main)             |
| DeepseekV2-lite   | [DeepSeek-V2-Lite-Chat-GGUF-Q4K-M](https://huggingface.co/mzwing/DeepSeek-V2-Lite-Chat-GGUF/tree/main)            |
| DeepSeek-R1       | [DeepSeek-R1-gguf-Q4K-M](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M)            |

</details>

<!-- pin block for jump -->

<span id='id_666'>

<h3>RESTful API 和 Web UI</h3>

不带网站启动：

```sh
ktransformers --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path /path/to/DeepSeek-V2-Lite-Chat-GGUF --port 10002
```

带网站启动：

```sh
ktransformers --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path /path/to/DeepSeek-V2-Lite-Chat-GGUF  --port 10002 --web True
```

或者如果您想使用 transformers 启动服务器，model_path 应包含 safetensors

```bash
ktransformers --type transformers --model_path /mnt/data/model/Qwen2-0.5B-Instruct --port 10002 --web True
```

通过 [http://localhost:10002/web/index.html#/chat](http://localhost:10002/web/index.html#/chat) 访问网站：

<p align="center">
  <picture>
    <img alt="Web UI" src="https://github.com/user-attachments/assets/615dca9b-a08c-4183-bbd3-ad1362680faf" width=90%>
  </picture>
</p>

关于 RESTful API 服务器的更多信息可以在[这里](../en/api/server/server.md)找到。您还可以在[这里](../en/api/server/tabby.md)找到与 Tabby 集成的示例。 