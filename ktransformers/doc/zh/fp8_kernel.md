# DeepSeek-V3/R1的FP8线性内核

## 概述
DeepSeek-AI团队为DeepSeek-R1/V3模型提供了FP8 safetensors。我们通过以下工作实现性能优化：
- **FP8 GPU内核集成**：FP8线性层加速内核集成在KTransformers中
- **混合量化架构**：
  - 注意力和共享专家模块使用FP8精度（提高计算精度）
  - 专家模块保留GGML量化（GGUF格式，驻留在CPU中以节省GPU内存）

因此，那些追求最佳性能的用户可以使用DeepSeek-V3/R1的FP8线性内核。

## 主要特点

✅ 混合精度架构（FP8 + GGML）<br>
✅ 内存优化（~19GB VRAM使用量）

## 快速开始
### 使用预合并权重

预合并权重可在Hugging Face上获取：<br>
[KVCache-ai/DeepSeek-V3-GGML-FP8-Hybrid](https://huggingface.co/KVCache-ai/DeepSeek-V3)<br>
[KVCache-ai/DeepSeek-R1-GGML-FP8-Hybrid](https://huggingface.co/KVCache-ai/DeepSeek-R1)

> 下载前请确认权重已完全上传。由于文件较大，Hugging Face上传时间可能会延长。


下载预合并权重
```shell
pip install -U huggingface_hub

# 可选：在特定地区使用HF镜像以加快下载速度
# export HF_ENDPOINT=https://hf-mirror.com 

huggingface-cli download --resume-download KVCache-ai/DeepSeek-V3-GGML-FP8-Hybrid --local-dir <本地目录>
```
### 使用合并脚本
如果您有本地DeepSeek-R1/V3 fp8 safetensors和gguf权重（例如q4km），可以使用以下脚本合并它们。

```shell
python merge_tensors/merge_safetensor_gguf.py \
  --safetensor_path <fp8_safetensor路径> \
  --gguf_path <gguf文件夹路径> \
  --output_path <合并输出路径>
```

* `--safetensor_path`：safetensor文件的输入路径（[下载](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main)）。
* `--gguf_path`：gguf文件夹的输入路径（[下载](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q4_K_M)）。
* `--output_path`：合并文件的输出路径。


### 执行说明

使用自定义量化专家启动local_chat.py
```shell
python ktransformers/local_chat.py \
  --model_path deepseek-ai/DeepSeek-V3 \
  --gguf_path <合并权重文件夹> \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-fp8-linear-ggml-experts.yaml \
  --cpu_infer <cpu核心数 + 1>
```


## 注意事项

⚠️ 硬件要求<br>
* FP8内核推荐最低19GB可用VRAM。
* 需要支持FP8的GPU（例如，4090）

⏳ 首次运行优化
JIT编译导致首次执行时间较长（后续运行保持优化速度）。

🔄 临时接口<br>
当前权重加载实现是临时性的 - 将在未来版本中改进

📁 路径指定<br>
尽管使用混合量化，合并权重仍存储为.safetensors - 将包含文件夹的路径传递给`--gguf_path` 