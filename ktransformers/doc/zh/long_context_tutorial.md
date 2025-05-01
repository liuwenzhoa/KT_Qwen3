## 如何使用ktransformers长上下文框架

目前，长上下文功能仅由我们的**local_chat.py**接口支持，与服务器接口的集成正在开发中。

为方便用户管理，我们已将模型配置、gguf和tokenizer上传到一个仓库。URL：https://huggingface.co/nilv234/internlm2_5_to_llama_1m/tree/main

通过将local_chat函数中的model_path和gguf_path设置为**/path/to/repo**，并将模式设置为**"long_context"**，您可以在24G VRAM上使用具有1m功能的InternLM2.5-7B-Chat-1M模型。

首次运行local_chat.py后，将在**~/.ktransformers**下自动创建一个config.yaml文件。长上下文的相关配置如下：

```python
chunk_size: 4096 # 预填充块大小
max_seq_len: 100000 # KVCache长度
block_size: 128 # KVCache块大小
local_windows_len: 4096 # 长度为local_windows_len的KVCache存储在GPU上
second_select_num: 96 # 预选后每次选择的KVCache块数量。如果 >= preselect_block_count，则使用预选的块
threads_num: 64 # CPU线程数
anchor_type: DYNAMIC # KVCache块代表性token选择方法
kv_type: FP16
dense_layer_num: 0 # 前几层不需要填充或选择KVCache
anchor_num: 1 # KVCache块内代表性token的数量
preselect_block: False # 是否预选
head_select_mode: SHARED # 所有kv_heads联合选择
preselect_block_count: 96 # 预选的块数量
layer_step: 1 # 每隔几层选择一次
token_step: 1 # 每隔几个token选择一次
```

不同上下文长度所需的内存如下表所示：

|                | 4K  | 32K  | 64K  | 128K | 512K | 1M     |
| -------------- | --- | ---- | ---- | ---- | ---- | ------ |
| DRAM大小 (GB)  | 0.5 | 4.29 | 8.58 | 17.1 | 68.7 | 145.49 |

请根据您的DRAM大小选择合适的max_seq_len。
例如：
```python
python local_chat.py --model_path="/data/model/internlm2_5_to_llama_1m"  --gguf_path="/data/model/internlm2_5_to_llama_1m" --max_new_tokens=500 --cpu_infer=10  --use_cuda_graph=True  --mode="long_context" --prompt_file="/path/to/file"
```

如果您已经通过prompt_file指定了输入文本，当终端显示chat:时，只需按Enter即可开始。 