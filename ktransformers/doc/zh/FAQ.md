<!-- omit in toc -->
# 常见问题
- [安装问题](#安装问题)
  - [问题：ImportError: /lib/x86\_64-linux-gnu/libstdc++.so.6: version GLIBCXX\_3.4.32' not found](#问题-importerror-libx86_64-linux-gnulibstdcso6-version-glibcxx_3432-not-found)
  - [问题：DeepSeek-R1没有输出初始<think>标记](#问题-deepseek-r1没有输出初始think标记)
- [使用问题](#使用问题)
  - [问题：如果我的显存超过模型需求，如何充分利用？](#问题如果我的显存超过模型需求如何充分利用)
  - [问题：如果显存不足但有多个GPU，如何利用它们？](#问题如果显存不足但有多个gpu如何利用它们)
  - [问题：如何获得最佳性能？](#问题如何获得最佳性能)
  - [问题：我的DeepSeek-R1模型没有思考过程。](#问题我的deepseek-r1模型没有思考过程)
  - [问题：加载gguf文件出错](#问题加载gguf文件出错)
  - [问题：Version \`GLIBCXX\_3.4.30' not found](#问题-version-glibcxx_3430-not-found)
  - [问题：运行bfloat16 MoE模型时，数据显示NaN](#问题运行bfloat16-moe模型时数据显示nan)
  - [问题：使用fp8预填充非常慢。](#问题使用fp8预填充非常慢)
  - [问题：如何在使用Volta和Turing架构的显卡上运行](#问题如何在使用volta和turing架构的显卡上运行)

## 安装问题
### 问题：ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.32' not found
```
在Ubuntu 22.04安装时需要添加：
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install --only-upgrade libstdc++6
```
来源：https://github.com/kvcache-ai/ktransformers/issues/117#issuecomment-2647542979

### 问题：DeepSeek-R1没有输出初始<think>标记

> 来自deepseek-R1文档：<br>
> 此外，我们观察到DeepSeek-R1系列模型在响应某些查询时倾向于绕过思考模式（即输出"\<think>\n\n\</think>"），这可能会对模型的性能产生不利影响。为确保模型进行充分推理，我们建议强制模型在每次输出开始时以"\<think>\n"开头。

因此，我们通过在提示词末尾手动添加"\<think>\n"标记来解决这个问题（您可以在local_chat.py中查看），并传递参数`--force_think true`可以让local_chat以"\<think>\n"开始响应。

来源：https://github.com/kvcache-ai/ktransformers/issues/129#issue-2842799552

## 使用问题
### 问题：如果我的显存超过模型需求，如何充分利用？

1. 扩大上下文窗口。
   1. local_chat.py：您可以通过设置`--max_new_tokens`为更大的值来增加上下文窗口大小。
   2. 服务器：增加`--cache_lens`的值。
2. 将更多权重移至GPU。
    参考ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu-4.yaml
    ```yaml
    - match:
       name: "^model\\.layers\\.([4-10])\\.mlp\\.experts$" # 在第4~10层注入marlin专家
     replace:
       class: ktransformers.operators.experts.KTransformersExperts  
       kwargs:
         generate_device: "cuda:0" # 在cuda:0上运行；marlin仅支持GPU
         generate_op:  "KExpertsMarlin" # 使用marlin专家
     recursive: False
    ```
    您可以根据需要修改层，例如将`name: "^model\\.layers\\.([4-10])\\.mlp\\.experts$"`改为`name: "^model\\.layers\\.([4-12])\\.mlp\\.experts$"`，以将更多权重移至GPU。

    > 注意：yaml中首先匹配的规则将被应用。例如，如果有两条规则匹配同一层，只有第一条规则的替换会生效。
    > 注意：当前，在GPU上执行专家会与CUDA Graph冲突。没有CUDA Graph，会有明显的性能下降。因此，除非您有大量显存（将DeepSeek-V3/R1的单层专家放在GPU上至少需要5.6GB显存），否则我们不建议启用此功能。我们正在积极优化。
    > 注意：KExpertsTorch未经测试。

### 问题：如果显存不足但有多个GPU，如何利用它们？

使用`--optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml`加载优化规则yaml文件。您也可以以此为例，编写自己的4/8 GPU优化规则yaml文件。

> 注意：ktransformers的多GPU策略是流水线，不能加速模型推理。它只用于模型权重分配。

### 问题：如何获得最佳性能？

您需要将`--cpu_infer`设置为希望使用的核心数。使用的核心越多，模型运行越快。但这并不是越多越好。将其调整为略低于您实际核心数的值。

### 问题：我的DeepSeek-R1模型没有思考过程。

根据DeepSeek的建议，您需要通过传递参数`--force_think True`来强制模型在每次输出开始时以"\<think>\n"开头。

### 问题：加载gguf文件出错

请确保：
1. 在`--gguf_path`目录中有`gguf`文件。
2. 该目录只包含来自一个模型的gguf文件。如有多个模型，需将它们分到不同目录。
3. 文件夹名称本身不应以`.gguf`结尾，例如`Deep-gguf`是正确的，而`Deep.gguf`是错误的。
4. 文件本身未损坏；您可以通过检查sha256sum与huggingface、modelscope或hf-mirror上的值是否匹配来验证。

### 问题：Version `GLIBCXX_3.4.30' not found
详细错误：
>ImportError: /mnt/data/miniconda3/envs/xxx/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/xxx/xxx/ktransformers/./cpuinfer_ext.cpython-312-x86_64-linux-gnu.so)

运行`conda install -c conda-forge libstdcxx-ng`可以解决此问题。

### 问题：运行bfloat16 MoE模型时，数据显示NaN
详细错误：
```shell
Traceback (most recent call last):
  File "/root/ktransformers/ktransformers/local_chat.py", line 183, in <module>
    fire.Fire(local_chat)
  File "/usr/local/lib/python3.10/dist-packages/fire/core.py", line 135, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/usr/local/lib/python3.10/dist-packages/fire/core.py", line 468, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "/usr/local/lib/python3.10/dist-packages/fire/core.py", line 684, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "/root/ktransformers/ktransformers/local_chat.py", line 177, in local_chat
    generated = prefill_and_generate(
  File "/root/ktransformers/ktransformers/util/utils.py", line 204, in prefill_and_generate
    next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, use_cuda_graph).to(torch_device)
  File "/root/ktransformers/ktransformers/util/utils.py", line 128, in decode_one_tokens
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```
**解决方案**：在Ubuntu 22.04上运行ktransformers的问题是由于当前系统的g++版本太旧，预定义宏不包括avx_bf16。我们已测试并确认在Ubuntu 22.04的g++ 11.4上可以正常工作。

### 问题：使用fp8预填充非常慢。

FP8内核是通过JIT构建的，因此首次运行会很慢。后续运行会更快。

### 问题：如何在使用Volta和Turing架构的显卡上运行

来源：https://github.com/kvcache-ai/ktransformers/issues/374

1. 首先，使用git下载最新源代码。
2. 然后，修改源代码中的DeepSeek-V3-Chat-multi-gpu-4.yaml和所有相关yaml文件，将所有KLinearMarlin实例替换为KLinearTorch。
3. 接下来，从ktransformers源代码编译，直到在本地机器上成功编译。
4. 然后，安装flash-attn。虽然不会使用它，但不安装会导致错误。
5. 然后，修改local_chat.py，将所有flash_attention_2实例替换为eager。
6. 然后，运行local_chat.py。确保遵循官方教程的命令，并根据本地机器的参数进行调整。
7. 运行过程中，检查内存使用情况。通过top命令观察其调用情况。单个CPU上的内存容量必须大于模型的完整大小。（对于多个CPU，只是一个副本。）
最后，确认模型已完全加载到内存中，特定权重层已完全加载到GPU内存中。然后，尝试在聊天界面中输入内容，观察是否有任何错误。

注意，为获得更好的性能，您可以查看问题中的[此方法](https://github.com/kvcache-ai/ktransformers/issues/374#issuecomment-2667520838)
>
>https://github.com/kvcache-ai/ktransformers/blob/89f8218a2ab7ff82fa54dbfe30df741c574317fc/ktransformers/operators/attention.py#L274-L279
>
>```diff
>+ original_dtype = query_states.dtype
>+ target_dtype = torch.half
>+ query_states = query_states.to(target_dtype)
>+ compressed_kv_with_k_pe = compressed_kv_with_k_pe.to(target_dtype)
>+ compressed_kv = compressed_kv.to(target_dtype)
>+ attn_output = attn_output.to(target_dtype)
>
>decode_attention_fwd_grouped(query_states, compressed_kv_with_k_pe, compressed_kv, attn_output,
>                             page_table,
>                             position_ids.squeeze(0).to(torch.int32)+1, attn_logits,
>                             4, #num_kv_splits # follow vLLM, fix it TODO
>                             self.softmax_scale,
>                             past_key_value.page_size)
>
>+ attn_output = attn_output.to(original_dtype)
>```
>
>https://github.com/kvcache-ai/ktransformers/blob/89f8218a2ab7ff82fa54dbfe30df741c574317fc/ktransformers/operators/attention.py#L320-L326
>
>```diff
>- attn_output = flash_attn_func( 
>-     query_states, 
>-     key_states, 
>-     value_states_padded, 
>-     softmax_scale=self.softmax_scale, 
>-     causal=True, 
>- )
>+ attn_output = F.scaled_dot_product_attention(
>+     query_states.transpose(1, 2),
>+     key_states.transpose(1, 2),
>+     value_states_padded.transpose(1, 2),
>+     scale=self.softmax_scale,
>+     is_causal=True
>+ ).transpose(1, 2)
>