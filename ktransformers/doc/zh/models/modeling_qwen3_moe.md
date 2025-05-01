# Qwen3Moe模型实现

## 概述
本文档提供了`modeling_qwen3_moe.py`文件中重要注释的中文翻译版本，帮助中文用户更好地理解Qwen3 MoE模型的实现细节。

## 原始文件注释中文翻译

```python
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           此文件是从src/transformers/models/qwen3_moe/modular_qwen3_moe.py自动生成的。
#               请勿手动编辑此文件，因为任何编辑都将在生成文件时被覆盖。
#             如果需要进行任何更改，请直接在modular_qwen3_moe.py文件中应用更改。我们的CI会强制执行这一点。
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
```

### 重要函数注释

```python
def rotate_half(x):
    """旋转输入的一半隐藏维度。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """将旋转位置嵌入应用于查询和键张量。

    参数:
        q (`torch.Tensor`): 查询张量。
        k (`torch.Tensor`): 键张量。
        cos (`torch.Tensor`): 旋转嵌入的余弦部分。
        sin (`torch.Tensor`): 旋转嵌入的正弦部分。
        position_ids (`torch.Tensor`, *可选*):
            已弃用且未使用。
        unsqueeze_dim (`int`, *可选*, 默认为1):
            'unsqueeze_dim'参数指定了沿哪个维度对cos[position_ids]和sin[position_ids]进行扩展，
            使它们能够正确地广播到q和k的维度。例如，注意cos[position_ids]和sin[position_ids]的形状为
            [batch_size, seq_len, head_dim]。然后，如果q和k的形状为[batch_size, heads, seq_len, head_dim]，
            设置unsqueeze_dim=1使cos[position_ids]和sin[position_ids]可以广播到q和k的形状。
            类似地，如果q和k的形状为[batch_size, seq_len, heads, head_dim]，则设置unsqueeze_dim=2。
    返回:
        包含使用旋转位置嵌入旋转的查询和键张量的`tuple(torch.Tensor)`。
    """


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这等同于torch.repeat_interleave(x, dim=1, repeats=n_rep)。隐藏状态从(batch, num_key_value_heads, seqlen, head_dim)
    变为(batch, num_attention_heads, seqlen, head_dim)
    """


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """标准注意力机制的前向传播实现"""


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    计算Switch Transformer中的辅助负载平衡损失 - 用PyTorch实现。

    参见Switch Transformer(https://arxiv.org/abs/2101.03961)了解更多详情。这个函数实现了论文中方程(4)-(6)
    中呈现的损失函数。它旨在惩罚专家之间路由不平衡的情况。

    参数:
        gate_logits:
            来自`gate`的logits，应该是一个model.config.num_hidden_layers张量的元组，
            形状为[batch_size X sequence_length, num_experts]。
        num_experts:
            专家数量
        top_k:
            每个token路由的专家数量，也可以解释为`top-k`路由参数。
        attention_mask (`torch.Tensor`, *可选*):
            在前向函数中使用的attention_mask，
            形状为[batch_size X sequence_length]（如果不为None）。

    返回:
        辅助损失。
    """
```

### 类注释

```python
class Qwen3MoeAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力"""


class Qwen3MoeMLP(nn.Module):
    """标准的MLP实现"""


class Qwen3MoeSparseMoeBlock(nn.Module):
    """稀疏混合专家块实现"""


class Qwen3MoeRMSNorm(nn.Module):
    """
    Qwen3MoeRMSNorm等同于T5LayerNorm
    """


class Qwen3MoeDecoderLayer(nn.Module):
    """单个Qwen3Moe解码器层实现"""


class Qwen3MoeRotaryEmbedding(nn.Module):
    """旋转位置嵌入实现"""


class Qwen3MoePreTrainedModel(PreTrainedModel):
    """
    这个模型继承自[`PreTrainedModel`]。查看父类文档以了解库为所有模型实现的通用方法
    (例如下载或保存、调整输入嵌入大小、修剪头部等)

    这个模型也是PyTorch的[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)子类。
    将其作为常规PyTorch模块使用，并参考PyTorch文档了解所有与一般用法和行为相关的事项。

    参数:
        config ([`Qwen3MoeConfig`]):
            模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型关联的权重，只加载配置。
            查看[`~PreTrainedModel.from_pretrained`]方法加载模型权重。
    """


class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    """
    由*config.num_hidden_layers*层组成的Transformer解码器。每一层都是[`Qwen3MoeDecoderLayer`]

    参数:
        config: Qwen3MoeConfig
    """


class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
    """
    用于因果语言建模的Qwen3Moe模型
    """


class Qwen3MoeForSequenceClassification(Qwen3MoePreTrainedModel):
    """
    带有序列分类头的Qwen3Moe模型transformer(顶部的线性层)。

    [`Qwen3MoeForSequenceClassification`]使用最后一个token进行分类，就像其他因果模型(例如GPT-2)一样。

    由于它对最后一个token进行分类，它需要知道最后一个token的位置。如果在配置中定义了`pad_token_id`，
    它会在每一行中找到不是填充token的最后一个token。如果没有定义`pad_token_id`，它简单地取每一行
    批次中的最后一个值。由于当传递`inputs_embeds`而不是`input_ids`时它无法猜测填充tokens，
    它会做同样的事情(取批次中每一行的最后一个值)。
    """


class Qwen3MoeForTokenClassification(Qwen3MoePreTrainedModel):
    """
    带有token分类头的Qwen3Moe模型transformer(顶部隐藏状态输出上的线性层)，例如用于命名实体识别(NER)任务。
    """


class Qwen3MoeForQuestionAnswering(Qwen3MoePreTrainedModel):
    """
    带有跨度分类头的Qwen3Moe模型transformer，用于提取式问答任务，如SQuAD(顶部隐藏状态输出上的线性层，
    用于计算`跨度开始logits`和`跨度结束logits`)。
    """
```

### 输入文档字符串 (DOCSTRING)

```python
QWEN3_MOE_INPUTS_DOCSTRING = r"""
    参数:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            词汇表中输入序列tokens的索引。默认情况下，你提供的填充将被忽略。

            索引可以使用[`AutoTokenizer`]获取。参见[`PreTrainedTokenizer.encode`]和
            [`PreTrainedTokenizer.__call__`]获取详细信息。

            [什么是input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *可选*):
            避免在填充token索引上执行注意力的掩码。掩码值在`[0, 1]`中选择:

            - 1表示**未被遮蔽**的tokens,
            - 0表示**被遮蔽**的tokens。

            [什么是attention masks?](../glossary#attention-mask)

            索引可以使用[`AutoTokenizer`]获取。参见[`PreTrainedTokenizer.encode`]和
            [`PreTrainedTokenizer.__call__`]获取详细信息。

            如果使用了`past_key_values`，可选择性地只输入最后的`input_ids`(参见`past_key_values`)。

            如果你想更改填充行为，你应该阅读[`modeling_opt._prepare_decoder_attention_mask`]
            并按你的需要修改。参见[论文](https://arxiv.org/abs/1910.13461)中的图1，
            了解有关默认策略的更多信息。

            - 1表示头部**未被遮蔽**,
            - 0表示头部**被遮蔽**。
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *可选*):
            每个输入序列token在位置嵌入中的索引。在`[0, config.n_positions - 1]`范围内选择。

            [什么是position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *可选*):
            预先计算好的隐藏状态(自注意力块和交叉注意力块中的key和value)，可用于加速顺序解码。
            这通常包括在`use_cache=True`或`config.use_cache=True`时由模型在上一阶段解码返回的`past_key_values`。

            它是一个[`~cache_utils.Cache`]实例。更多详情，请参见我们的[kv cache指南](https://huggingface.co/docs/transformers/en/kv_cache)。

            如果使用了`past_key_values`，用户可以选择性地只输入最后的`input_ids`(那些没有给该模型的past key value states的输入)，
            形状为`(batch_size, 1)`，而不是所有`input_ids`的形状`(batch_size, sequence_length)`。
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *可选*):
            可选择直接传递嵌入表示，而不是传递`input_ids`。如果你想对如何将`input_ids`索引转换为相关向量
            有更多控制，这很有用，而不是使用模型的内部嵌入查找矩阵。
        use_cache (`bool`, *可选*):
            如果设置为`True`，则返回`past_key_values` key value状态，可用于加速解码(参见`past_key_values`)。
        output_attentions (`bool`, *可选*):
            是否返回所有注意力层的注意力张量。有关详细信息，请参见返回的张量下的`attentions`。
        output_hidden_states (`bool`, *可选*):
            是否返回所有层的隐藏状态。有关详细信息，请参见返回的张量下的`hidden_states`。
        return_dict (`bool`, *可选*):
            是否返回[`~utils.ModelOutput`]而不是普通元组。
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *可选*):
            描述输入序列tokens在序列中的位置的索引。与`position_ids`相反，此张量不受填充影响。
            它用于在正确位置更新缓存并推断完整的序列长度。
"""
```

### 生成方法文档说明

```python
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    # **kwargs: Unpack[KwargsForCausalLM],
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *可选*):
            用于计算掩码语言建模损失的标签。索引应该在`[0, ..., config.vocab_size]`范围内，
            或者为-100(参见`input_ids`文档字符串)。索引设置为`-100`的tokens将被忽略(掩码)，
            只有标签在`[0, ..., config.vocab_size]`范围内的tokens才会计算损失。

        logits_to_keep (`int`或`torch.Tensor`, *可选*):
            如果是`int`，为最后`logits_to_keep`个tokens计算logits。如果是`0`，为所有`input_ids`计算logits(特殊情况)。
            对于生成，只需要最后一个token的logits，只为该token计算logits可以节省内存，
            这对于长序列或大词汇量来说变得非常显著。
            如果是`torch.Tensor`，必须是一维的，对应于序列长度维度中要保留的索引。
            当使用打包张量格式(批次和序列长度的单一维度)时，这很有用。

    返回:

    示例:

    ```python
    >>> from transformers import AutoTokenizer, Qwen3MoeForCausalLM

    >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # 生成
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
``` 