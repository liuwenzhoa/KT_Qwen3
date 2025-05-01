# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3MoE模型配置"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen3MoeConfig(PretrainedConfig):
    r"""
    这是[`Qwen3MoeModel`]的配置类，用于存储模型配置。它用于根据指定参数实例化Qwen3MoE模型，定义模型架构。
    使用默认参数实例化配置将产生类似于[Qwen/Qwen3-MoE-15B-A2B](https://huggingface.co/Qwen/Qwen3-15B-A2B)的配置。
    
    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。更多信息请参阅[`PretrainedConfig`]的文档。
    
    参数:
        vocab_size (`int`, *可选*, 默认为151936):
            Qwen3MoE模型的词汇表大小。定义了在调用[`Qwen3MoeModel`]时通过`inputs_ids`表示的不同标记数量。
        hidden_size (`int`, *可选*, 默认为2048):
            隐藏表示的维度。
        intermediate_size (`int`, *可选*, 默认为6144):
            MLP表示的维度。
        num_hidden_layers (`int`, *可选*, 默认为24):
            Transformer编码器中的隐藏层数。
        num_attention_heads (`int`, *可选*, 默认为32):
            Transformer编码器中每个注意力层的注意力头数。
        num_key_value_heads (`int`, *可选*, 默认为4):
            用于实现分组查询注意力(GQA)的key_value头数。如果`num_key_value_heads=num_attention_heads`，
            模型将使用多头注意力(MHA)；如果`num_key_value_heads=1`，模型将使用多查询注意力(MQA)；
            否则使用GQA。将多头检查点转换为GQA检查点时，每个组的key和value头应通过对该组内所有原始头进行平均池化来构建。
            更多详情请查看[此论文](https://arxiv.org/pdf/2305.13245.pdf)。如果未指定，将默认为`32`。
        hidden_act (`str`或`function`, *可选*, 默认为`"silu"`):
            解码器中的非线性激活函数(函数或字符串)。
        max_position_embeddings (`int`, *可选*, 默认为32768):
            此模型可能使用的最大序列长度。
        initializer_range (`float`, *可选*, 默认为0.02):
            用于初始化所有权重矩阵的truncated_normal_initializer的标准差。
        rms_norm_eps (`float`, *可选*, 默认为1e-06):
            rms归一化层使用的epsilon值。
        use_cache (`bool`, *可选*, 默认为`True`):
            模型是否应返回最后的key/values注意力(并非所有模型都使用)。仅当`config.is_decoder=True`时相关。
        tie_word_embeddings (`bool`, *可选*, 默认为`False`):
            模型的输入和输出词嵌入是否应该绑定。
        rope_theta (`float`, *可选*, 默认为10000.0):
            RoPE嵌入的基本周期。
        rope_scaling (`Dict`, *可选*):
            包含RoPE嵌入缩放配置的字典。注意：如果应用新的rope类型并期望模型在更长的`max_position_embeddings`上工作，
            我们建议相应地更新此值。
            
            预期内容:
                `rope_type` (`str`):
                    要使用的RoPE子变体。可以是['default', 'linear', 'dynamic', 'yarn', 'longrope', 'llama3']之一，
                    'default'是原始RoPE实现。
                `factor` (`float`, *可选*):
                    用于除'default'外的所有rope类型。应用于RoPE嵌入的缩放因子。在大多数缩放类型中，
                    因子x将使模型能够处理长度为x * 原始最大预训练长度的序列。
                `original_max_position_embeddings` (`int`, *可选*):
                    用于'dynamic', 'longrope'和'llama3'。预训练期间使用的原始最大位置嵌入。
                `attention_factor` (`float`, *可选*):
                    用于'yarn'和'longrope'。应用于注意力计算的缩放因子。如果未指定，
                    它默认为实现推荐的值，使用`factor`字段推断建议值。
                `beta_fast` (`float`, *可选*):
                    仅用于'yarn'。参数设置线性斜坡函数中(仅)外推的边界。如果未指定，默认为32。
                `beta_slow` (`float`, *可选*):
                    仅用于'yarn'。参数设置线性斜坡函数中(仅)插值的边界。如果未指定，默认为1。
                `short_factor` (`List[float]`, *可选*):
                    仅用于'longrope'。应用于短上下文(<`original_max_position_embeddings`)的缩放因子。
                    必须是长度等于隐藏大小除以注意力头数除以2的数字列表。
                `long_factor` (`List[float]`, *可选*):
                    仅用于'longrope'。应用于长上下文(<`original_max_position_embeddings`)的缩放因子。
                    必须是长度等于隐藏大小除以注意力头数除以2的数字列表。
                `low_freq_factor` (`float`, *可选*):
                    仅用于'llama3'。应用于RoPE低频分量的缩放因子。
                `high_freq_factor` (`float`, *可选*):
                    仅用于'llama3'。应用于RoPE高频分量的缩放因子。
        attention_bias (`bool`, 默认为`False`, *可选*, 默认为`False`):
            在自注意力期间是否在查询、键、值和输出投影层中使用偏置。
        use_sliding_window (`bool`, *可选*, 默认为`False`):
            是否使用滑动窗口注意力。
        sliding_window (`int`, *可选*, 默认为4096):
            滑动窗口注意力(SWA)窗口大小。如果未指定，将默认为`4096`。
        max_window_layers (`int`, *可选*, 默认为28):
            使用SWA(滑动窗口注意力)的层数。底层使用SWA，而顶层使用全注意力。
        attention_dropout (`float`, *可选*, 默认为0.0):
            注意力概率的丢弃率。
        decoder_sparse_step (`int`, *可选*, 默认为1):
            MoE层的频率。
        moe_intermediate_size (`int`, *可选*, 默认为768):
            路由专家的中间大小。
        num_experts_per_tok (`int`, *可选*, 默认为8):
            选定的专家数量。
        num_experts (`int`, *可选*, 默认为128):
            路由专家的数量。
        norm_topk_prob (`bool`, *可选*, 默认为`False`):
            是否对topk概率进行归一化。
        output_router_logits (`bool`, *可选*, 默认为`False`):
            模型是否应返回路由器logits。启用此功能还将允许模型输出辅助损失，包括负载平衡损失和路由器z损失。
        router_aux_loss_coef (`float`, *可选*, 默认为0.001):
            总损失的辅助损失因子。
        mlp_only_layers (`List[int]`, *可选*, 默认为`[]`):
            指示哪些层使用Qwen3MoeMLP而不是Qwen3MoeSparseMoeBlock。
            如果我们有num_layers层，列表包含从0到num_layers-1的层索引。
            如果`mlp_only_layers`为空，则使用`decoder_sparse_step`确定稀疏性。
            
    ```python
    >>> from transformers import Qwen3MoeModel, Qwen3MoeConfig
    >>> # 初始化Qwen3MoE风格的配置
    >>> configuration = Qwen3MoeConfig()
    >>> # 从Qwen3-15B-A2B风格配置初始化模型
    >>> model = Qwen3MoeModel(configuration)
    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    # 基础模型`Qwen3Moe`的默认张量并行计划
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # 验证旋转位置嵌入参数的正确性
        # BC: 如果存在'type'字段，将其移至'rope_type'。
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # MoE参数
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3MoeConfig"] 