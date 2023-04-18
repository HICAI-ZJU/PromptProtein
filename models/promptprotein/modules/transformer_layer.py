import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from .multihead_attention import MultiheadAttention
from .utils import gelu

class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self._init_submodules(add_bias_kv)

    def _init_submodules(self, add_bias_kv):
        BertLayerNorm = LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

        self.layer_gated = nn.Linear(self.embed_dim, 1, bias=False)

    def forward(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False, with_prompt_num=0
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        if with_prompt_num != 0:
            gate = self.layer_gated(F.normalize(x[-with_prompt_num:, :, :].clone(), dim=2)).mean()
            x[:-with_prompt_num, :, :] = (1 - gate) * x[:-with_prompt_num, :, :].clone()
        x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn