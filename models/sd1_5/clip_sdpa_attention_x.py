import torch    
import torch.nn as nn
from typing import Optional, Tuple

from transformers.models.clip.modeling_clip import CLIPAttention

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_2
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)






class CLIPSdpaAttentionX(CLIPAttention):
    """
    SDPA attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `CLIPAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, config):
        super().__init__(config)
        self.dummy = 0
        
    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype

        # if self.upcast_attention:
        #     query = query.float()
        #     key = key.float()




        ###equivalent to head_to_batch_dim in processor:
        query = query.squeeze()
        key = key.squeeze()
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze()
        # print(key.size())
        # exit() 
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        # if self.upcast_softmax:

        #     attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    # Adapted from CLIPAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Match Transformers' current CLIPAttention calculation exactly, while
        # storing the attention probabilities needed by T-SAM.
        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask
        is_causal = kwargs.get("is_causal", False)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        elif is_causal:
            query_length, key_length = attention_scores.shape[-2:]
            causal_mask = torch.ones(
                query_length,
                key_length,
                device=attention_scores.device,
                dtype=torch.bool,
            ).triu(1)
            attention_scores = attention_scores.masked_fill(
                causal_mask, torch.finfo(attention_scores.dtype).min
            )
        attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_probs = nn.functional.dropout(attention_probs, p=0.0 if not self.training else self.dropout, training=self.training)

        if attention_probs.size(-1) in [120, 77] and self.dummy == 0:
            self.attn_data_x = torch.mean(attention_probs, dim=1).squeeze(0)
        self.dummy = 1

        attn_output = torch.matmul(attention_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attention_probs if output_attentions else None
    
    
