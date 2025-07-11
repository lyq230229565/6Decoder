"""
    The TransformerDecoder implementation is derived from and modified based on the torch.nn.transformer code.
"""




# mypy: allow-untyped-defs
import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList


__all__ = ['TransformerDecoder', 'TransformerDecoderLayer']

def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]



class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt)
    """

    __constants__ = ['norm']

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 num_layers: int = 6, dropout: float = 0.1, norm: Optional[nn.Module] = None,
                 batch_first: bool = False, norm_first: bool = False) -> None:
        super().__init__()
        
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                        dropout=dropout, batch_first=batch_first, norm_first=norm_first,
                                        activation=activation)
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: Optional[bool] = None, ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn1.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_is_causal=tgt_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt)
    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        super().__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
                如果设置tgt_is_causal=True，那么就无需输入atten_mask，MultiheadAttention会自动生成倒三角的因果掩码
                如果想自己输入特别的atten_mask，那么就设置tgt_is_causal=False

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt

        if self.norm_first:
            # 1. Self-attention layer 1
            x = x + self._sa1_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            # 2. Self-attention layer 2
            x = x + self._sa2_block(self.norm2(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            # 3. Feed-forward layer
            x = x + self._ff_block(self.norm3(x))
        else:        
            # 1. Self-attention layer 1
            x = self.norm1(x + self._sa1_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            # 2. Self-attention layer 2
            x = self.norm2(x + self._sa2_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            # 3. Feed-forward layer
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa1_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn1(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _sa2_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn2(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            is_causal=is_causal,
                            need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal
