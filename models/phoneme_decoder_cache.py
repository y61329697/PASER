"""
moudule: phoneme_decoder_cache.py
function: add kv-cache mechanism for phoneme decoder to improve inference speed, based on pytorch standard transformer decoder
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional
from torch.nn.modules.transformer import _get_seq_len, _detect_is_causal_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CachedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    This code is based on PyTorch's TransformerDecoderLayer class, with added caching mechanism to improve inference speed
    """
    def __init__(self, *args, **kwargs):
        super(CachedTransformerDecoderLayer, self).__init__(*args, **kwargs)
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        kv_cache: Optional[list] = None,
        return_kv_cache: bool = False,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is None:
            kv_cache = [None, None]

        if kv_cache[0] is None:
            kv_cache[0] = tgt.new_zeros(0, tgt.shape[1], self.self_attn.embed_dim)
        if kv_cache[1] is None:
            kv_cache[1] = tgt.new_zeros(0, tgt.shape[1], self.self_attn.embed_dim)

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal, kv_cache)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal, kv_cache))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))
        
        if return_kv_cache:
            return x, kv_cache
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool, kv_cache: list) -> Tensor:
        if kv_cache[0] is None:
            kv_cache[0] = x.new_zeros(0, x.shape[1], self.self_attn.embed_dim)
        if kv_cache[1] is None:
            kv_cache[1] = x.new_zeros(0, x.shape[1], self.self_attn.embed_dim)
            
        kv_cache[0] = torch.cat((kv_cache[0], x), dim=0)
        kv_cache[1] = torch.cat((kv_cache[1], x), dim=0)
        
        tgt2, _ = self.self_attn(x, kv_cache[0], kv_cache[1], attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask, is_causal=is_causal, need_weights=False)

        return self.dropout1(tgt2)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
class CachedTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, *args, **kwargs):
        super(CachedTransformerDecoder, self).__init__(*args, **kwargs)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False, kv_cache=None, return_kv_cache=False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

            kv_cache: the cached key-value pairs for each layer (optional).
            return_kv_cache: whether to return the new key-value cache (optional).

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        new_kv_cache = []
        for i, mod in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            if return_kv_cache:
                output, new_layer_kv_cache = mod(output, memory, tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask,
                                                 tgt_is_causal=tgt_is_causal,
                                                 memory_is_causal=memory_is_causal,
                                                 kv_cache=layer_kv_cache,
                                                 return_kv_cache=return_kv_cache)
                new_kv_cache.append(new_layer_kv_cache)
            else:
                output = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             tgt_is_causal=tgt_is_causal,
                             memory_is_causal=memory_is_causal,
                             kv_cache=layer_kv_cache,
                             return_kv_cache=return_kv_cache)

        if self.norm is not None:
            output = self.norm(output)

        if return_kv_cache:
            return output, new_kv_cache
        return output

class PhonemeDecoder(nn.Module):
    def __init__(self, phoneme_vocab_size=75, d_model=768, num_decoder_layers=2, nhead=12, dim_feedforward=768, dropout=0.1):
        super(PhonemeDecoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(phoneme_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        decoder_layer = CachedTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = CachedTransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, phoneme_vocab_size)
        
    def forward(self, tgt_input, memory, tgt_mask=None, memory_key_padding_mask=None, kv_cache=None, return_kv_cache=False):
        
        tgt_input = tgt_input.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        tgt_emb = self.embedding(tgt_input) * math.sqrt(self.d_model)
        
        if kv_cache is None or kv_cache[0] is None:
            tgt_emb = self.pos_encoder(tgt_emb)
        else:
            zero_tensor = torch.zeros_like(kv_cache[0][0])
            # when using cache mechanism, the structure of current decoding and position encoding need to be aligned
            tgt_emb_tmp = self.pos_encoder(torch.concat([zero_tensor, tgt_emb], dim=0))
            tgt_emb = tgt_emb_tmp[-tgt_input.size(0), :, :].unsqueeze(0)

        if return_kv_cache:
            decoder_output, kv_cache = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, 
                                                                memory_key_padding_mask=memory_key_padding_mask,
                                                                kv_cache=kv_cache,
                                                                return_kv_cache=return_kv_cache)
        else:
            decoder_output = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, 
                                                      memory_key_padding_mask=memory_key_padding_mask,
                                                      kv_cache=kv_cache,
                                                      return_kv_cache=return_kv_cache)
        
        output = self.fc_out(decoder_output)
        output = output.transpose(0, 1)
        decoder_output = decoder_output.transpose(0, 1)
        
        if return_kv_cache:
            return output, decoder_output, kv_cache
        return output, decoder_output

def predict_batch(model, memory, batch_size, start_token=2, end_token=3, max_length=100, vocab_size=75, memory_mask=None):
    device = memory.device
    tgt_input = torch.full((batch_size, 1), start_token, dtype=torch.long).to(device)
    output_sequences = [[start_token] for _ in range(batch_size)]
    sequence_lengths = [1] * batch_size
    finished = [False] * batch_size

    kv_cache = [None] * len(model.transformer_decoder.layers)

    for _ in range(max_length):
        output, _, kv_cache = model(tgt_input, memory, memory_key_padding_mask=memory_mask, kv_cache=kv_cache, return_kv_cache=True)
        
        next_token_logits = output[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        for i, next_token in enumerate(next_tokens):
            if not finished[i]:
                output_sequences[i].append(next_token.item())
                sequence_lengths[i] += 1
                if next_token == end_token:
                    finished[i] = True
        
        if all(finished):
            break

        next_token_tensor = next_tokens.unsqueeze(1)
        tgt_input = next_token_tensor

    max_seq_length = max(sequence_lengths)

    padded_output_sequences = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(device)
    for i, seq in enumerate(output_sequences):
        padded_output_sequences[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long).to(device)
    return padded_output_sequences, sequence_lengths_tensor
